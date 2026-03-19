"""Pi0-FAST Implicit CoT (CODI-style) model.

Architecture Overview
---------------------
Student path: [img_tokens | prefix_tokens | latent_1 | ... | latent_K | action_tokens]
  - K latent tokens replace explicit CoT tokens
  - Each latent_i is generated sequentially using KV cache
  - Produces student CE loss on action tokens

Teacher path: [img_tokens | prefix_tokens | CoT_tokens | action_tokens]
  - Existing Pi0-FAST explicit CoT behavior (teacher forcing)
  - Produces teacher CE loss on CoT+action tokens
  - Provides teacher hidden state at action boundary for distillation

Decoder path: for step i, given latent_i embedding, predict CoT_section_i tokens
  - Produces decoder (explain) loss per CoT section
  - Uses a SEPARATE decoder LLM (self.decoder_llm) following CODI design
  - Decoder parameters are updated independently during training
  - Decoder weights are saved as part of the Pi0FASTImplicit checkpoint

Distillation: align student hidden state after last latent with teacher hidden state
  at action boundary using SmoothL1.

Checkpoint notes
----------------
Pi0FASTImplicit checkpoints contain both the main VLA (self.PaliGemma) and the
separate decoder (self.decoder_llm).  When initialising from a Pi0FAST base
checkpoint (which has no decoder_llm key), the decoder is automatically
bootstrapped from the main LLM weights so training starts from a sensible point.
"""

import openpi.models.gemma_fast as _gemma
import dataclasses
import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from openpi.models import model as _model
from openpi.models.pi0_fast import PALIGEMMA_EOS_TOKEN, Pi0FAST, Pi0FASTConfig, left_to_right_align, make_attn_mask, put_along_last_axis
from openpi.shared import array_typing as at
from typing_extensions import override

logger = logging.getLogger("openpi")


@dataclasses.dataclass(frozen=True)
class Pi0FASTImplicitConfig(Pi0FASTConfig):
    """Configuration for Pi0-FAST-Implicit (CODI-style) model."""

    # Number of latent tokens (= number of CoT sections)
    num_latent: int = 9

    # Token budget for prefix-only sequence (no CoT, no actions)
    max_prefix_len: int = 80

    # Token budget for FAST action tokens only
    max_action_token_len: int = 80

    # Token budget per CoT section
    max_cot_step_len: int = 100

    # Loss weights
    distill_loss_factor: float = 1.0
    explain_loss_factor: float = 1.0
    ref_loss_factor: float = 1.0

    # Distillation loss type: "smooth_l1" or "mse"
    distill_loss_type: str = "smooth_l1"

    # Whether to enable decoder (explain) loss
    use_decoder: bool = True

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI0_FAST_IMPLICIT

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0FASTImplicit":
        return Pi0FASTImplicit(self, rngs=nnx.Rngs(rng))

    @override
    def load(self, params: at.Params, *, remove_extra_params: bool = True) -> "Pi0FASTImplicit":
        """Load model, bootstrapping decoder_llm from PaliGemma.llm if not in checkpoint.

        When loading from a Pi0FAST base checkpoint (no decoder_llm key), the
        decoder is initialised with the same weights as the main LLM so that
        training starts from a sensible warm-start point — exactly what CODI
        does when it passes the same model path for both student and decoder.

        When loading from a Pi0FASTImplicit checkpoint the decoder_llm key is
        already present, so no bootstrap is applied.
        """
        model = nnx.eval_shape(self.create, jax.random.key(0))
        graphdef, state = nnx.split(model)
        state_dict = state.to_pure_dict()

        # Bootstrap decoder_llm from PaliGemma.llm if this is a base checkpoint.
        if "decoder_llm" not in params:
            pg_llm_params = params.get("PaliGemma", {}).get("llm", {})
            # Shallow-copy the top-level params dict so we don't mutate the caller's tree,
            # then assign the decoder key (JAX arrays are immutable so sharing is safe).
            params = dict(params)
            params["decoder_llm"] = pg_llm_params

        if remove_extra_params:
            params = ocp.transform_utils.intersect_trees(state_dict, params)

        at.check_pytree_equality(expected=state_dict, got=params, check_shapes=True, check_dtypes=False)
        state.replace_by_pure_dict(params)
        return nnx.merge(graphdef, state)

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "base_1_rgb": image_spec,
                    "wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "base_1_rgb": image_mask_spec,
                    "wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                # Teacher path fields
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                token_ar_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                token_loss_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.bool_),
                # Student path fields
                tokenized_prefix=jax.ShapeDtypeStruct([batch_size, self.max_prefix_len], jnp.int32),
                tokenized_prefix_mask=jax.ShapeDtypeStruct([batch_size, self.max_prefix_len], jnp.bool_),
                prefix_ar_mask=jax.ShapeDtypeStruct([batch_size, self.max_prefix_len], jnp.int32),
                tokenized_action_tokens=jax.ShapeDtypeStruct(
                    [batch_size, self.max_action_token_len], jnp.int32
                ),
                tokenized_action_mask=jax.ShapeDtypeStruct(
                    [batch_size, self.max_action_token_len], jnp.bool_
                ),
                cot_step_tokens=jax.ShapeDtypeStruct(
                    [batch_size, self.num_latent, self.max_cot_step_len], jnp.int32
                ),
                cot_step_masks=jax.ShapeDtypeStruct(
                    [batch_size, self.num_latent, self.max_cot_step_len], jnp.bool_
                ),
                ref_answer_position=jax.ShapeDtypeStruct([batch_size], jnp.int32),
            )
        action_spec = jax.ShapeDtypeStruct(
            [batch_size, self.action_horizon, self.action_dim], jnp.float32
        )
        return observation_spec, action_spec


class Pi0FASTImplicit(Pi0FAST):
    """Pi0-FAST with Implicit CoT (CODI-style) training.

    Inherits PaliGemma, embed_inputs from Pi0FAST.
    Overrides compute_loss and sample_actions.
    """

    def __init__(self, config: Pi0FASTImplicitConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs=rngs)
        self._implicit_config = config

        # Separate decoder LLM (following CODI design: independent parameters,
        # trained alongside the main VLA to predict CoT text from latent tokens).
        # Uses the same Gemma architecture as self.PaliGemma.llm so that
        # dimensions always match and no projection layers are needed.
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        decoder_llm = nnx_bridge.ToNNX(
            _gemma.Module(
                **paligemma_config,
                embed_dtype=config.dtype,
                cache_dtype=config.dtype,
            )
        )
        decoder_llm.lazy_init(rngs=rngs, method="init")
        self.decoder_llm = decoder_llm

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _embed_prefix_only(
        self, obs: _model.Observation
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Int[at.Array, "b s"],
    ]:
        """Embed images + prefix-only tokens (no CoT, no actions).

        Returns:
            - embeddings [B, img_len + max_prefix_len, D]
            - full_mask [B, img_len + max_prefix_len]  (valid token mask)
            - full_ar_mask [B, img_len + max_prefix_len]  (all zeros = bidirectional)
        """
        token_embeddings = []
        input_mask = []

        # Embed images
        for name in obs.images:
            image_token_embeddings, _ = self.PaliGemma.img(obs.images[name], train=False)
            token_embeddings.append(image_token_embeddings)
            input_mask.append(
                einops.repeat(obs.image_masks[name], "b -> b s", s=image_token_embeddings.shape[1])
            )

        # Embed prefix-only tokens
        assert obs.tokenized_prefix is not None, "tokenized_prefix required for implicit CoT"
        assert obs.tokenized_prefix_mask is not None
        prefix_embeds = self.PaliGemma.llm(obs.tokenized_prefix, embed_only=True)
        token_embeddings.append(prefix_embeds)
        input_mask.append(obs.tokenized_prefix_mask)

        full_embeds = jnp.concatenate(token_embeddings, axis=1)
        full_mask = jnp.concatenate(input_mask, axis=1)
        full_ar_mask = jnp.zeros_like(full_mask, dtype=jnp.int32)  # all bidirectional

        return full_embeds, full_mask, full_ar_mask

    def _compute_decoder_loss(
        self,
        latent_embd: at.Float[at.Array, "b 1 d"],
        step_tokens: at.Int[at.Array, "b sl"],
        step_mask: at.Bool[at.Array, "b sl"],
    ) -> at.Float[at.Array, " b"]:
        """Compute decoder (explain) loss for one CoT section.

        The decoder sees [latent_embd | step_tokens[:, :-1]] and predicts step_tokens.

        Args:
            latent_embd: Latent hidden state embedding, shape [B, 1, D].
            step_tokens: Token ids for this CoT section, shape [B, step_len].
            step_mask: Valid token mask, shape [B, step_len].

        Returns:
            Per-batch CE loss averaged over valid positions, shape [B].
        """
        B, step_len = step_tokens.shape
        vocab_size = self.decoder_llm.module.vocab_size

        # Embed step tokens using the decoder's own embedding table
        step_embeds = self.decoder_llm(step_tokens, embed_only=True)  # [B, step_len, D]

        # Concatenate: [latent | step_tokens[:-1]] -> input
        # [latent | step_tokens[0], ..., step_tokens[step_len-2]] -> predict step_tokens[0..step_len-1]
        decoder_input = jnp.concatenate([latent_embd, step_embeds[:, :-1]], axis=1)  # [B, step_len, D]

        # Standard causal forward pass through the separate decoder LLM
        decoder_pre_logits, _, _ = self.decoder_llm(
            embedded_prefix=decoder_input,
            return_prelogits=True,
        )  # [B, step_len, D]

        # Decode logits
        decoder_logits, _ = self.decoder_llm(
            pre_logits=decoder_pre_logits,
        )  # [B, step_len, vocab_size]

        # Compute CE loss
        targets = jax.nn.one_hot(step_tokens, vocab_size)  # [B, step_len, V]
        logp = jax.nn.log_softmax(decoder_logits, axis=-1)
        token_pplx = jnp.sum(targets * logp, axis=-1)  # [B, step_len]

        # Average over valid positions (per batch element)
        valid_count = jnp.sum(step_mask.astype(jnp.float32), axis=-1)  # [B]
        loss = -jnp.sum(token_pplx * step_mask, axis=-1) / jnp.clip(valid_count, 1.0)  # [B]
        return loss

    def _decode_section_tokens(
        self,
        context_embd: at.Float[at.Array, "b 1 d"],
        max_steps: int,
    ) -> np.ndarray:
        """Greedily decode text tokens from a latent context embedding.

        Uses the separately trained ``self.decoder_llm`` (same architecture as
        the main VLA backbone, but with independent parameters) to map from the
        main model's latent space back into readable text — mirroring what the
        decoder was trained to do via ``_compute_decoder_loss``.

        NOTE: This method uses a Python for-loop and is NOT JIT-compatible.
        It is intended for offline evaluation / debugging only.

        Decoder logic (matches training):
          - Position 0: context_embd (latent hidden state from main VLA)
          - Position t: embedding of token_{t-1} (via decoder_llm embed table)
          - Output at position t predicts token_t

        Args:
            context_embd: Latent hidden state [B, 1, D], the pre_logits output
                from latent step i in the main VLA's latent loop.
            max_steps: Maximum number of tokens to generate.

        Returns:
            np.ndarray [B, max_steps] of int32 token ids, zero-padded after EOS.
        """
        B = context_embd.shape[0]
        all_tokens: list[np.ndarray] = []
        current_embeds = context_embd  # [B, seq_len, D], grows each step
        finished = np.zeros(B, dtype=bool)

        for _ in range(max_steps):
            # Full causal forward pass on all accumulated embeddings
            pre_logits, _, _ = self.decoder_llm(
                embedded_prefix=current_embeds,
                return_prelogits=True,
            )  # [B, seq_len, D]

            # Logit at last position predicts the next token
            logit, _ = self.decoder_llm(
                pre_logits=pre_logits[:, -1:, :]
            )  # [B, 1, vocab_size]

            # Greedy argmax; bring to numpy
            token_t = np.array(jnp.argmax(logit[:, 0, :], axis=-1))  # [B]
            # Zero-out already-finished sequences
            token_t = np.where(finished, 0, token_t)
            all_tokens.append(token_t.copy())

            # Mark sequences that just hit EOS
            finished = finished | (token_t == PALIGEMMA_EOS_TOKEN)
            if np.all(finished):
                # Pad remaining slots with zeros and exit early
                while len(all_tokens) < max_steps:
                    all_tokens.append(np.zeros(B, dtype=np.int32))
                break

            # Embed the new token using the decoder's own embedding table
            token_emb = self.decoder_llm(
                jnp.array(token_t[:, None], dtype=jnp.int32), embed_only=True
            )  # [B, 1, D]
            current_embeds = jnp.concatenate([current_embeds, token_emb], axis=1)

        # Final zero-pad if max_steps was reached without all EOS
        while len(all_tokens) < max_steps:
            all_tokens.append(np.zeros(B, dtype=np.int32))

        return np.stack(all_tokens, axis=1)  # [B, max_steps]

    # ------------------------------------------------------------------
    # compute_loss
    # ------------------------------------------------------------------

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> tuple[at.Float[at.Array, "*b"], dict[str, at.Float[at.Array, ""]]]:
        cfg = self._implicit_config
        obs = _model.preprocess_observation(
            rng, observation, train=train, image_keys=list(observation.images.keys())
        )
        B = obs.state.shape[0]

        # ================================================================
        # STEP 1: TEACHER PATH (full tokenized_prompt with CoT + actions)
        # ================================================================
        input_embeds, input_mask, ar_mask = self.embed_inputs(obs)
        T = input_embeds.shape[1]  # img_len + max_token_len
        attn_mask = make_attn_mask(input_mask, ar_mask)

        targets_teacher = jax.nn.one_hot(
            obs.tokenized_prompt[:, 1:], self.PaliGemma.llm.module.vocab_size
        )

        teacher_pre_logits, _, _ = self.PaliGemma.llm(
            embedded_prefix=input_embeds[:, :-1],
            mask=attn_mask[:, :-1, :-1],
            return_prelogits=True,
        )  # [B, T-1, D]

        teacher_logits, _ = self.PaliGemma.llm(
            pre_logits=teacher_pre_logits[:, -targets_teacher.shape[1]:],
        )  # [B, T-1, V]

        teacher_loss_mask = obs.token_loss_mask[:, 1:]  # [B, T-1]
        logp_teacher = jax.nn.log_softmax(teacher_logits, axis=-1)
        token_pplx_teacher = jnp.sum(targets_teacher * logp_teacher, axis=-1)  # [B, T-1]
        teacher_ce_loss = -jnp.sum(token_pplx_teacher * teacher_loss_mask, axis=-1) / jnp.clip(
            jnp.sum(teacher_loss_mask, axis=-1), 1.0
        )  # [B]

        # Extract teacher hidden state at action boundary:
        # img_len = T - max_token_len (number of image tokens)
        img_len = T - self.max_token_len
        # ref_answer_position = prefix_len + cot_len (in tokenized_prompt)
        # teacher_h = hidden state just before first action token
        # = teacher_pre_logits[:, img_len + ref_answer_position - 1, :]
        teacher_h_idx = jnp.clip(
            img_len + obs.ref_answer_position - 1, 0, teacher_pre_logits.shape[1] - 1
        )  # [B]
        D = teacher_pre_logits.shape[-1]
        idx_t_expanded = jnp.broadcast_to(teacher_h_idx[:, None, None], [B, 1, D])
        teacher_h = jnp.take_along_axis(teacher_pre_logits, idx_t_expanded, axis=1)[:, 0, :]  # [B, D]

        # ================================================================
        # STEP 2: STUDENT PATH — Prefix Prefill + Latent Loop
        # ================================================================
        prefix_embeds, prefix_full_mask, _ = self._embed_prefix_only(obs)
        # prefix_embeds: [B, img_len + max_prefix_len, D]
        # prefix_full_mask: [B, img_len + max_prefix_len]

        total_cache_size = img_len + cfg.max_prefix_len + cfg.num_latent + cfg.max_action_token_len

        # Build prefix attention mask padded to total_cache_size
        prefix_attn_mask_base = make_attn_mask(
            prefix_full_mask, jnp.zeros_like(prefix_full_mask, dtype=jnp.int32)
        )  # [B, img_len+max_prefix_len, img_len+max_prefix_len]
        prefix_attn_mask = jnp.pad(
            prefix_attn_mask_base[:, None, :, :],
            ((0, 0), (0, 0), (0, 0), (0, total_cache_size - prefix_embeds.shape[1])),
        )  # [B, 1, img_len+max_prefix_len, total_cache_size]

        # Positions for prefix
        prefix_positions = jnp.cumsum(prefix_full_mask.astype(jnp.int32), axis=-1) - 1  # [B, T_prefix]

        # Prefill prefix into KV cache
        prefix_pre_logits, kv_cache, _ = self.PaliGemma.llm(
            embedded_prefix=prefix_embeds,
            mask=prefix_attn_mask,
            positions=prefix_positions,
            decode=True,
            return_prelogits=True,
        )  # [B, img_len+max_prefix_len, D]

        # Extract last valid hidden state as initial latent embedding
        prefix_text_valid_len = jnp.sum(obs.tokenized_prefix_mask.astype(jnp.int32), axis=-1)  # [B]
        last_prefix_idx = jnp.clip(img_len + prefix_text_valid_len - 1, 0, prefix_pre_logits.shape[1] - 1)  # [B]
        idx_p_expanded = jnp.broadcast_to(last_prefix_idx[:, None, None], [B, 1, D])
        latent_embd = jnp.take_along_axis(prefix_pre_logits, idx_p_expanded, axis=1)  # [B, 1, D]

        # ================================================================
        # STEP 2b: LATENT LOOP via jax.lax.scan
        # Replaces the Python for-loop to compile the body ONCE and let XLA
        # use step-boundary checkpointing during the backward pass, reducing
        # activation memory from O(K) to O(1) for the loop body.
        # ================================================================
        remaining_size = cfg.num_latent + cfg.max_action_token_len

        # Pre-compute positions for all K steps: [K, B, 1]
        step_offsets = jnp.arange(cfg.num_latent)[:, None, None]  # [K, 1, 1]
        positions_all = (img_len + prefix_text_valid_len)[:, None][None, ...] + step_offsets  # [K, B, 1]

        # Pre-compute attention masks for all K steps: [K, B, 1, 1, total_cache_size]
        # prev_latent_masks[i, col] = True iff col < i  (attend only to earlier latents)
        prev_latent_masks = (
            jnp.arange(remaining_size)[None, :] < jnp.arange(cfg.num_latent)[:, None]
        )  # [K, R]
        masks_all = jnp.concatenate([
            jnp.broadcast_to(
                jnp.ones([1, B, 1, 1, img_len], dtype=jnp.bool_),
                [cfg.num_latent, B, 1, 1, img_len],
            ),
            jnp.broadcast_to(
                obs.tokenized_prefix_mask[None, :, None, None, :],
                [cfg.num_latent, B, 1, 1, cfg.max_prefix_len],
            ),
            jnp.broadcast_to(
                prev_latent_masks[:, None, None, None, :],
                [cfg.num_latent, B, 1, 1, remaining_size],
            ),
        ], axis=-1)  # [K, B, 1, 1, total_cache_size]

        # CoT tokens/masks transposed from [B, K, L] → [K, B, L] for scan xs
        xs_step_tokens = jnp.transpose(obs.cot_step_tokens, (1, 0, 2))  # [K, B, L]
        xs_step_masks  = jnp.transpose(obs.cot_step_masks,  (1, 0, 2))  # [K, B, L]

        def latent_step(carry, xs):
            latent_embd_c, kv_cache_c = carry
            positions_i, mask_i, step_tokens_i, step_mask_i = xs

            # One decode step — pre_logits_i is the OUTPUT (= latent_i)
            pre_logits_i, kv_cache_new, _ = self.PaliGemma.llm(
                embedded_prefix=latent_embd_c,
                mask=mask_i,
                positions=positions_i,
                decode=True,
                kv_cache=kv_cache_c,
                return_prelogits=True,
            )  # [B, 1, D]

            # Decoder (explain) loss: reconstruct CoT section_i from latent_i.
            # cfg.use_decoder is a static Python bool — resolved at trace time.
            if cfg.use_decoder:
                explain_loss_i = self._compute_decoder_loss(pre_logits_i, step_tokens_i, step_mask_i)
            else:
                explain_loss_i = jnp.zeros([B])

            return (pre_logits_i, kv_cache_new), (pre_logits_i, explain_loss_i)

        xs = (positions_all, masks_all, xs_step_tokens, xs_step_masks)
        (latent_embd, kv_cache), (all_pre_logits, all_explain_losses) = jax.lax.scan(
            latent_step, (latent_embd, kv_cache), xs, length=cfg.num_latent
        )
        # all_pre_logits:     [K, B, 1, D]
        # all_explain_losses: [K, B]

        # Student hidden state (after last latent)
        student_hidden = latent_embd[:, 0, :]  # [B, D]

        # ================================================================
        # STEP 3: STUDENT ACTION CE LOSS
        # Build a full forward pass using collected latent pre_logits.
        # Input: [prefix_embeds | all_K_latent_pre_logits | action_embeds[:, :-1]]
        # This avoids the KV cache multi-token limitation for action tokens.
        # ================================================================
        all_latent_embeds = jnp.transpose(all_pre_logits[:, :, 0, :], (1, 0, 2))  # [B, K, D]

        # Embed action tokens (all but last)
        al = cfg.max_action_token_len
        action_embeds_full = self.PaliGemma.llm(obs.tokenized_action_tokens, embed_only=True)  # [B, al, D]
        action_embeds_input = action_embeds_full[:, :-1, :]  # [B, al-1, D]

        # Concatenate student sequence: [prefix_embeds | latents | action[:, :-1]]
        student_input = jnp.concatenate([prefix_embeds, all_latent_embeds, action_embeds_input], axis=1)
        # Shape: [B, img_len+max_prefix_len+K+al-1, D]

        # Build attention mask for student action forward pass
        # Sequence: [images | prefix | latents | action[:al-1]]
        # - images: bidirectional
        # - prefix: bidirectional to images+prefix
        # - latents: causal among themselves, attend to all prefix
        # - action token j: causal, attends to all prefix+latents + actions 0..j
        img_mask_sa = jnp.ones([B, img_len], dtype=jnp.bool_)
        prefix_mask_sa = obs.tokenized_prefix_mask  # [B, max_prefix_len]
        latent_mask_sa = jnp.ones([B, cfg.num_latent], dtype=jnp.bool_)  # all K latents valid
        action_mask_sa = obs.tokenized_action_mask[:, :-1]  # [B, al-1]

        # Combined input mask and AR mask for make_attn_mask
        sa_input_mask = jnp.concatenate(
            [img_mask_sa, prefix_mask_sa, latent_mask_sa, action_mask_sa], axis=1
        )  # [B, student_input_len]

        # AR mask: 0 for images+prefix (bidirectional), 1 for latents+actions (causal)
        sa_ar_mask = jnp.concatenate(
            [
                jnp.zeros([B, img_len + cfg.max_prefix_len], dtype=jnp.int32),
                jnp.ones([B, cfg.num_latent + al - 1], dtype=jnp.int32),
            ],
            axis=1,
        )  # [B, student_input_len]

        student_attn_mask = make_attn_mask(sa_input_mask, sa_ar_mask)  # [B, student_input_len, student_input_len]

        # Forward pass for student action prediction
        student_pre_logits, _, _ = self.PaliGemma.llm(
            embedded_prefix=student_input,
            mask=student_attn_mask,
            return_prelogits=True,
        )  # [B, student_input_len, D]

        # Logits for action tokens: last al-1 positions in output predict action tokens
        action_logit_range = cfg.num_latent + al - 1  # position range for action token logits
        action_pre_logits = student_pre_logits[:, -action_logit_range:, :]  # [B, K+al-1, D]
        # We only need the action portion (skip latent pre_logits here)
        action_only_pre_logits = action_pre_logits[:, cfg.num_latent:, :]  # [B, al-1, D]

        student_action_logits, _ = self.PaliGemma.llm(
            pre_logits=action_only_pre_logits,
        )  # [B, al-1, V]

        # Targets for student action CE loss: action_tokens[:, 1:]
        action_targets = jax.nn.one_hot(
            obs.tokenized_action_tokens[:, 1:], self.PaliGemma.llm.module.vocab_size
        )  # [B, al-1, V]
        action_loss_mask = obs.tokenized_action_mask[:, 1:]  # [B, al-1]

        logp_student = jax.nn.log_softmax(student_action_logits, axis=-1)
        token_pplx_student = jnp.sum(action_targets * logp_student, axis=-1)  # [B, al-1]
        student_ce_loss = -jnp.sum(token_pplx_student * action_loss_mask, axis=-1) / jnp.clip(
            jnp.sum(action_loss_mask, axis=-1), 1.0
        )  # [B]

        # ================================================================
        # STEP 4: DISTILLATION LOSS
        # ================================================================
        if cfg.distill_loss_type == "smooth_l1":
            diff = student_hidden - jax.lax.stop_gradient(teacher_h)
            distill_loss = jnp.mean(
                jnp.where(jnp.abs(diff) < 1.0, 0.5 * diff**2, jnp.abs(diff) - 0.5),
                axis=-1,
            )  # [B]
        else:
            diff = student_hidden - jax.lax.stop_gradient(teacher_h)
            distill_loss = jnp.mean(diff**2, axis=-1)  # [B]

        # ================================================================
        # STEP 5: EXPLAIN (DECODER) LOSS
        # ================================================================
        if cfg.use_decoder:
            explain_loss = jnp.mean(all_explain_losses, axis=0)  # [B], average over K steps
        else:
            explain_loss = jnp.zeros([B])

        # ================================================================
        # STEP 6: TOTAL LOSS
        # ================================================================
        total_loss = (
            student_ce_loss
            + cfg.ref_loss_factor * teacher_ce_loss
            + cfg.distill_loss_factor * distill_loss
            + cfg.explain_loss_factor * explain_loss
        )

        aux = {
            "ce_loss": jnp.mean(student_ce_loss),
            "teacher_ce_loss": jnp.mean(teacher_ce_loss),
            "distill_loss": jnp.mean(distill_loss),
            "explain_loss": jnp.mean(explain_loss),
        }
        return total_loss, aux

    # ------------------------------------------------------------------
    # sample_actions
    # ------------------------------------------------------------------

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        max_decoding_steps: int | at.Int[at.Array, ""] = 256,
        temperature: float = 0.0,
        return_reasoning: bool = False,
    ) -> "_model.Actions | tuple[_model.Actions, np.ndarray]":
        """Generate actions using implicit CoT (K latent steps + autoregressive decode).

        Args:
            rng: Random key.
            observation: Model observation. Must contain tokenized_prefix /
                tokenized_prefix_mask for the student path.
            max_decoding_steps: Max action tokens to generate.
            temperature: Sampling temperature (0 = greedy).
            return_reasoning: If True, also decode each of the K latent tokens
                back into readable text using the shared LLM decoder, and return
                ``(actions, reasoning_tokens)``.

                reasoning_tokens is np.ndarray [B, K, max_cot_step_len] of int32
                token ids (zero-padded after the first EOS per section).  Pass
                these through ``FASTTokenizer._paligemma_tokenizer.decode()`` to
                get human-readable strings.

                IMPORTANT: when return_reasoning=True this function runs a Python
                for-loop per section and is NOT JIT-compilable.  Use it for offline
                evaluation only, not for real-time robot control.

        Returns:
            actions [B, max_decoding_steps] int32 when return_reasoning=False.
            (actions, reasoning_tokens) when return_reasoning=True.

        Decoder weight note:
            The decoder uses ``self.decoder_llm``, a SEPARATE model with
            independent parameters that was trained alongside the main VLA.
            Both the main VLA and decoder weights are saved in every
            Pi0FASTImplicit checkpoint — no separate decoder checkpoint file
            is needed.  The decoder is automatically bootstrapped from the
            main LLM weights when loading from a Pi0FAST base checkpoint.
        """
        cfg = self._implicit_config
        obs = _model.preprocess_observation(
            None, observation, train=False, image_keys=list(observation.images.keys())
        )
        B = obs.state.shape[0]

        # === Embed prefix only ===
        prefix_embeds, prefix_full_mask, _ = self._embed_prefix_only(obs)
        prefix_attn_mask_base = make_attn_mask(
            prefix_full_mask, jnp.zeros_like(prefix_full_mask, dtype=jnp.int32)
        )

        # Right-align for inference (valid tokens at end)
        prefix_embeds, prefix_full_mask, prefix_attn_mask_base = left_to_right_align(
            prefix_embeds, prefix_full_mask, prefix_attn_mask_base
        )
        prefill_size = prefix_embeds.shape[1]
        prefix_valid_len = jnp.sum(prefix_full_mask.astype(jnp.int32), axis=-1)  # [B]
        prefix_start = prefill_size - prefix_valid_len  # [B]

        # Pad to total_cache_size for KV cache
        total_cache_size = prefill_size + cfg.num_latent + max_decoding_steps
        prefix_attn_mask = jnp.pad(
            prefix_attn_mask_base[:, None, :, :],
            ((0, 0), (0, 0), (0, 0), (0, total_cache_size - prefill_size)),
        )

        # Positions for prefix
        prefix_positions = jnp.cumsum(prefix_full_mask.astype(jnp.int32), axis=-1) - 1

        # Prefill prefix into KV cache
        prefix_pre_logits, kv_cache, _ = self.PaliGemma.llm(
            embedded_prefix=prefix_embeds,
            mask=prefix_attn_mask,
            positions=prefix_positions,
            decode=True,
            return_prelogits=True,
        )  # [B, prefill_size, D]

        # Last valid hidden state becomes the first latent input
        latent_embd = prefix_pre_logits[:, -1:, :]  # [B, 1, D]

        # === K latent generation steps ===
        # decoder_contexts[i] = pre_logits_i (OUTPUT of step i = latent_i),
        # so the decoder reconstructs section_i from latent_i (matches training).
        decoder_contexts: list[at.Array] = []
        for i in range(cfg.num_latent):
            positions_i = (prefix_valid_len + i)[:, None]  # [B, 1]
            mask_i = jnp.logical_and(
                jnp.arange(total_cache_size)[None, None, None, :] >= prefix_start[:, None, None, None],
                jnp.arange(total_cache_size)[None, None, None, :] < (prefix_valid_len + i + 1)[:, None, None, None],
            )

            pre_logits_i, kv_cache, _ = self.PaliGemma.llm(
                embedded_prefix=latent_embd,
                mask=mask_i,
                positions=positions_i,
                decode=True,
                kv_cache=kv_cache,
                return_prelogits=True,
            )  # [B, 1, D]

            latent_embd = pre_logits_i  # update for next step

            # Capture OUTPUT of step i for decoding section_i text (matches training).
            if return_reasoning:
                decoder_contexts.append(pre_logits_i)

        # === Optional: decode each latent context → text tokens (eval mode) ===
        # This section uses Python loops and is NOT JIT-compilable.
        reasoning_tokens: np.ndarray | None = None
        if return_reasoning:
            section_arrays = []
            for ctx in decoder_contexts:
                # ctx is the decoder context for this section (latent_{i-1} output)
                toks = self._decode_section_tokens(ctx, cfg.max_cot_step_len)  # [B, max_cot_step_len]
                section_arrays.append(toks)
            reasoning_tokens = np.stack(section_arrays, axis=1)  # [B, K, max_cot_step_len]

        # === Autoregressive action decode ===
        # Seed the while_loop with the first logit from the last latent hidden state
        last_logit, _ = self.PaliGemma.llm(pre_logits=latent_embd)  # [B, 1, vocab_size]
        output_tokens = jnp.zeros((B, max_decoding_steps), dtype=jnp.int32)

        def step(carry):
            rng, last_logit, output_tokens, cache, finished, step_idx = carry

            rng, rng_step = jax.random.split(rng)
            sampled_token = jax.lax.cond(
                temperature > 0.0,
                lambda _: jax.random.categorical(rng_step, last_logit / temperature, axis=-1),
                lambda _: jnp.argmax(last_logit, axis=-1),
                operand=None,
            )
            # Once a sequence emits EOS, keep subsequent slots padded with zeros
            # instead of continuing to sample arbitrary tokens for that batch row.
            token = jnp.where(finished[:, None], 0, sampled_token)
            output_tokens = put_along_last_axis(
                output_tokens,
                jnp.broadcast_to(step_idx, (token.shape[0], 1)),
                token,
            )
            finished = finished | jnp.any(token == PALIGEMMA_EOS_TOKEN, axis=-1)
            token_embedding = self.PaliGemma.llm(token, embed_only=True)
            positions = (prefix_valid_len + cfg.num_latent)[:, None] + step_idx + 1
            mask = jnp.logical_and(
                jnp.arange(total_cache_size)[None, None, :]
                >= prefix_start[:, None, None],
                jnp.arange(total_cache_size)[None, None, :]
                < (prefix_valid_len + cfg.num_latent + step_idx + 2)[:, None, None],
            )
            last_logit, cache, _ = self.PaliGemma.llm(
                embedded_prefix=token_embedding,
                mask=mask,
                positions=positions,
                decode=True,
                kv_cache=cache,
            )
            return rng, last_logit, output_tokens, cache, finished, step_idx + 1

        def cond(carry):
            _, _, _, _, finished, step_idx = carry
            return (~jnp.all(finished)) & (step_idx < max_decoding_steps)

        _, _, output_tokens, _, _, _ = jax.lax.while_loop(
            cond, step, (rng, last_logit, output_tokens, kv_cache, jnp.zeros((B,), dtype=jnp.bool_), 0)
        )

        if return_reasoning:
            return output_tokens, reasoning_tokens
        return output_tokens
