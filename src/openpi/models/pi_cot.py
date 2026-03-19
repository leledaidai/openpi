"""PiCOT: Hybrid VLA combining Pi0.5-style flow matching with Pi0FAST-style FAST action tokenization.

Architecture:
- Dual-stream transformer: PaliGemma LLM (stream 0) + action expert with adaRMS (stream 1)
- Text stream: text prefix → CoT reasoning tokens → FAST action tokens (all causal)
- Action stream: noisy continuous actions conditioned on prefix + CoT only (NOT FAST)

Training: single joint forward pass computing two losses:
  1. Joint AR cross-entropy loss over [CoT | FAST] tokens (teacher forcing, single normalisation)
  2. Flow-matching loss on continuous actions, weighted by flow_matching_loss_weight

  total_loss = ar_loss + flow_matching_loss_weight * flow_loss

  The action expert (stream-1) is blocked from attending to FAST tokens in the attention mask
  to prevent leaking the target action encoding into the denoising stream.

Inference: autoregressive CoT → autoregressive FAST tokens → 10-step flow matching denoising
"""

import dataclasses
import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
import openpi.shared.download as _download

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Tokens can attend to valid input tokens with cumulative mask_ar <= theirs."""
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


def posemb_sincos(pos, embedding_dim, min_period, max_period):
    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij", pos, 1.0 / period * 2 * jnp.pi, precision=jax.lax.Precision.HIGHEST
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


@dataclasses.dataclass(frozen=True)
class PiCOTConfig(_model.BaseModelConfig):
    """Configuration for the PiCOT model."""

    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Action dimensions (must match base class fields)
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = 200  # L: text prefix token budget

    # CoT token budget
    max_cot_tokens: int = 512  # C

    # FAST action token budget
    max_fast_tokens: int = 64  # F

    # Path to FAST tokenizer (HuggingFace repo or local path)
    fast_tokenizer_path: str = "physical-intelligence/fast"

    # Weight applied to the flow-matching loss term.
    # The AR loss (CoT + FAST tokens jointly) always has weight 1.0.
    flow_matching_loss_weight: float = 1.0

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI_COT

    @override
    def create(self, rng: at.KeyArrayLike) -> "PiCOT":
        return PiCOT(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1):
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={k: image_spec for k in _model.IMAGE_KEYS},
                image_masks={k: image_mask_spec for k in _model.IMAGE_KEYS},
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                tokenized_cot_reasoning=jax.ShapeDtypeStruct(
                    [batch_size, self.max_cot_tokens], jnp.int32
                ),
                tokenized_cot_reasoning_mask=jax.ShapeDtypeStruct(
                    [batch_size, self.max_cot_tokens], bool
                ),
                tokenized_fast_actions=jax.ShapeDtypeStruct(
                    [batch_size, self.max_fast_tokens], jnp.int32
                ),
                tokenized_fast_actions_mask=jax.ShapeDtypeStruct(
                    [batch_size, self.max_fast_tokens], bool
                ),
            )
        action_spec = jax.ShapeDtypeStruct(
            [batch_size, self.action_horizon, self.action_dim], jnp.float32
        )
        return observation_spec, action_spec


class PiCOT(_model.BaseModel):
    """PiCOT model: flow matching + CoT + FAST action tokens."""

    def __init__(self, config: PiCOTConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.max_cot_tokens = config.max_cot_tokens
        self.max_fast_tokens = config.max_fast_tokens
        self.flow_matching_loss_weight = config.flow_matching_loss_weight

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        # Dual-stream LLM: stream 0 = PaliGemma (normal RMSNorm), stream 1 = action expert (adaRMS)
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=True,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True])

        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)

        # Action expert projections (pi0.5 / adaRMS pattern)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.time_mlp_in = nnx.Linear(
            action_expert_config.width, action_expert_config.width, rngs=rngs
        )
        self.time_mlp_out = nnx.Linear(
            action_expert_config.width, action_expert_config.width, rngs=rngs
        )
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # Pre-compute seed token IDs for inference generation.
        # These are Python lists (compile-time constants) — they don't participate in JAX param trees.
        import sentencepiece  # noqa: PLC0415

        pg_tok_path = _download.maybe_download(
            "gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"}
        )
        with pg_tok_path.open("rb") as f:
            _pg_tok = sentencepiece.SentencePieceProcessor(model_proto=f.read())

        self._cot_seed_ids: list[int] = _pg_tok.encode("TASK:", add_bos=False, add_eos=False)
        self._fast_seed_ids: list[int] = _pg_tok.encode("Action: ", add_bos=False, add_eos=False)
        # "|" EOS token — used as stop signal for FAST sequence
        self._fast_stop_id: int = int(_pg_tok.encode("|", add_eos=True)[-1])
        self._eos_id: int = 1  # PaliGemma EOS

        self.deterministic = True

    # ------------------------------------------------------------------ #
    # Embedding helpers                                                    #
    # ------------------------------------------------------------------ #

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
    ]:
        """Embed images + text prompt tokens (bidirectional prefix)."""
        tokens, input_mask, ar_mask = [], [], []

        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(obs.image_masks[name], "b -> b s", s=image_tokens.shape[1])
            )
            ar_mask += [False] * image_tokens.shape[1]

        if obs.tokenized_prompt is not None:
            lang_emb = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(lang_emb)
            input_mask.append(obs.tokenized_prompt_mask)
            ar_mask += [False] * lang_emb.shape[1]

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_cot_and_fast(
        self, obs: _model.Observation
    ) -> tuple[
        at.Float[at.Array, "b cf emb"],
        at.Bool[at.Array, "b cf"],
        at.Bool[at.Array, " cf"],
    ]:
        """Embed CoT reasoning + FAST action tokens (both causal)."""
        tokens, masks, ar_mask = [], [], []

        if obs.tokenized_cot_reasoning is not None:
            cot_emb = self.PaliGemma.llm(obs.tokenized_cot_reasoning, method="embed")
            cot_mask = obs.tokenized_cot_reasoning_mask
            cot_emb = cot_emb * cot_mask[:, :, None].astype(cot_emb.dtype)
            tokens.append(cot_emb)
            masks.append(cot_mask)
            ar_mask += [True] * cot_emb.shape[1]

        if obs.tokenized_fast_actions is not None:
            fast_emb = self.PaliGemma.llm(obs.tokenized_fast_actions, method="embed")
            fast_mask = obs.tokenized_fast_actions_mask
            fast_emb = fast_emb * fast_mask[:, :, None].astype(fast_emb.dtype)
            tokens.append(fast_emb)
            masks.append(fast_mask)
            ar_mask += [True] * fast_emb.shape[1]

        return (
            jnp.concatenate(tokens, axis=1),
            jnp.concatenate(masks, axis=1),
            jnp.array(ar_mask),
        )

    @at.typecheck
    def embed_suffix(
        self,
        obs: _model.Observation,
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, " b"],
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"],
    ]:
        """Embed noisy continuous actions with adaRMS timestep injection (pi0.5 pattern)."""
        action_tokens = self.action_in_proj(noisy_actions)
        time_emb = posemb_sincos(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0
        )
        time_emb = nnx.swish(self.time_mlp_in(time_emb))
        time_emb = nnx.swish(self.time_mlp_out(time_emb))

        suffix_tokens = action_tokens
        input_mask = jnp.ones(suffix_tokens.shape[:2], dtype=jnp.bool_)
        # First action token starts a new causal group; the rest share it (bidirectional within group)
        ar_mask = jnp.array([True] + [False] * (self.action_horizon - 1))
        return suffix_tokens, input_mask, ar_mask, time_emb

    # ------------------------------------------------------------------ #
    # Training                                                             #
    # ------------------------------------------------------------------ #

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ):
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        x_t = time[..., None, None] * noise + (1 - time[..., None, None]) * actions
        u_t = noise - actions

        # Embed all three segments
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        cf_tokens, cf_mask, cf_ar_mask = self.embed_cot_and_fast(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, x_t, time
        )

        # Build joint stream-0 sequence: [prefix | CoT | FAST]
        s0_tokens = jnp.concatenate([prefix_tokens, cf_tokens], axis=1)
        s0_mask = jnp.concatenate([prefix_mask, cf_mask], axis=1)
        s0_ar = jnp.concatenate([prefix_ar_mask, cf_ar_mask], axis=0)

        # Build base attention mask over full joint sequence [s0 | suffix]
        joint_mask = jnp.concatenate([s0_mask, suffix_mask], axis=1)
        joint_ar = jnp.concatenate([s0_ar, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(joint_mask, joint_ar)  # [B, P+C+F+H, P+C+F+H]

        # Block stream-1 (action expert) from attending to FAST action tokens.
        # FAST tokens occupy positions [P+C, P+C+F) in the joint sequence.
        # Stream-1 tokens occupy positions [P+C+F, P+C+F+H).
        # Allowing stream-1 to attend to FAST would leak the target action encoding
        # into the flow-matching denoising stream.
        P = prefix_tokens.shape[1]
        C = self.max_cot_tokens
        F = self.max_fast_tokens
        total_len = joint_mask.shape[1]
        all_idx = jnp.arange(total_len)
        is_suffix_row = all_idx >= (P + C + F)                                    # stream-1 query positions
        is_fast_col = (all_idx >= (P + C)) & (all_idx < (P + C + F))             # FAST key positions
        no_leak = ~(is_suffix_row[:, None] & is_fast_col[None, :])               # [total_len, total_len]
        attn_mask = attn_mask & no_leak[None, :, :]                               # broadcast over batch

        positions = jnp.cumsum(joint_mask, axis=1) - 1

        # Single joint forward pass
        (s0_out, s1_out), _ = self.PaliGemma.llm(
            [s0_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
        )

        # ── AR loss: CoT + FAST tokens as one joint sequence ──────────────────
        # s0_logits[b, i, :] is the distribution predicting the token at position i+1.
        # Shifting by one: output at positions [P-1, P+C+F-1) predicts targets
        #   [CoT[0..C-1] | FAST[0..F-1]]  (length C+F).
        s0_logits = self.PaliGemma.llm(s0_out, method="decode_logits")   # [B, P+C+F, V]
        cf_logits = s0_logits[:, P - 1 : P + C + F - 1, :]              # [B, C+F, V]

        batch_size = actions.shape[0]
        cf_targets = jnp.concatenate(
            [observation.tokenized_cot_reasoning, observation.tokenized_fast_actions], axis=1
        )  # [B, C+F]
        cf_loss_mask = jnp.concatenate(
            [observation.tokenized_cot_reasoning_mask, observation.tokenized_fast_actions_mask],
            axis=1,
        ).astype(jnp.float32)  # [B, C+F]

        cf_logp = jax.nn.log_softmax(cf_logits, axis=-1)
        b_idx = jnp.arange(batch_size)[:, None]
        t_idx = jnp.arange(C + F)[None, :]
        target_logp = cf_logp[b_idx, t_idx, cf_targets]                  # [B, C+F]
        # Normalise per sample so long sequences do not dominate
        ar_loss = -jnp.sum(target_logp * cf_loss_mask, axis=-1) / jnp.maximum(
            jnp.sum(cf_loss_mask, axis=-1), 1.0
        )  # [B]

        # ── Flow-matching loss ────────────────────────────────────────────────
        v_t = self.action_out_proj(s1_out[:, -self.action_horizon :])
        flow_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)  # [B, H]

        # ── Total loss ────────────────────────────────────────────────────────
        # ar_loss [B] is broadcast to [B, H] so jnp.mean(total_loss) weights
        # both terms correctly regardless of action_horizon.
        total_loss = flow_loss + self.flow_matching_loss_weight * ar_loss[:, None]

        return total_loss, {
            "ar_loss": jnp.mean(ar_loss),
            "flow_loss": jnp.mean(flow_loss),
        }

    # ------------------------------------------------------------------ #
    # Inference: autoregressive generation                                 #
    # ------------------------------------------------------------------ #

    def generate_cot_tokens(
        self,
        rng: at.KeyArrayLike,
        prefix_tokens: at.Float[at.Array, "b s emb"],
        prefix_mask: at.Bool[at.Array, "b s"],
        prefix_ar_mask: at.Bool[at.Array, " s"],
        temperature: float = 0.0,
    ):
        """Autoregressively generate CoT tokens from the prefix KV cache.

        Uses Python for-loop (JIT-unrolled) to avoid jax.lax.while_loop incompatibility
        with NNX Linen bridge (following the same pattern as pi0.py:generate_cot_tokens).
        """
        batch_size = prefix_tokens.shape[0]
        prefix_len = prefix_tokens.shape[1]
        valid_prefix_lens = jnp.sum(prefix_mask, axis=1)  # [B]

        # Compute prefix KV cache
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=prefix_positions
        )

        # Warm up with "TASK:" seed tokens (all but last)
        seed_ids = self._cot_seed_ids  # Python list of ints — compile-time constant
        seed_len = len(seed_ids)  # typically 4 tokens
        warm_len = seed_len - 1  # tokens to warm up before the loop

        if warm_len > 0:
            warm_arr = jnp.array(seed_ids[:warm_len], dtype=jnp.int32)
            warm_ids = jnp.broadcast_to(warm_arr[None, :], (batch_size, warm_len))
            warm_emb = self.PaliGemma.llm(warm_ids, method="embed")
            warm_positions = (
                jnp.arange(warm_len, dtype=jnp.int32)[None, :] + valid_prefix_lens[:, None]
            )
            prefix_col = jnp.ones((batch_size, warm_len, prefix_len), dtype=jnp.bool_)
            causal_col = jnp.broadcast_to(
                jnp.tril(jnp.ones((warm_len, warm_len), dtype=jnp.bool_))[None, :, :],
                (batch_size, warm_len, warm_len),
            )
            warm_mask = jnp.concatenate([prefix_col, causal_col], axis=-1)
            _, kv_cache = self.PaliGemma.llm(
                [warm_emb, None],
                mask=warm_mask,
                positions=warm_positions,
                kv_cache=kv_cache,
                adarms_cond=[None, None],
            )

        # Total output length: all seed tokens + generated tokens
        # seed_len = warm_len + 1 (the last seed token is also recorded in the buffer)
        total_len = seed_len + self.max_cot_tokens

        # Initialise output buffer (all zeros = PAD)
        out_buf = jnp.zeros((batch_size, total_len), dtype=jnp.int32)
        # Write ALL seed tokens (including the last one) into the buffer
        for i in range(seed_len):
            out_buf = out_buf.at[:, i].set(seed_ids[i])

        # Current token starts as the last seed token
        cur_token = jnp.full((batch_size, 1), seed_ids[-1], dtype=jnp.int32)
        has_finished = jnp.zeros((batch_size,), dtype=jnp.bool_)

        # Pre-allocate KV cache to full size so shapes are static for jax.lax.scan.
        # kv_cache currently covers prefix_len + warm_len tokens; pad to T_max with zeros.
        T_max = prefix_len + seed_len + self.max_cot_tokens
        cache_k, cache_v = kv_cache
        pad_len = T_max - cache_k.shape[2]
        pad_k = jnp.zeros(
            (cache_k.shape[0], cache_k.shape[1], pad_len, cache_k.shape[3], cache_k.shape[4]),
            dtype=cache_k.dtype,
        )
        pad_v = jnp.zeros_like(pad_k)
        full_kv_cache = (
            jnp.concatenate([cache_k, pad_k], axis=2),
            jnp.concatenate([cache_v, pad_v], axis=2),
        )

        def while_body(carry):
            """while_loop body: one AR step.

            Uses scatter-write KV cache (static shape) + while_loop for early
            stopping when all sequences have generated EOS.
            """
            cur_tok, buf, kvc, loop_rng, finished, loop_i = carry

            cur_emb = self.PaliGemma.llm(cur_tok, method="embed")
            write_idx = prefix_len + warm_len + loop_i
            cur_pos = (valid_prefix_lens + warm_len + loop_i)[:, None]
            cur_mask = jnp.arange(T_max) <= write_idx  # [T_max]
            cur_mask = jnp.broadcast_to(cur_mask[None, None, :], (batch_size, 1, T_max))

            (out0, _), next_kvc = self.PaliGemma.llm(
                [cur_emb, None],
                mask=cur_mask,
                positions=cur_pos,
                kv_cache=kvc,
                kv_write_index=write_idx,
                adarms_cond=[None, None],
            )

            logits = self.PaliGemma.llm(out0[:, -1:, :], method="decode_logits")[:, 0, :]

            new_rng, sample_rng = jax.random.split(loop_rng)
            safe_temp = jnp.maximum(temperature, 1e-8)
            next_id = jax.lax.cond(
                temperature > 0.0,
                lambda: jax.random.categorical(sample_rng, logits / safe_temp, axis=-1),
                lambda: jnp.argmax(logits, axis=-1),
            )

            new_finished = finished | (next_id == self._eos_id)
            safe_next = jnp.where(finished, jnp.zeros_like(next_id), next_id)
            buf = buf.at[:, seed_len + loop_i].set(safe_next)

            return (safe_next[:, None], buf, next_kvc, new_rng, new_finished, loop_i + 1)

        def while_cond(carry):
            # Stop when all sequences finished OR max steps reached
            _, _, _, _, finished, loop_i = carry
            return jnp.logical_and(loop_i < self.max_cot_tokens, ~jnp.all(finished))

        carry = (cur_token, out_buf, full_kv_cache, rng, has_finished, jnp.zeros((), dtype=jnp.int32))
        _, all_ids, _, _, _, _ = jax.lax.while_loop(while_cond, while_body, carry)

        # Build embeddings + mask for the generated sequence
        cot_emb = self.PaliGemma.llm(all_ids, method="embed")
        cot_mask = all_ids > 0
        cot_emb = cot_emb * cot_mask[:, :, None].astype(cot_emb.dtype)
        cot_ar_mask = jnp.ones(total_len, dtype=jnp.bool_)

        return cot_emb, cot_mask, cot_ar_mask, all_ids


    # ------------------------------------------------------------------ #
    # Inference: sample_actions                                            #
    # ------------------------------------------------------------------ #

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        cot_temperature: float = 0.0,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]

        rng, noise_rng, cot_rng = jax.random.split(rng, 3)
        if noise is None:
            noise = jax.random.normal(
                noise_rng, (batch_size, self.action_horizon, self.action_dim)
            )

        # Phase 1: embed prefix (images + text prompt)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)

        # Phase 2: autoregressive CoT generation
        cot_emb, cot_mask, cot_ar_mask, _ = self.generate_cot_tokens(
            cot_rng, prefix_tokens, prefix_mask, prefix_ar_mask, temperature=cot_temperature
        )

        # Phase 3: build KV cache for [prefix | CoT] with static shapes for while_loop
        ext_prefix = jnp.concatenate([prefix_tokens, cot_emb], axis=1)
        ext_mask = jnp.concatenate([prefix_mask, cot_mask], axis=1)
        ext_ar = jnp.concatenate([prefix_ar_mask, cot_ar_mask], axis=0)
        ext_attn_mask = make_attn_mask(ext_mask, ext_ar)
        ext_positions = jnp.cumsum(ext_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm(
            [ext_prefix, None], mask=ext_attn_mask, positions=ext_positions
        )

        # Phase 5: flow matching denoising loop (identical to pi0.5 sample_actions)
        def step(carry):
            x_t, time = carry
            suf_tok, suf_mask, suf_ar, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, (batch_size,))
            )
            suf_self_mask = make_attn_mask(suf_mask, suf_ar)
            suf_to_pre = einops.repeat(ext_mask, "b p -> b s p", s=suf_tok.shape[1])
            full_mask = jnp.concatenate([suf_to_pre, suf_self_mask], axis=-1)
            positions = (
                jnp.sum(ext_mask, axis=-1)[:, None] + jnp.cumsum(suf_mask, axis=-1) - 1
            )
            (_, suf_out), _ = self.PaliGemma.llm(
                [None, suf_tok],
                mask=full_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            v_t = self.action_out_proj(suf_out[:, -self.action_horizon :])
            return x_t + dt * v_t, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
