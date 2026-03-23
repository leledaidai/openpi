import dataclasses
import logging
from typing import Any

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
import sentencepiece
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_fast as _pi0_fast
import openpi.models.gemma_fast as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
import openpi.shared.download as _download
import openpi.shared.nnx_utils as nnx_utils
from openpi.utils import cot_utils

logger = logging.getLogger("openpi")


def _decoder_variant_from_main(variant: _gemma.Variant) -> _gemma.Variant:
    return "gemma_2b" if variant == "gemma_2b_lora" else variant


class LatentProjection(nnx.Module):
    def __init__(
        self,
        width: int,
        hidden_dim: int,
        *,
        dropout: float,
        no_ln: bool,
        rngs: nnx.Rngs,
    ):
        self.dropout = nnx.Dropout(dropout, rngs=rngs)
        self.fc_in = nnx.Linear(in_features=width, out_features=hidden_dim, rngs=rngs)
        self.fc_out = nnx.Linear(in_features=hidden_dim, out_features=width, rngs=rngs)
        self.norm = None if no_ln else nnx.LayerNorm(num_features=width, rngs=rngs)

    def __call__(self, x: at.Float[at.Array, "... d"]) -> at.Float[at.Array, "... d"]:
        x = self.dropout(x)
        x = nnx.gelu(self.fc_in(x))
        x = self.fc_out(x)
        if self.norm is not None:
            x = self.norm(x)
        return x


class DummyImageEncoder(nnx.Module):
    def __init__(self, width: int, *, rngs: nnx.Rngs):
        self.proj = nnx.Linear(in_features=3, out_features=width, rngs=rngs)

    def __call__(self, images: at.Float[at.Array, "b h w c"], *, train: bool = False):
        del train
        pooled = jnp.mean(images, axis=(1, 2))
        return self.proj(pooled)[:, None, :], None


def _masked_mean(values: at.Float[at.Array, "b t"], mask: at.Float[at.Array, "b t"]) -> at.Float[at.Array, "b"]:
    return jnp.sum(values * mask, axis=-1) / jnp.maximum(jnp.sum(mask, axis=-1), 1.0)


def _gather_positions(x: at.Float[at.Array, "b t d"], positions: at.Int[at.Array, "b"]) -> at.Float[at.Array, "b d"]:
    return jnp.take_along_axis(x, positions[:, None, None], axis=1)[:, 0, :]


def _gather_last_valid(x: at.Float[at.Array, "b t d"], mask: at.Bool[at.Array, "b t"]) -> at.Float[at.Array, "b 1 d"]:
    last_idx = jnp.maximum(jnp.sum(mask.astype(jnp.int32), axis=-1) - 1, 0)
    return _gather_positions(x, last_idx)[:, None, :]


def _pack_sequence(
    embeddings: at.Float[at.Array, "b t d"],
    mask: at.Bool[at.Array, "b t"],
    ar_mask: at.Int[at.Array, "b t"],
) -> tuple[at.Float[at.Array, "b t d"], at.Bool[at.Array, "b t"], at.Int[at.Array, "b t"]]:
    seq_len = embeddings.shape[1]
    indices = jnp.arange(seq_len)[None, :]
    order = jnp.argsort(jnp.where(mask, indices, seq_len + indices), axis=-1)
    return (
        jnp.take_along_axis(embeddings, order[:, :, None], axis=1),
        jnp.take_along_axis(mask, order, axis=1),
        jnp.take_along_axis(ar_mask, order, axis=1),
    )


@dataclasses.dataclass(frozen=True)
class Pi0FASTImplicitCoTConfig(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    decoder_variant: _gemma.Variant | None = None

    action_dim: int = 32
    action_horizon: int = 32
    max_token_len: int = 250
    max_cot_tokens: int = 512
    max_action_postfix_tokens: int = 128
    implicit_max_step_tokens: int = 128
    implicit_step_tags: tuple[str, ...] = dataclasses.field(
        default_factory=lambda: tuple(cot_utils.get_implicit_cot_tags_list())
    )

    use_prj: bool = True
    prj_dim: int | None = None
    prj_dropout: float = 0.0
    prj_no_ln: bool = False

    teacher_ce_loss_weight: float = 1.0
    decoder_loss_weight: float = 1.0
    distill_loss_weight: float = 1.0
    latent_decode_max_tokens: int = 64

    fast_model_tokenizer: Any | None = None
    fast_model_tokenizer_kwargs: dict[str, Any] | None = None

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI0_FAST_IMPLICIT_COT

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0FASTImplicitCoT":
        return Pi0FASTImplicitCoT(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={k: image_spec for k in _model.IMAGE_KEYS},
                image_masks={k: image_mask_spec for k in _model.IMAGE_KEYS},
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.bool_),
                tokenized_teacher_cot=jax.ShapeDtypeStruct([batch_size, self.max_cot_tokens], jnp.int32),
                tokenized_teacher_cot_mask=jax.ShapeDtypeStruct([batch_size, self.max_cot_tokens], jnp.bool_),
                tokenized_action_postfix=jax.ShapeDtypeStruct(
                    [batch_size, self.max_action_postfix_tokens], jnp.int32
                ),
                tokenized_action_postfix_mask=jax.ShapeDtypeStruct(
                    [batch_size, self.max_action_postfix_tokens], jnp.bool_
                ),
                tokenized_implicit_cot_steps=jax.ShapeDtypeStruct(
                    [batch_size, len(self.implicit_step_tags), self.implicit_max_step_tokens], jnp.int32
                ),
                tokenized_implicit_cot_steps_mask=jax.ShapeDtypeStruct(
                    [batch_size, len(self.implicit_step_tags), self.implicit_max_step_tokens], jnp.bool_
                ),
                has_cot=jax.ShapeDtypeStruct([batch_size], jnp.bool_),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)
        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        if "lora" in self.paligemma_variant:
            return nnx.All(nnx_utils.PathRegex(".*main_llm.*"), nnx.Not(nnx_utils.PathRegex(".*lora.*")))
        return nnx.Nothing


class Pi0FASTImplicitCoT(_model.BaseModel):
    def __init__(self, config: Pi0FASTImplicitCoTConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.max_cot_tokens = config.max_cot_tokens
        self.max_action_postfix_tokens = config.max_action_postfix_tokens
        self.implicit_max_step_tokens = config.implicit_max_step_tokens
        self.implicit_step_tags = config.implicit_step_tags
        self.teacher_ce_loss_weight = config.teacher_ce_loss_weight
        self.decoder_loss_weight = config.decoder_loss_weight
        self.distill_loss_weight = config.distill_loss_weight
        self.latent_decode_max_tokens = config.latent_decode_max_tokens
        self.num_implicit_steps = len(config.implicit_step_tags)
        self.use_prj = config.use_prj
        self.use_cot = True

        main_cfg = _gemma.get_config(config.paligemma_variant)
        decoder_variant = config.decoder_variant or _decoder_variant_from_main(config.paligemma_variant)
        decoder_cfg = _gemma.get_config(decoder_variant)
        if main_cfg.width != decoder_cfg.width:
            raise ValueError(
                f"Decoder width must match main LLM width for implicit CoT. Got {decoder_cfg.width} vs {main_cfg.width}."
            )

        self.main_llm = nnx_bridge.ToNNX(
            _gemma.Module(
                **main_cfg,
                embed_dtype=config.dtype,
                cache_dtype=config.dtype,
            )
        )
        self.main_llm.lazy_init(rngs=rngs, method="init")

        self.decoder_llm = nnx_bridge.ToNNX(
            _gemma.Module(
                **decoder_cfg,
                embed_dtype=config.dtype,
                cache_dtype=config.dtype,
            )
        )
        self.decoder_llm.lazy_init(rngs=rngs, method="init")

        if config.paligemma_variant == "dummy":
            self.img = DummyImageEncoder(main_cfg.width, rngs=rngs)
        else:
            self.img = nnx_bridge.ToNNX(
                _siglip.Module(
                    num_classes=main_cfg.width,
                    variant="So400m/14",
                    pool_type="none",
                    scan=True,
                    dtype_mm=config.dtype,
                )
            )
            self.img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)

        self.prj = None
        if config.use_prj:
            self.prj = LatentProjection(
                width=main_cfg.width,
                hidden_dim=config.prj_dim or main_cfg.width,
                dropout=config.prj_dropout,
                no_ln=config.prj_no_ln,
                rngs=rngs,
            )

        tok_path = _download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with tok_path.open("rb") as f:
            self._text_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())
        self._action_prefix_token_len = len(self._text_tokenizer.encode("Action: "))

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Int[at.Array, "b s"]]:
        token_embeddings = []
        input_mask = []
        ar_mask = []
        for name in obs.images:
            image_token_embeddings, _ = self.img(obs.images[name], train=False)
            token_embeddings.append(image_token_embeddings)
            img_mask = einops.repeat(obs.image_masks[name], "b -> b s", s=image_token_embeddings.shape[1])
            input_mask.append(img_mask)
            ar_mask.append(jnp.zeros_like(img_mask, dtype=jnp.int32))

        assert obs.tokenized_prompt is not None
        assert obs.tokenized_prompt_mask is not None
        prompt_emb = self.main_llm(obs.tokenized_prompt, embed_only=True)
        token_embeddings.append(prompt_emb)
        input_mask.append(obs.tokenized_prompt_mask)
        ar_mask.append(jnp.zeros_like(obs.tokenized_prompt_mask, dtype=jnp.int32))
        return (
            jnp.concatenate(token_embeddings, axis=1),
            jnp.concatenate(input_mask, axis=1),
            jnp.concatenate(ar_mask, axis=1),
        )

    def _apply_prj(self, latent: at.Float[at.Array, "b 1 d"]) -> at.Float[at.Array, "b 1 d"]:
        if self.prj is None:
            return latent
        return self.prj(latent)

    def _run_llm_sequence(
        self,
        llm: nnx.Module,
        embeddings: at.Float[at.Array, "b s d"],
        mask: at.Bool[at.Array, "b s"],
        ar_mask: at.Int[at.Array, "b s"],
        *,
        return_hidden_states: bool = False,
    ) -> tuple[at.Float[at.Array, "b s d"], tuple[at.Float[at.Array, "b s d"], ...]]:
        embeddings, mask, ar_mask = _pack_sequence(embeddings, mask, ar_mask)
        attn_mask = _pi0_fast.make_attn_mask(mask, ar_mask)
        pre_logits, _, out = llm(
            embedded_prefix=embeddings,
            mask=attn_mask,
            return_prelogits=True,
            return_hidden_states=return_hidden_states,
        )
        hidden_states = tuple(out.get("hidden_states", (pre_logits,)))
        return pre_logits, hidden_states

    def _concat_segments(
        self,
        segments: list[tuple[at.Float[at.Array, "b s d"], at.Bool[at.Array, "b s"], at.Int[at.Array, "b s"]]],
    ) -> tuple[at.Float[at.Array, "b s d"], at.Bool[at.Array, "b s"], at.Int[at.Array, "b s"]]:
        embeds = jnp.concatenate([seg[0] for seg in segments], axis=1)
        masks = jnp.concatenate([seg[1] for seg in segments], axis=1)
        ar_masks = jnp.concatenate([seg[2] for seg in segments], axis=1)
        return embeds, masks, ar_masks

    def _build_student_latents(
        self,
        prefix_embeds: at.Float[at.Array, "b s d"],
        prefix_mask: at.Bool[at.Array, "b s"],
        prefix_ar: at.Int[at.Array, "b s"],
    ) -> tuple[list[at.Float[at.Array, "b 1 d"]], list[at.Float[at.Array, "b 1 d"]]]:
        prefix_pre_logits, _ = self._run_llm_sequence(self.main_llm, prefix_embeds, prefix_mask, prefix_ar)
        current_latent = self._apply_prj(_gather_last_valid(prefix_pre_logits, prefix_mask))
        decoded_latents = [current_latent]
        fed_latents = []

        batch_size = prefix_embeds.shape[0]
        for _ in range(self.num_implicit_steps - 1):
            fed_latents.append(current_latent)
            latent_embeds = jnp.concatenate(fed_latents, axis=1)
            latent_mask = jnp.ones((batch_size, len(fed_latents)), dtype=jnp.bool_)
            latent_ar = jnp.ones((batch_size, len(fed_latents)), dtype=jnp.int32)
            context_embeds, context_mask, context_ar = self._concat_segments(
                [
                    (prefix_embeds, prefix_mask, prefix_ar),
                    (latent_embeds, latent_mask, latent_ar),
                ]
            )
            context_pre_logits, _ = self._run_llm_sequence(self.main_llm, context_embeds, context_mask, context_ar)
            current_latent = self._apply_prj(_gather_last_valid(context_pre_logits, context_mask))
            decoded_latents.append(current_latent)

        return decoded_latents, fed_latents

    def _ce_from_prelogits(
        self,
        llm: nnx.Module,
        pre_logits: at.Float[at.Array, "b t d"],
        start_positions: at.Int[at.Array, "b"],
        targets: at.Int[at.Array, "b target_t"],
        target_mask: at.Bool[at.Array, "b target_t"],
    ) -> at.Float[at.Array, "b"]:
        batch_size, target_len = targets.shape
        offsets = start_positions[:, None] + jnp.arange(target_len)[None, :]
        selected = jnp.take_along_axis(pre_logits, offsets[:, :, None], axis=1)
        logits, _ = llm(pre_logits=selected)
        logp = jax.nn.log_softmax(logits, axis=-1)
        target_logp = jnp.take_along_axis(logp, targets[:, :, None], axis=-1)[:, :, 0]
        return -_masked_mean(target_logp, target_mask.astype(jnp.float32))

    def _hidden_state_distill(
        self,
        student_hidden_states: tuple[at.Float[at.Array, "b t d"], ...],
        student_positions: at.Int[at.Array, "b"],
        teacher_hidden_states: tuple[at.Float[at.Array, "b t d"], ...],
        teacher_positions: at.Int[at.Array, "b"],
        has_cot: at.Float[at.Array, "b"],
    ) -> at.Float[at.Array, "b"]:
        losses = []
        for student_hidden, teacher_hidden in zip(student_hidden_states, teacher_hidden_states, strict=False):
            student_selected = _gather_positions(student_hidden, student_positions)
            teacher_selected = _gather_positions(teacher_hidden, teacher_positions)
            losses.append(jnp.mean(jnp.square(student_selected - teacher_selected), axis=-1))
        if not losses:
            return jnp.zeros_like(has_cot)
        return jnp.mean(jnp.stack(losses, axis=0), axis=0) * has_cot

    def _student_action_loss(
        self,
        prefix_embeds: at.Float[at.Array, "b s d"],
        prefix_mask: at.Bool[at.Array, "b s"],
        prefix_ar: at.Int[at.Array, "b s"],
        fed_latents: list[at.Float[at.Array, "b 1 d"]],
        action_tokens: at.Int[at.Array, "b t"],
        action_mask: at.Bool[at.Array, "b t"],
        *,
        return_hidden_states: bool = False,
    ) -> tuple[
        at.Float[at.Array, "b"],
        tuple[at.Float[at.Array, "b t d"], ...],
        at.Int[at.Array, "b"],
        at.Int[at.Array, "b"],
    ]:
        batch_size = prefix_embeds.shape[0]
        segments = [(prefix_embeds, prefix_mask, prefix_ar)]
        prefix_len = jnp.sum(prefix_mask.astype(jnp.int32), axis=-1)

        if fed_latents:
            latent_embeds = jnp.concatenate(fed_latents, axis=1)
            latent_mask = jnp.ones((batch_size, len(fed_latents)), dtype=jnp.bool_)
            latent_ar = jnp.ones((batch_size, len(fed_latents)), dtype=jnp.int32)
            segments.append((latent_embeds, latent_mask, latent_ar))
        else:
            latent_mask = jnp.zeros((batch_size, 0), dtype=jnp.bool_)

        action_embeds = self.main_llm(action_tokens, embed_only=True)
        action_ar = jnp.ones_like(action_mask, dtype=jnp.int32)
        segments.append((action_embeds, action_mask, action_ar))
        seq_embeds, seq_mask, seq_ar = self._concat_segments(segments)
        pre_logits, hidden_states = self._run_llm_sequence(
            self.main_llm, seq_embeds, seq_mask, seq_ar, return_hidden_states=return_hidden_states
        )

        context_len = prefix_len + jnp.sum(latent_mask.astype(jnp.int32), axis=-1)
        start_positions = jnp.maximum(context_len - 1, 0)
        loss = self._ce_from_prelogits(self.main_llm, pre_logits, start_positions, action_tokens, action_mask)
        first_fast_positions = jnp.maximum(context_len - 1 + self._action_prefix_token_len, 0)
        return loss, hidden_states, first_fast_positions, context_len

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ):
        if train:
            self.train()
        else:
            self.eval()

        observation = _model.preprocess_observation(
            rng if train else None,
            observation,
            train=train,
            image_keys=list(observation.images.keys()),
        )

        assert observation.tokenized_action_postfix is not None
        assert observation.tokenized_action_postfix_mask is not None
        assert observation.tokenized_teacher_cot is not None
        assert observation.tokenized_teacher_cot_mask is not None
        assert observation.tokenized_implicit_cot_steps is not None
        assert observation.tokenized_implicit_cot_steps_mask is not None

        has_cot = (
            observation.has_cot.astype(jnp.float32)
            if observation.has_cot is not None
            else jnp.ones(actions.shape[0], dtype=jnp.float32)
        )

        prefix_embeds, prefix_mask, prefix_ar = self.embed_prefix(observation)
        decoded_latents, fed_latents = self._build_student_latents(prefix_embeds, prefix_mask, prefix_ar)

        student_action_loss, student_hidden_states, student_fast_positions, _ = self._student_action_loss(
            prefix_embeds,
            prefix_mask,
            prefix_ar,
            fed_latents,
            observation.tokenized_action_postfix,
            observation.tokenized_action_postfix_mask,
            return_hidden_states=True,
        )

        batch_size = prefix_embeds.shape[0]
        teacher_cot_embeds = self.main_llm(observation.tokenized_teacher_cot, embed_only=True)
        teacher_cot_ar = jnp.ones_like(observation.tokenized_teacher_cot_mask, dtype=jnp.int32)
        action_embeds = self.main_llm(observation.tokenized_action_postfix, embed_only=True)
        action_ar = jnp.ones_like(observation.tokenized_action_postfix_mask, dtype=jnp.int32)
        teacher_seq_embeds, teacher_seq_mask, teacher_seq_ar = self._concat_segments(
            [
                (prefix_embeds, prefix_mask, prefix_ar),
                (teacher_cot_embeds, observation.tokenized_teacher_cot_mask, teacher_cot_ar),
                (action_embeds, observation.tokenized_action_postfix_mask, action_ar),
            ]
        )
        teacher_pre_logits, teacher_hidden_states = self._run_llm_sequence(
            self.main_llm,
            teacher_seq_embeds,
            teacher_seq_mask,
            teacher_seq_ar,
            return_hidden_states=True,
        )
        prefix_len = jnp.sum(prefix_mask.astype(jnp.int32), axis=-1)
        teacher_start_positions = jnp.maximum(prefix_len - 1, 0)
        teacher_targets = jnp.concatenate(
            [observation.tokenized_teacher_cot, observation.tokenized_action_postfix], axis=1
        )
        teacher_target_mask = jnp.concatenate(
            [observation.tokenized_teacher_cot_mask, observation.tokenized_action_postfix_mask], axis=1
        )
        teacher_ce_loss = self._ce_from_prelogits(
            self.main_llm, teacher_pre_logits, teacher_start_positions, teacher_targets, teacher_target_mask
        ) * has_cot

        teacher_context_len = prefix_len + jnp.sum(observation.tokenized_teacher_cot_mask.astype(jnp.int32), axis=-1)
        teacher_fast_positions = jnp.maximum(teacher_context_len - 1 + self._action_prefix_token_len, 0)
        distill_loss = self._hidden_state_distill(
            student_hidden_states,
            student_fast_positions,
            teacher_hidden_states,
            teacher_fast_positions,
            has_cot,
        )

        decoder_losses = []
        effective_steps = []
        for step_idx, latent in enumerate(decoded_latents):
            step_tokens = observation.tokenized_implicit_cot_steps[:, step_idx, :]
            step_mask = observation.tokenized_implicit_cot_steps_mask[:, step_idx, :]
            latent_mask = jnp.ones((batch_size, 1), dtype=jnp.bool_)
            latent_ar = jnp.ones((batch_size, 1), dtype=jnp.int32)
            step_embeds = self.decoder_llm(step_tokens, embed_only=True)
            step_ar = jnp.ones_like(step_mask, dtype=jnp.int32)
            decoder_embeds, decoder_mask, decoder_ar = self._concat_segments(
                [(latent, latent_mask, latent_ar), (step_embeds, step_mask, step_ar)]
            )
            decoder_pre_logits, _ = self._run_llm_sequence(self.decoder_llm, decoder_embeds, decoder_mask, decoder_ar)
            step_loss = self._ce_from_prelogits(
                self.decoder_llm,
                decoder_pre_logits,
                jnp.zeros(batch_size, dtype=jnp.int32),
                step_tokens,
                step_mask,
            )
            has_tokens = jnp.any(step_mask, axis=-1).astype(jnp.float32) * has_cot
            decoder_losses.append(step_loss * has_tokens)
            effective_steps.append(has_tokens)

        if decoder_losses:
            decoder_loss = jnp.sum(jnp.stack(decoder_losses, axis=0), axis=0) / jnp.maximum(
                jnp.sum(jnp.stack(effective_steps, axis=0), axis=0), 1.0
            )
        else:
            decoder_loss = jnp.zeros_like(student_action_loss)

        total_loss = (
            student_action_loss
            + self.teacher_ce_loss_weight * teacher_ce_loss
            + self.decoder_loss_weight * decoder_loss
            + self.distill_loss_weight * distill_loss
        )
        return total_loss, {
            "student_action_loss": jnp.mean(student_action_loss),
            "teacher_ce_loss": jnp.mean(teacher_ce_loss),
            "decoder_loss": jnp.mean(decoder_loss),
            "distill_loss": jnp.mean(distill_loss),
        }

    def _sample_from_context(
        self,
        rng: at.KeyArrayLike,
        llm: nnx.Module,
        context_embeds: at.Float[at.Array, "b s d"],
        context_mask: at.Bool[at.Array, "b s"],
        context_ar_mask: at.Int[at.Array, "b s"],
        *,
        max_decoding_steps: int,
        temperature: float = 0.0,
    ) -> at.Int[at.Array, "b t"]:
        context_embeds, context_mask, context_ar_mask = _pack_sequence(context_embeds, context_mask, context_ar_mask)
        context_attn_mask = _pi0_fast.make_attn_mask(context_mask, context_ar_mask)
        context_embeds, context_mask, context_attn_mask = _pi0_fast.left_to_right_align(
            context_embeds, context_mask, context_attn_mask
        )
        prefill_size = context_embeds.shape[1]
        prefill_len = jnp.sum(context_mask, axis=-1)
        prefix_start = prefill_size - prefill_len

        context_attn_mask = jnp.pad(context_attn_mask, ((0, 0), (0, 0), (0, max_decoding_steps)))
        context_positions = jnp.cumsum(context_mask, axis=-1) - 1
        context_logits, kv_cache, _ = llm(
            embedded_prefix=context_embeds,
            mask=context_attn_mask,
            positions=context_positions,
            decode=True,
        )

        last_logit = context_logits[:, -1:]
        output_tokens = jnp.zeros((last_logit.shape[0], max_decoding_steps), dtype=jnp.int32)

        def step(carry):
            rng, prev_logit, tokens, cache, all_eos, step_idx = carry
            rng, step_rng = jax.random.split(rng)
            token = jax.lax.cond(
                temperature > 0.0,
                lambda _: jax.random.categorical(step_rng, prev_logit / temperature, axis=-1),
                lambda _: jnp.argmax(prev_logit, axis=-1),
                operand=None,
            )
            tokens = _pi0_fast.put_along_last_axis(
                tokens, jnp.broadcast_to(step_idx, (token.shape[0], 1)), token.astype(jnp.int32)
            )

            has_eos = jnp.any(token == _pi0_fast.PALIGEMMA_EOS_TOKEN, axis=-1)
            all_eos = jnp.all(has_eos)

            token_emb = llm(token, embed_only=True)
            positions = prefill_len[:, None] + step_idx + 1
            mask = jnp.logical_and(
                jnp.arange(prefill_size + max_decoding_steps)[None, None, :] >= prefix_start[:, None, None],
                jnp.arange(prefill_size + max_decoding_steps)[None, None, :]
                < jnp.broadcast_to(prefill_size + step_idx + 1, (prefix_start.shape[0], 1, 1)),
            )
            next_logit, next_cache, _ = llm(
                embedded_prefix=token_emb,
                mask=mask,
                positions=positions,
                decode=True,
                kv_cache=cache,
            )
            return rng, next_logit, tokens, next_cache, all_eos, step_idx + 1

        def cond(carry):
            _, _, _, _, all_eos, step_idx = carry
            return (~all_eos) & (step_idx < max_decoding_steps)

        _, _, output_tokens, _, _, _ = jax.lax.while_loop(
            cond, step, (rng, last_logit, output_tokens, kv_cache, False, 0)
        )
        return output_tokens

    def _student_context_for_sampling(
        self, observation: _model.Observation
    ) -> tuple[
        at.Float[at.Array, "b s d"],
        at.Bool[at.Array, "b s"],
        at.Int[at.Array, "b s"],
        list[at.Float[at.Array, "b 1 d"]],
    ]:
        prefix_embeds, prefix_mask, prefix_ar = self.embed_prefix(observation)
        decoded_latents, fed_latents = self._build_student_latents(prefix_embeds, prefix_mask, prefix_ar)
        batch_size = prefix_embeds.shape[0]
        segments = [(prefix_embeds, prefix_mask, prefix_ar)]
        if fed_latents:
            latent_embeds = jnp.concatenate(fed_latents, axis=1)
            latent_mask = jnp.ones((batch_size, len(fed_latents)), dtype=jnp.bool_)
            latent_ar = jnp.ones((batch_size, len(fed_latents)), dtype=jnp.int32)
            segments.append((latent_embeds, latent_mask, latent_ar))
        return (*self._concat_segments(segments), decoded_latents)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        max_decoding_steps: int = 256,
        temperature: float = 0.0,
        decode_latent_text: bool = False,
        latent_decode_temperature: float = 0.0,
        latent_decode_max_tokens: int | None = None,
    ) -> at.Int[at.Array, "b t"]:
        if decode_latent_text:
            return self.sample_actions_with_debug(
                rng,
                observation,
                max_decoding_steps=max_decoding_steps,
                temperature=temperature,
                latent_decode_temperature=latent_decode_temperature,
                latent_decode_max_tokens=latent_decode_max_tokens,
            )["actions"]

        self.eval()
        observation = _model.preprocess_observation(None, observation, train=False, image_keys=list(observation.images))
        context_embeds, context_mask, context_ar, _ = self._student_context_for_sampling(observation)
        return self._sample_from_context(
            rng,
            self.main_llm,
            context_embeds,
            context_mask,
            context_ar,
            max_decoding_steps=max_decoding_steps,
            temperature=temperature,
        )

    def sample_actions_with_debug(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        max_decoding_steps: int = 256,
        temperature: float = 0.0,
        latent_decode_temperature: float = 0.0,
        latent_decode_max_tokens: int | None = None,
    ) -> dict[str, at.Array]:
        self.eval()
        observation = _model.preprocess_observation(None, observation, train=False, image_keys=list(observation.images))
        context_embeds, context_mask, context_ar, decoded_latents = self._student_context_for_sampling(observation)
        actions = self._sample_from_context(
            rng,
            self.main_llm,
            context_embeds,
            context_mask,
            context_ar,
            max_decoding_steps=max_decoding_steps,
            temperature=temperature,
        )

        decode_steps = latent_decode_max_tokens or self.latent_decode_max_tokens
        latent_step_token_ids = []
        for latent in decoded_latents:
            batch_size = latent.shape[0]
            latent_mask = jnp.ones((batch_size, 1), dtype=jnp.bool_)
            latent_ar = jnp.ones((batch_size, 1), dtype=jnp.int32)
            latent_step_token_ids.append(
                self._sample_from_context(
                    rng,
                    self.decoder_llm,
                    latent,
                    latent_mask,
                    latent_ar,
                    max_decoding_steps=decode_steps,
                    temperature=latent_decode_temperature,
                )
            )

        return {
            "actions": actions,
            "latent_step_token_ids": jnp.stack(latent_step_token_ids, axis=1),
        }

    def decode_latent_step_token_ids(self, token_ids) -> list[list[str]]:
        token_ids = jnp.asarray(token_ids)
        if token_ids.ndim == 2:
            token_ids = token_ids[None, ...]
        output = []
        for batch_tokens in token_ids:
            step_texts = []
            for step in batch_tokens:
                step_ids = step.tolist()
                if 0 in step_ids:
                    step_ids = step_ids[: step_ids.index(0)]
                step_texts.append(self._text_tokenizer.decode(step_ids).strip())
            output.append(step_texts)
        return output
