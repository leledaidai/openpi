import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.use_cot = config.use_cot
        self.max_cot_tokens = config.max_cot_tokens
        self.cot_loss_weight = config.cot_loss_weight

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
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
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond
    
    
    @at.typecheck
    def generate_cot_tokens(
        self,
        rng: at.KeyArrayLike,
        prefix_tokens: at.Float[at.Array, "b s emb"],
        prefix_mask: at.Bool[at.Array, "b s"],
        prefix_ar_mask: at.Bool[at.Array, " s"],
        cot_target_tokens: at.Int[at.Array, "b cot_len"] | None = None,
        temperature: float = 0.0,
    ) -> tuple[
        at.Float[at.Array, "b cot_len emb"],
        at.Bool[at.Array, "b cot_len"],
        at.Bool[at.Array, " cot_len"],
        at.Int[at.Array, "b cot_len"] | None,
    ]:
        batch_size = prefix_tokens.shape[0]

        if cot_target_tokens is not None:
            # === 训练阶段：使用 Teacher Forcing ===
            # 直接使用ground truth CoT tokens，不需要自回归生成
            cot_len = cot_target_tokens.shape[1]
            cot_embeddings = self.PaliGemma.llm(cot_target_tokens, method="embed")
            cot_mask = (cot_target_tokens > 0)
            # CoT tokens之间使用因果掩码：第一个token可以attend到prefix，后续tokens采用自回归
            cot_ar_mask = jnp.array([True] * cot_len)
            return cot_embeddings, cot_mask, cot_ar_mask, cot_target_tokens

        # --- 推理阶段：使用 jax.lax.while_loop 进行自回归生成 ---
        EOS_TOKEN = 1  # End of sequence token (verified from PaliGemma tokenizer)
        BOS_TOKEN = 2  # Beginning of sequence token (verified from PaliGemma tokenizer)

        # 强制以 "TASK:" 开始生成，匹配训练时的格式
        # 参考: OpenVLA实现中使用 self.base_prompt = "TASK:" 作为生成起始
        TASK_PREFIX = "TASK:"
        # SentencePiece uses add_bos/add_eos, not add_special_tokens
        task_prefix_tokens = self.PaliGemma.llm.tokenizer.encode(TASK_PREFIX, add_bos=False, add_eos=False)
        task_prefix_tokens = jnp.array(task_prefix_tokens, dtype=jnp.int32)
        task_prefix_len = len(task_prefix_tokens)

        # prefix_len: padded sequence length (static); used for KV cache size computation.
        # valid_prefix_lens: per-example true token count (dynamic); used for RoPE positions
        # so that CoT tokens continue directly after the last valid prefix token with no gap.
        prefix_len = int(prefix_tokens.shape[1])
        valid_prefix_lens = jnp.sum(prefix_mask, axis=1)  # [batch_size]

        # 1. 初始化 KV Cache：计算 Prefix 部分的 keys 和 values
        # 这样在生成CoT tokens时就不需要重复计算prefix的attention
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, init_kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=prefix_positions
        )

        # 1b. [Bug Fix] 预热 TASK: 前缀的 KV Cache
        # 问题：init_kv_cache 只含 prefix tokens。循环从最后一个 TASK: token 开始，
        # 但 "T","A","S","K" 等前置 token 从未通过模型，KV cache 中不存在，
        # 导致最后一个 TASK: token 无法 attend 到完整的 "TASK:" 上下文。
        # 修复：将 TASK[0..task_prefix_len-2]（除最后一个以外）先做一次前向，
        # 写入 KV cache。循环中处理最后一个 TASK: token 时即可看到完整指令。
        if task_prefix_len > 1:
            warm_len = task_prefix_len - 1
            warm_token_ids = task_prefix_tokens[:warm_len][None, :].repeat(batch_size, axis=0)
            warm_emb = self.PaliGemma.llm(warm_token_ids, method="embed")
            # 位置：valid_prefix_len, valid_prefix_len+1, ..., valid_prefix_len+warm_len-1
            warm_positions = (
                jnp.arange(warm_len, dtype=jnp.int32)[None, :] + valid_prefix_lens[:, None]
            )  # [batch_size, warm_len]
            # Mask [batch, warm_len, prefix_len+warm_len]：
            #   - 每个 warm token 可以 attend 所有 prefix tokens（全 1）
            #   - warm tokens 之间使用因果掩码（下三角）
            prefix_col = jnp.ones((warm_len, prefix_len), dtype=jnp.bool_)
            causal_col = jnp.tril(jnp.ones((warm_len, warm_len), dtype=jnp.bool_))
            warm_mask = jnp.concatenate(
                [prefix_col, causal_col], axis=1
            )[None, ...].repeat(batch_size, axis=0)
            _, init_kv_cache = self.PaliGemma.llm(
                [warm_emb, None],
                mask=warm_mask,
                positions=warm_positions,
                kv_cache=init_kv_cache,
                adarms_cond=[None, None],
            )
        # init_kv_cache 现在包含 (prefix_len + task_prefix_len - 1) 个 token 的 KV。

        # 2. 预分配结果数组 (JIT 要求固定形状)
        # 结果包含: "TASK:" tokens + 最多 max_cot_tokens 个生成的 tokens
        total_max_len = self.max_cot_tokens + task_prefix_len
        init_tokens = jnp.zeros((batch_size, total_max_len), dtype=jnp.int32)
        # 设置初始的 "TASK:" tokens
        for i in range(task_prefix_len):
            init_tokens = init_tokens.at[:, i].set(task_prefix_tokens[i])

        # 3. 定义循环状态 (Carry)
        # 元组包含: (当前步数, 当前token, tokens缓存, kv_cache, 随机种子, 完成状态标记)
        # 注意：step从task_prefix_len-1开始，因为我们已经有了"TASK:"的前缀
        initial_carry = (
            task_prefix_len - 1,  # step: 从TASK前缀的最后一个token开始
            task_prefix_tokens[-1:][None, :].repeat(batch_size, axis=0),  # cur_token: TASK的最后一个token
            init_tokens,  # tokens_buf: 所有生成的tokens缓存（已包含TASK前缀）
            init_kv_cache,  # kv_cache: 键值缓存
            rng,  # loop_rng: 随机数生成器
            jnp.zeros((batch_size,), dtype=jnp.bool_)  # has_finished: 每个样本是否已生成EOS
        )

        def body_fun(carry, loop_i):
            # loop_i: concrete Python int (loop iteration index, 0-based).
            # At entry to iteration loop_i, the KV cache holds (prefix_len + loop_i) entries
            # (prefix tokens + loop_i tokens added by previous iterations).
            # This call adds 1 more, making the total KV = prefix_len + loop_i + 1.
            step, cur_token, tokens_buf, kv_cache, loop_rng, has_finished = carry

            # 嵌入当前步 token
            cur_emb = self.PaliGemma.llm(cur_token, method="embed")

            # 构造当前步 Position
            # loop_i=0 处理最后一个 TASK: token，其在全序列中的位置是
            #   valid_prefix_lens + (task_prefix_len - 1)。
            # loop_i=1 处理第 1 个新生成 token，位置是 valid_prefix_lens + task_prefix_len。
            # 通用公式：valid_prefix_lens + task_prefix_len - 1 + loop_i（per-batch，无间隙）。
            current_pos = (valid_prefix_lens + (task_prefix_len - 1 + loop_i))[:, None]  # [batch_size, 1]

            # 构造 Attention Mask
            # 经过 1b 预热后，KV cache 在循环开始时有 (prefix_len + task_prefix_len - 1) 个 token。
            # 每次迭代向 KV 添加 1 个 token，所以 loop_i 次迭代后 KV 有
            #   (prefix_len + task_prefix_len - 1 + loop_i) 个已缓存 token。
            # 加上当前 token 本身，总 KV = prefix_len + task_prefix_len + loop_i。
            # 全 True：当前 token 可以 attend 到所有已缓存的历史 token。
            kv_total = prefix_len + task_prefix_len + loop_i
            current_mask = jnp.ones((batch_size, 1, kv_total), dtype=jnp.bool_)

            # 模型前向 (增量更新 Cache)
            # CoT tokens 是语言 tokens，走 stream 0 (PaliGemma LLM，普通 RMSNorm)。
            # stream 1 (action expert，AdaRMS) 传 None 跳过，避免 ScopeParamNotFoundError。
            (out_cot, _), next_kv_cache = self.PaliGemma.llm(
                [cur_emb, None],
                mask=current_mask,
                positions=current_pos,
                kv_cache=kv_cache,
                adarms_cond=[None, None]
            )

            # 采样逻辑：从 stream 0 的输出取 logits
            logits = out_cot[:, -1, :]  # [batch, vocab_size]
            new_rng, sample_rng = jax.random.split(loop_rng)

            # 温度采样，添加安全检查避免除以接近0的值
            safe_temperature = jnp.maximum(temperature, 1e-8)
            next_token_id = jax.lax.cond(
                temperature > 0.0,
                lambda: jax.random.categorical(sample_rng, logits / safe_temperature, axis=-1),
                lambda: jnp.argmax(logits, axis=-1)
            )
            next_token = next_token_id[:, None]  # [batch, 1]

            # 更新完成状态：如果之前已完成或现在刚生成 EOS
            new_has_finished = has_finished | (next_token_id == EOS_TOKEN)

            # 关键修复：对于已完成的样本，使用PAD token (0)
            # 这里使用new_has_finished来判断，但为了正确处理，应该在生成EOS后立即停止
            # 使用旧的has_finished，这样在生成EOS的那一步还能正常记录EOS token
            safe_next_token = jnp.where(has_finished[:, None],
                                       jnp.zeros_like(next_token),  # PAD token
                                       next_token)

            # 更新结果 Buffer
            # step从task_prefix_len-1开始，所以step+1对应正确的位置
            new_tokens_buf = tokens_buf.at[:, step + 1].set(safe_next_token.squeeze(-1))

            return (step + 1, safe_next_token, new_tokens_buf, next_kv_cache, new_rng, new_has_finished)

        # 4. 执行循环
        # jax.lax.while_loop 与 ToNNX (Linen bridge) 不兼容：body_fun 闭包捕获了 self
        # (一个 NNX 模块)，而 while_loop 要求 body_fun 是纯函数只依赖 carry。
        # Linen 的 scope 系统在 while_loop tracing 下会失效。
        # 解决方案：用 Python for 循环代替，JIT 编译时会静态展开 (unroll)，
        # NNX 模块调用在展开后的 JIT 上下文中可以正常工作。
        # 代价：编译时间随 max_cot_tokens 线性增长，运行时语义与 while_loop 等价
        # (has_finished 已负责 EOS 后的 PAD masking，不需要提前退出)。
        carry = initial_carry
        for i in range(self.max_cot_tokens):
            carry = body_fun(carry, i)
        final_step, _, all_tokens, _, _, _ = carry

        # 5. 后处理：计算 Embedding 和生成掩码
        # 由于 while_loop 返回的是固定长度的 total_max_len，为了 JIT 兼容性，我们按固定长度返回
        # 后续处理会通过 cot_mask 来忽略padding tokens
        cot_embeddings = self.PaliGemma.llm(all_tokens, method="embed")

        # 生成有效token掩码：非零token被认为是有效的（0是PAD token）
        cot_mask = all_tokens > 0  # [batch, total_max_len]

        # AR掩码：TASK前缀的tokens可以互相attend，后续生成的tokens采用因果掩码
        # 前task_prefix_len个tokens之间可以互相attend（都是False）
        # 后续生成的tokens采用因果掩码（True）
        cot_ar_mask = jnp.array([True] * total_max_len)

        return cot_embeddings, cot_mask, cot_ar_mask, all_tokens

    
    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # Embed prefix (images + prompt)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)

        # Embed suffix early so it is available for the merged LLM call below.
        # embed_suffix only depends on (observation, x_t, time) — independent of CoT.
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)

        cot_loss = 0.0
        if self.use_cot and observation.tokenized_cot_reasoning is not None:
            # Teacher-forcing: embed ground-truth CoT tokens.
            cot_target_tokens = observation.tokenized_cot_reasoning
            cot_rng, _ = jax.random.split(preprocess_rng)
            cot_embeddings, cot_mask, cot_ar_mask, _ = self.generate_cot_tokens(
                cot_rng, prefix_tokens, prefix_mask, prefix_ar_mask, cot_target_tokens=cot_target_tokens
            )

            # RC2 fix: zero-gate padding positions so they are truly inert.
            cot_embeddings = cot_embeddings * cot_mask[:, :, None].astype(cot_embeddings.dtype)

            # Build CoT-extended prefix.
            extended_prefix_tokens = jnp.concatenate([prefix_tokens, cot_embeddings], axis=1)
            extended_prefix_mask = jnp.concatenate([prefix_mask, cot_mask], axis=1)
            extended_prefix_ar_mask = jnp.concatenate([prefix_ar_mask, cot_ar_mask], axis=0)

            # ── RC5 fix: single merged forward pass ──────────────────────────────
            # Joint mask over [extended_prefix | suffix].
            #
            # Causal structure (via cumsum of ar_mask):
            #   prefix positions         cumsum = 0
            #   CoT position 0           cumsum = 0   (ar_mask=False, shares group with prefix)
            #   CoT positions 1..C-1     cumsum = 1..C-1  (ar_mask=True, causal)
            #   state / action tokens    cumsum >= C
            #
            # Consequence: CoT stream-0 positions (cumsum 0..C-1) CANNOT attend to
            # suffix positions (cumsum >= C), so the CoT logits computed below are
            # identical to those from the old separate CoT-only pass.  Action tokens
            # in stream 1 (cumsum >= C) CAN attend to the full CoT-extended prefix,
            # which is exactly the desired behaviour at inference time too.
            #
            # A single jax.grad call over (action_loss + w*cot_loss) now produces
            # one coherent gradient vector for LLM parameters instead of two
            # independently-computed gradients that could point in opposite directions.
            joint_input_mask = jnp.concatenate([extended_prefix_mask, suffix_mask], axis=1)
            joint_ar_mask = jnp.concatenate([extended_prefix_ar_mask, suffix_ar_mask], axis=0)
            joint_attn_mask = make_attn_mask(joint_input_mask, joint_ar_mask)
            joint_positions = jnp.cumsum(joint_input_mask, axis=1) - 1

            (extended_prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [extended_prefix_tokens, suffix_tokens],
                mask=joint_attn_mask,
                positions=joint_positions,
                adarms_cond=[None, adarms_cond],
            )

            # CoT loss — stream-0 output at positions [P-1 .. P+C-2] predicts tokens [0 .. C-1].
            # P = original prefix length (prefix_tokens.shape[1]), C = max_cot_tokens.
            # The slice [P-1 : -1] works because extended_prefix_out has length P+C,
            # so -1 refers to position P+C-1, giving exactly C prediction positions.
            cot_logits = extended_prefix_out[:, prefix_tokens.shape[1] - 1 : -1, :]
            cot_targets = cot_target_tokens
            log_probs = jax.nn.log_softmax(cot_logits, axis=-1)
            batch_size, seq_len = cot_targets.shape
            batch_indices = jnp.arange(batch_size)[:, None]
            seq_indices = jnp.arange(seq_len)[None, :]
            target_log_probs = log_probs[batch_indices, seq_indices, cot_targets]
            cot_target_mask = cot_targets > 0
            masked_log_probs = target_log_probs * cot_target_mask
            cot_loss = -jnp.sum(masked_log_probs) / jnp.maximum(jnp.sum(cot_target_mask), 1.0)

        else:
            # No CoT — standard single pass over prefix + suffix.
            input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
            ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
            attn_mask = make_attn_mask(input_mask, ar_mask)
            positions = jnp.cumsum(input_mask, axis=1) - 1
            (_, suffix_out), _ = self.PaliGemma.llm(
                [prefix_tokens, suffix_tokens],
                mask=attn_mask,
                positions=positions,
                adarms_cond=[None, adarms_cond],
            )

        # Compute action loss from stream-1 output (unchanged by the merge).
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        action_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)

        # Combine losses.
        if self.use_cot and observation.tokenized_cot_reasoning is not None:
            total_loss = action_loss + self.cot_loss_weight * cot_loss
            aux = {
                'cot_loss': cot_loss,
                'action_loss': jnp.mean(action_loss),
            }
        else:
            total_loss = action_loss
            aux = {
                'cot_loss': jnp.array(0.0),
                'action_loss': jnp.mean(action_loss),
            }

        return total_loss, aux

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        cot_temperature: float = 0.0,  # Temperature for CoT sampling (0.0 = greedy)
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]

        # Split RNG for noise and CoT generation
        rng, noise_rng, cot_rng = jax.random.split(rng, 3)#
        if noise is None:
            noise = jax.random.normal(noise_rng, (batch_size, self.action_horizon, self.action_dim))

        # First fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)

        # Generate CoT tokens if enabled
        if self.use_cot:
            cot_embeddings, cot_mask, cot_ar_mask, cot_tokens = self.generate_cot_tokens(
                cot_rng, prefix_tokens, prefix_mask, prefix_ar_mask,
                cot_target_tokens=None, temperature=cot_temperature
            )
            # Extend prefix with CoT embeddings
            prefix_tokens = jnp.concatenate([prefix_tokens, cot_embeddings], axis=1)
            prefix_mask = jnp.concatenate([prefix_mask, cot_mask], axis=1)
            prefix_ar_mask = jnp.concatenate([prefix_ar_mask, cot_ar_mask], axis=0)

        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
