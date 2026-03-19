compute_loss 中的注意力机制分析    

  序列结构                                                                                                                              
  
  整个 joint 序列由四段拼接而成：                                                                                                       
                  
  [ Prefix (P) | CoT (C) | FAST (F) | Suffix/Actions (H) ]
    图像+文本   | 推理token | 动作token | 噪声连续动作
    Stream-0              Stream-0   Stream-1(action expert)

  ---
  AR mask 与 cumsum 机制

  make_attn_mask 的核心逻辑（第 39-45 行）：

  cumsum = jnp.cumsum(mask_ar, axis=1)
  attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
  # token i 能 attend token j  ⟺  cumsum[j] <= cumsum[i]

  各段的 ar_mask 值：

  ┌───────────────────────────┬───────────────────────────┬───────────────┬───────────────┐
  │            段             │          ar_mask          │   cumsum 值   │     含义      │
  ├───────────────────────────┼───────────────────────────┼───────────────┼───────────────┤
  │ Prefix (P tokens)         │ [False, False, ...]       │ 全为 0        │ 同属第0组     │
  ├───────────────────────────┼───────────────────────────┼───────────────┼───────────────┤
  │ CoT (C tokens)            │ [True, True, ...]         │ 1, 2, ..., C  │ 每个token一组 │
  ├───────────────────────────┼───────────────────────────┼───────────────┼───────────────┤
  │ FAST (F tokens)           │ [True, True, ...]         │ C+1, ..., C+F │ 每个token一组 │
  ├───────────────────────────┼───────────────────────────┼───────────────┼───────────────┤
  │ Suffix/Actions (H tokens) │ [True, False, False, ...] │ 全为 C+F+1    │ 同属一组      │
  └───────────────────────────┴───────────────────────────┴───────────────┴───────────────┘

  ---
  注意力可达性矩阵

                Prefix(P)  CoT(C)   FAST(F)  Actions(H)
               ┌─────────┬────────┬─────────┬──────────┐
  Prefix(P)    │  双向 ✓  │   ✗    │    ✗    │    ✗     │  Stream-0
               ├─────────┼────────┼─────────┼──────────┤
  CoT(C)       │   ✓      │ 因果 ✓ │    ✗    │    ✗     │  Stream-0
               ├─────────┼────────┼─────────┼──────────┤
  FAST(F)      │   ✓      │  ✓(全) │ 因果 ✓  │    ✗     │  Stream-0
               ├─────────┼────────┼─────────┼──────────┤
  Actions(H)   │   ✓      │  ✓(全) │  ✗(禁!) │  双向 ✓  │  Stream-1
               └─────────┴────────┴─────────┴──────────┘

  ---
  各段详细说明

  1. Prefix（双向）
  - cumsum=0，所有 Prefix token 同组，相互双向 attend
  - 作为静态上下文，不能看后续任何内容

  2. CoT（因果）
  - 每个 CoT_i 的 cumsum=i，可以看到：
    - 所有 Prefix（cumsum=0 ≤ i）
    - 所有位置 ≤ i 的 CoT（标准因果自回归）
  - 不能看 FAST 或 Actions

  3. FAST（因果）
  - 每个 FAST_i 的 cumsum=C+i，可以看到：
    - 所有 Prefix、所有 CoT（条件生成动作token）
    - 位置 ≤ i 的 FAST（因果）
  - 不能看 Actions

  4. Actions/Suffix（双向 + 信息泄露屏蔽）
  - 基础 mask：所有 Action token 的 cumsum=C+F+1（相同），故内部双向 attend
  - 能看 Prefix 和所有 CoT
  - 关键屏蔽（第 322-330 行）：

  is_suffix_row = all_idx >= (P + C + F)     # Action expert 的查询位置
  is_fast_col = (all_idx >= (P + C)) & (all_idx < (P + C + F))  # FAST token 位置
  no_leak = ~(is_suffix_row[:, None] & is_fast_col[None, :])
  attn_mask = attn_mask & no_leak[None, :, :]

  Stream-1（action expert）被显式阻止看 FAST tokens，原因在文档注释中说明：防止目标动作编码（FAST tokens 是 ground truth
  动作的离散编码）泄漏到 flow-matching 去噪流中，否则 stream-1 可以"作弊"直接复制动作信息。