# CoT Generation 加速：问题分析与修改说明

## 问题：CoT 生成极慢

`generate_cot_tokens` 中有一个 Python for 循环：

```python
for i in range(self.max_cot_tokens):  # max_cot_tokens = 768
    carry = body_fn(carry, i)
```

JAX JIT 编译时会将 Python 循环**静态展开（unroll）**，即把 768 次完整的 PaliGemma 2B 前向传播全部写入 XLA 计算图。结果：

- 编译时间：O(N)，N = 768，极长
- XLA 图大小：768 × 完整 LLM 前向，内存占用巨大
- 运行时无法利用循环优化

## 根本原因：KV Cache 形状动态增长

要用 `jax.lax.scan` 替换 Python 循环，carry 中所有张量的形状必须在编译时静态确定。但 `gemma.py` 的 `Attention.__call__` 每步都通过 `jnp.concatenate` 增长 KV cache：

```python
# gemma.py Attention.__call__ — 原始代码
k = jnp.concatenate([cache_k, k], axis=1)  # T 维度每步 +1
v = jnp.concatenate([cache_v, v], axis=1)
```

第 `i` 步后 KV cache 形状为 `[l, b, prefix_len + i, k, h]`，形状随 `i` 变化 → `jax.lax.scan` 无法使用。

## 为什么必须修改 gemma.py

KV cache 的增长逻辑在 `gemma.py` 的 `Attention` 模块内部。`pi_cot.py` 只能传入 cache，无法控制 `Attention` 内部如何更新它。

如果不修改 `gemma.py`，唯一的替代方案是：

1. **每步不传 KV cache，从头重算**：正确性不变，但速度更慢（O(T²) 复杂度）
2. **每步手动截取 cache 传入**：`Attention` 仍然 concatenate，形状仍然增长，scan 仍然不可用
3. **减小 `max_cot_tokens`**：治标不治本，用户明确拒绝

因此，必须在 `Attention` 内部增加一种**原地写入（scatter-write）**模式，使 KV cache 形状保持静态。

## 修改内容

### `gemma.py`

**1. `Attention.__call__`** — 新增 `kv_write_index` 参数

```python
def __call__(self, xs, positions, attn_mask, kv_cache, kv_write_index=None):
    ...
    if kv_cache is not None:
        cache_k, cache_v = kv_cache
        if kv_write_index is not None:
            # scatter-write 模式：原地写入预分配的 cache，形状不变
            k = jax.lax.dynamic_update_slice(cache_k, k, (0, kv_write_index, 0, 0))
            v = jax.lax.dynamic_update_slice(cache_v, v, (0, kv_write_index, 0, 0))
        else:
            # 原有 concatenate 模式，保持向后兼容
            k = jnp.concatenate([cache_k, k], axis=1)
            v = jnp.concatenate([cache_v, v], axis=1)
```

- `kv_write_index=None`（默认）：行为与原来完全一致，不影响训练和其他推理路径
- `kv_write_index=i`（JAX traced integer）：用 `dynamic_update_slice` 在位置 `i` 原地写入，cache 形状固定为 `T_max`

**2. `Block.__call__`** — 透传 `kv_write_index`

```python
def __call__(self, xs, kv_cache, positions, attn_mask, adarms_cond, deterministic=True, kv_write_index=None):
    ...
    post_attn, kv_cache = attn(pre_attn, positions, attn_mask, kv_cache, kv_write_index)
```

**3. `Module.setup` 中的 `nn.scan`** — 新增第 6 个 broadcast 输入轴

```python
in_axes=(
    0,            # kv_cache: 按层扫描
    nn.broadcast, # positions
    nn.broadcast, # mask
    nn.broadcast, # adarms_cond
    nn.broadcast, # deterministic
    nn.broadcast, # kv_write_index  ← 新增
),
```

**4. `Module.__call__`** — 新增 `kv_write_index` 参数并传入 `self.layers`

```python
def __call__(self, ..., kv_write_index: int | None = None, ...):
    ...
    embedded, kv_cache = self.layers(..., deterministic, kv_write_index)
```

### `pi_cot.py`

**预分配 KV cache + 使用 `jax.lax.scan`**

```python
# 1. 预分配完整大小的 KV cache（形状静态）
T_max = prefix_len + seed_len + self.max_cot_tokens
pad_len = T_max - cache_k.shape[2]
full_kv_cache = (
    jnp.concatenate([cache_k, zeros(pad_len)], axis=2),  # [l, b, T_max, k, h]
    jnp.concatenate([cache_v, zeros(pad_len)], axis=2),
)

# 2. scan body：每步在固定位置写入，形状不变
def scan_body(carry, loop_i):
    write_idx = prefix_len + warm_len + loop_i
    cur_mask = jnp.arange(T_max) <= write_idx  # 只 attend 到当前位置
    (out0, _), next_kvc = self.PaliGemma.llm(
        ..., kv_cache=kvc, kv_write_index=write_idx, ...
    )
    ...
    return carry, None

# 3. 用 scan 替换 Python for 循环
(_, all_ids, _, _, _), _ = jax.lax.scan(
    scan_body, carry, jnp.arange(self.max_cot_tokens)
)
```

### `config.py`

将之前错误减小的配置值恢复：
- `max_cot_tokens`: 256 → **768**
- `max_fast_tokens`: 64 → **128**

## 效果对比

| | 修改前 | 修改后 |
|---|---|---|
| XLA 图大小 | 768 × LLM forward | 1 × LLM forward（循环体） |
| JIT 编译时间 | O(N)，极长 | O(1) |
| 运行时复杂度 | 相同 | 相同 |
| 向后兼容性 | — | 完全兼容（`kv_write_index=None` 默认） |
| `max_cot_tokens` | 需要减小才能用 | 可保持 768 |
