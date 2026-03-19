# test_run.py
import time
import jax
import jax.numpy as jnp
from openpi.models.pi_cot import PiCOTConfig

print(">>> 1. Imports done")
config = PiCOTConfig(
    paligemma_variant='dummy', 
    action_expert_variant='dummy',
    action_dim=4, action_horizon=4, 
    max_token_len=8, max_cot_tokens=4, max_fast_tokens=4
)
print(">>> 2. Config created")

print(">>> 3. Creating model...")
t0 = time.time()
model = config.create(jax.random.PRNGKey(0))
jax.block_until_ready(model)
print(f">>> Model created in {time.time()-t0:.2f}s")

obs, acts = config.inputs_spec(batch_size=2)
obs = jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), obs)
acts = jnp.ones(acts.shape, acts.dtype)
print(">>> 4. Inputs ready. Starting Loss Calculation (Compilation may take a while)...")

t1 = time.time()
# 务必使用 train=True 触发完整计算图
loss, aux = model.compute_loss(jax.random.PRNGKey(1), obs, acts, train=True)
loss = jax.block_until_ready(loss)
print(f">>> Loss computed in {time.time()-t1:.2f}s")

print("✅ Success!")
print("Loss shape:", loss.shape)
print("Aux keys:", aux.keys() if isinstance(aux, dict) else "N/A")