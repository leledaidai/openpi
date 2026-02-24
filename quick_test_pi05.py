#!/usr/bin/env python3
"""
Quick test - skips CoT generation to test model loading only.
This helps identify if the issue is with CoT generation or model loading.
"""

import sys
import time
from pathlib import Path

# Add openpi to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("=" * 80)
print("QUICK TEST - MODEL LOADING ONLY (NO COT)")
print("=" * 80)

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

print(f"\nJAX devices: {jax.devices()}")
print(f"Using device: {jax.devices()[0]}")

from openpi.training import config as train_config
import openpi.models.pi0 as pi0_model
import openpi.models.tokenizer as tokenizer
from openpi.models.model import Observation, preprocess_observation
from flax import nnx
import orbax.checkpoint as ocp

print("\n" + "=" * 80)
print("Step 1/5: Loading config...")
print("=" * 80)
config = train_config.get_config('pi05_bridge_rlds_finetune_cot')
print(f"✓ Config loaded (CoT enabled: {config.model.use_cot})")

print("\n" + "=" * 80)
print("Step 2/5: Initializing model...")
print("=" * 80)

# Use absolute path (required by Orbax)
checkpoint_path = Path("checkpoints/pi05_bridge_rlds_finetune_cot/pi05_bridge_rlds_finetune_cot/29999/params").resolve()
print(f"Checkpoint path: {checkpoint_path}")

rng = nnx.Rngs(0)

start = time.time()
model = pi0_model.Pi0(config.model, rngs=rng)
print(f"✓ Model initialized in {time.time() - start:.2f}s")

print("\n" + "=" * 80)
print("Step 3/5: Loading checkpoint...")
print("=" * 80)
print(f"Note: Checkpoint was saved on 8 GPUs, restoring on {len(jax.devices())} GPU(s)")
print(f"This is OK - will create replicated sharding for single GPU")

# Create proper sharding for restoration
from jax.sharding import NamedSharding, Mesh, PartitionSpec

# Create mesh with single device
devices = jax.devices()
mesh = Mesh(devices, axis_names=('replica',))

# Create replicated sharding (all params on all devices - in this case just 1)
replicated_sharding = NamedSharding(mesh, PartitionSpec())

# Create sharding tree matching the target structure
def make_sharding_tree(pytree):
    """Create a sharding tree where all leaves use replicated sharding."""
    return jax.tree.map(lambda x: replicated_sharding, pytree)

handler = ocp.PyTreeCheckpointHandler()
target_params = nnx.state(model)

# Create restore args with proper sharding
from orbax.checkpoint import ArrayRestoreArgs
restore_args = jax.tree.map(
    lambda x: ArrayRestoreArgs(sharding=replicated_sharding),
    target_params
)

target = {"params": target_params}
restore_args_dict = {"params": restore_args}

start = time.time()
# Restore with explicit sharding
restored = handler.restore(
    checkpoint_path,
    item=target,
    restore_args=restore_args_dict
)
nnx.update(model, restored["params"])
model.eval()
print(f"✓ Checkpoint loaded in {time.time() - start:.2f}s")

print("\n" + "=" * 80)
print("Step 4/5: Preparing input...")
print("=" * 80)
# Load and preprocess image
image = Image.open("test_obs.png")
resized_image = image.resize((224, 224))
img_array = np.array(resized_image).astype(np.float32) / 255.0
img_array = (img_array - 0.5) / 0.5

images = np.stack([
    img_array,
    np.zeros((224, 224, 3), dtype=np.float32),
    np.zeros((224, 224, 3), dtype=np.float32),
])

# Tokenize
tok = tokenizer.PaligemmaTokenizer(max_len=config.model.max_token_len)

# Attach tokenizer to model for CoT generation
model.PaliGemma.llm.tokenizer = tok._tokenizer

instruction = "place the watermelon on the towel"
state = np.zeros(7, dtype=np.float32)

if config.model.discrete_state_input:
    prompt_tokens, _ = tok.tokenize(instruction, state=state)
else:
    prompt_tokens, _ = tok.tokenize(instruction, state=None)

# Create observation WITHOUT CoT
obs = Observation(
    images={
        "base_0_rgb": jnp.array(images[0][None]),  # Primary view, shape (1, 224, 224, 3)
        "left_wrist_0_rgb": jnp.array(images[1][None]),  # Left wrist (zeros)
        "right_wrist_0_rgb": jnp.array(images[2][None]),  # Right wrist (zeros)
    },
    image_masks={
        "base_0_rgb": jnp.array([True]),
        "left_wrist_0_rgb": jnp.array([False]),
        "right_wrist_0_rgb": jnp.array([False]),
    },
    tokenized_prompt=jnp.array(prompt_tokens[None]),
    tokenized_prompt_mask=jnp.ones((1, len(prompt_tokens)), dtype=bool),
    state=jnp.array(state[None]),
    tokenized_cot_reasoning=None,  # No CoT for quick test
)
print(f"✓ Input prepared")

print("\n" + "=" * 80)
print("Step 4.5/5: Generating and decoding CoT...")
print("=" * 80)

if model.use_cot:
    # Preprocess obs the same way sample_actions does internally
    processed_obs = preprocess_observation(None, obs, train=False)
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(processed_obs)

    cot_rng, _ = jax.random.split(rng())
    _, _, _, cot_tokens = model.generate_cot_tokens(
        cot_rng, prefix_tokens, prefix_mask, prefix_ar_mask,
        cot_target_tokens=None, temperature=0.3,
    )
    # cot_tokens: int32 [batch, total_max_len]; 0 = PAD token
    token_ids = np.array(cot_tokens[0]).tolist()
    valid_ids = [t for t in token_ids if t > 0]  # strip PAD (0)
    cot_text = tok._tokenizer.decode(valid_ids)
    print(f"\nCoT reasoning (raw token ids: {len(valid_ids)} tokens):")
    print("-" * 80)
    print(cot_text)
    print("-" * 80)
else:
    print("CoT disabled (model.use_cot is False)")

print("\n" + "=" * 80)
print("Step 5/5: Running inference (WITH CoT)...")
print("=" * 80)

print(f"\nNote: CoT enabled: {model.use_cot}")

start = time.time()
try:
    # Get a JAX random key from the Rngs object
    sample_key = rng()
    actions = model.sample_actions(sample_key, obs, num_steps=10)
    elapsed = time.time() - start
    print(f"✓ Actions sampled in {elapsed:.2f}s")
    print(f"  Action shape: {actions.shape}")
    print(f"  First action: {actions[0, 0]}")

    print("\n" + "=" * 80)
    print("✓ QUICK TEST PASSED!")
    print("=" * 80)
    print("\nModel loading and inference work correctly.")

except Exception as e:
    print(f"\n✗ Error during inference: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
