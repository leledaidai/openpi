# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

openpi is Physical Intelligence's open-source robotics repository containing Vision-Language-Action (VLA) models:
- **π₀**: Flow-based VLA model
- **π₀-FAST**: Autoregressive VLA with FAST action tokenizer
- **π₀.₅**: Upgraded π₀ with better generalization via knowledge insulation

The repository supports both **JAX** (primary) and **PyTorch** implementations, with fine-tuning capabilities for various robot platforms (DROID, ALOHA, LIBERO, UR5).

## Core Architecture

### Model Hierarchy

```
BaseModel (src/openpi/models/model.py)
├── Pi0 (src/openpi/models/pi0.py) - Flow matching model
├── Pi0Fast (src/openpi/models/pi0_fast.py) - Autoregressive model
└── Pi0Pytorch (src/openpi/models_pytorch/) - PyTorch implementations
```

**Key abstractions:**
- `BaseModel`: Abstract base with `compute_loss()` and `sample_actions()` methods
- `Observation`: Structured input containing images, state, tokenized prompts, and optional CoT reasoning
- `Actions`: Model output actions with shape `[batch, action_horizon, action_dim]`

### Policy System

```
BasePolicy (openpi_client package)
└── Policy (src/openpi/policies/policy.py)
    ├── Wraps a model with input/output transforms
    ├── Handles both JAX and PyTorch models
    └── Provides unified infer() interface
```

**Platform-specific policies** (in `src/openpi/policies/`):
- `droid_policy.py`: DROID robot (Franka arm)
- `aloha_policy.py`: ALOHA bimanual platform
- `libero_policy.py`: LIBERO simulation benchmark
- `bridge_policy.py`: Bridge dataset integration

Each policy defines:
- `Inputs`: Maps environment observations to model format
- `Outputs`: Maps model actions to robot commands

### Training Pipeline

```
TrainConfig (src/openpi/training/config.py)
├── model_config: Pi0Config with architecture settings
├── data_configs: List of DataConfig for multi-dataset training
├── optimizer_config: Learning rate, weight decay, etc.
└── weight_loader: Loads pretrained weights (base models)
```

**Data flow:**
1. **Raw data** → LeRobot dataset or RLDS format
2. **DataConfig transforms**:
   - `repack_transforms`: Dataset-specific format conversion
   - `data_transforms`: Robot-specific preprocessing
   - `model_transforms`: Model-specific augmentation
3. **Normalization**: Z-score or quantile normalization using precomputed stats
4. **Model input**: `Observation` + `Actions` objects

### Chain-of-Thought (CoT) Integration

The codebase includes CoT reasoning capabilities:
- **Training**: Uses teacher forcing with ground truth CoT tokens (`tokenized_cot_reasoning`)
- **Inference**: Autoregressive generation with KV cache optimization
- **Key files**:
  - `src/openpi/models/pi0.py`: `generate_cot_tokens()` method (lines 193-311)
  - `src/openpi/training/data_loader.py`: CoT data loading
  - `src/openpi/transforms.py`: CoT tokenization transforms

**Important**: When modifying CoT generation, ensure:
- Temperature scaling uses safe minimum (1e-8) to avoid numerical issues
- EOS token handling preserves the token before padding
- KV cache is properly initialized with prefix tokens

## Common Commands

### Environment Setup

```bash
# Clone with submodules
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git
git submodule update --init --recursive

# Install dependencies
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# For RLDS data loading (DROID full dataset)
uv sync --group rlds
```

### Training

```bash
# Compute normalization statistics (required before training)
uv run scripts/compute_norm_stats.py --config-name <config_name>

# JAX training
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py <config_name> --exp-name=<name> [--overwrite]

# PyTorch training (single GPU)
uv run scripts/train_pytorch.py <config_name> --exp_name <name> [--resume]

# PyTorch multi-GPU (single node)
uv run torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <name>

# With FSDP (reduces memory, slower training)
uv run scripts/train.py <config_name> --exp-name=<name> --fsdp-devices <num_gpus>
```

### Inference

```bash
# Serve policy (for robot deployment or evaluation)
uv run scripts/serve_policy.py policy:checkpoint --policy.config=<config_name> --policy.dir=<checkpoint_dir>

# Test inference without robot
uv run examples/simple_client/test_inference.py --config <config_name>
```

### PyTorch-Specific

```bash
# Convert JAX checkpoint to PyTorch
uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir <jax_checkpoint> \
    --config_name <config_name> \
    --output_path <pytorch_checkpoint>

# Apply transformers library patches (required for PyTorch)
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
```

### Testing

```bash
# Run tests
uv run pytest

# Run specific test file
uv run pytest src/openpi/models/model_test.py

# Lint code
uv run ruff check .
uv run ruff format .
```

## Configuration System

Configs are defined in `src/openpi/training/config.py` using dataclasses. Key patterns:

### Creating a New Config

```python
@dataclasses.dataclass(frozen=True)
class MyRobotDataConfig(DataConfig):
    repo_id: str = "my_org/my_robot_dataset"
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=lambda: _transforms.Group([
        # Convert dataset format to standard format
    ]))
    data_transforms: _transforms.Group = dataclasses.field(default_factory=lambda: _transforms.Group([
        # Robot-specific transforms (e.g., action space conversion)
    ]))

@dataclasses.dataclass(frozen=True)
class MyTrainConfig(TrainConfig):
    model_config: pi0_config.Pi0Config = pi0_config.Pi0Config(
        action_dim=7,
        action_horizon=10,
        # ... other model settings
    )
    data_configs: Sequence[DataConfig] = (MyRobotDataConfig(),)
    weight_loader: weight_loaders.WeightLoader = weight_loaders.Pi0BaseWeightLoader()
```

Register in `_CONFIGS` dict at bottom of `config.py`.

### Multi-Dataset Training (Co-training)

```python
data_configs: Sequence[DataConfig] = (
    DataConfig1(repo_id="dataset1", ...),
    DataConfig2(repo_id="dataset2", ...),
)
```

Data loader samples from datasets according to their relative sizes.

## Key Implementation Details

### JAX vs PyTorch

**JAX** (default):
- Uses Flax NNX for model definition
- JIT compilation via `nnx_utils.module_jit()`
- Mixed precision: weights/gradients in float32, activations in bfloat16
- FSDP support for multi-GPU training

**PyTorch**:
- Requires transformers library patches for AdaRMS and precision control
- Full bfloat16 or float32 training (no mixed precision yet)
- No FSDP, LoRA, or EMA support currently
- Comparable inference speed with torch.compile

### Normalization Statistics

Stored in `<checkpoint_dir>/assets/<asset_id>/norm_stats.json`:
- Computed via `scripts/compute_norm_stats.py`
- Can be reloaded from base model using `AssetsConfig`
- Contains `q01`, `q99`, `std` for each state/action dimension
- **Watch for**: Very small values can cause divergence after normalization

### Weight Loading

`WeightLoader` classes (in `src/openpi/training/weight_loaders.py`):
- `Pi0BaseWeightLoader`: Loads π₀ base model weights
- `Pi05BaseWeightLoader`: Loads π₀.₅ base model weights
- `PytorchWeightLoader`: Loads converted PyTorch checkpoints
- Supports partial loading (e.g., freeze vision encoder, load only LLM)

### Image Processing

- All models expect 3 camera views: `base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb`
- Resolution: 224×224 (defined in `model.IMAGE_RESOLUTION`)
- Format: float32 in [-1, 1] or uint8 in [0, 255] (converted automatically)
- Missing views should have `image_mask` set to False

### Action Spaces

Different robots use different action representations:
- **DROID**: Joint positions (7D) or joint velocities (7D) + gripper (1D)
- **ALOHA**: Bimanual joint positions (14D)
- **LIBERO**: Joint positions (7D) in simulation

Transforms in policy classes handle conversion between robot commands and model actions.

## File Organization

```
openpi/
├── src/openpi/
│   ├── models/          # Model architectures (JAX)
│   │   ├── model.py     # BaseModel, Observation, Actions
│   │   ├── pi0.py       # Flow matching model
│   │   ├── pi0_fast.py  # Autoregressive model
│   │   ├── gemma.py     # LLM backbone
│   │   └── siglip.py    # Vision encoder
│   ├── models_pytorch/  # PyTorch implementations
│   ├── policies/        # Robot-specific policies
│   ├── training/        # Training infrastructure
│   │   ├── config.py    # All training configs
│   │   ├── data_loader.py
│   │   ├── droid_rlds_dataset.py
│   │   └── bridge_rlds_dataset.py
│   ├── transforms.py    # Data transformation pipeline
│   ├── serving/         # Policy server for deployment
│   └── shared/          # Utilities (normalization, download, etc.)
├── scripts/
│   ├── train.py         # JAX training script
│   ├── train_pytorch.py # PyTorch training script
│   ├── compute_norm_stats.py
│   └── serve_policy.py
├── examples/            # Platform-specific examples
│   ├── droid/
│   ├── aloha_real/
│   ├── aloha_sim/
│   ├── libero/
│   └── ur5/
└── docs/                # Additional documentation
```

## Troubleshooting

### GPU Memory Issues
- Set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` (or higher)
- Use `--fsdp-devices <n>` for fully-sharded data parallelism
- Disable EMA in config if needed
- For PyTorch: use bfloat16 precision

### CUDA Errors
- Uninstall system CUDA libraries (uv installs its own)
- For Docker: ensure nvidia-container-toolkit is installed
- Verify GPU compatibility (requires NVIDIA GPU with >8GB memory)

### Diverging Loss
- Check `norm_stats.json` for very small `q01`, `q99`, or `std` values
- Manually adjust problematic dimensions
- Verify action space matches robot platform

### Import Errors
- Run `uv sync` to ensure all dependencies installed
- For RLDS: `uv sync --group rlds`
- For PyTorch: apply transformers patches

### Dataset Issues
- For HuggingFace datasets: `huggingface-cli login`
- Verify `repo_id` in DataConfig matches dataset name
- Check `action_sequence_keys` matches dataset structure

## Remote Inference

The policy server (`scripts/serve_policy.py`) enables running models on a separate GPU server:
- Server listens on port 8000 (configurable)
- Client sends observations via websocket
- See `docs/remote_inference.md` for implementation details
- Useful for keeping robot and model environments separate

## Additional Resources

- Main README: `README.md`
- Docker setup: `docs/docker.md`
- Normalization stats: `docs/norm_stats.md`
- Remote inference: `docs/remote_inference.md`
- Platform-specific guides: `examples/<platform>/README.md`
- DROID training: `examples/droid/README_train.md`
