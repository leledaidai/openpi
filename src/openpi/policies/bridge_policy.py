import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_bridge_example() -> dict:
    """Creates a random input example for the Bridge-style policy."""
    return {
        "observation/state": np.random.rand(8),
        "observation/primary_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
        # Optional for training:
        # "actions": np.random.randn(10, 7).astype(np.float32),
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _pad_or_truncate_actions(actions: np.ndarray, action_dim: int) -> np.ndarray:
    """
    Ensures actions have last-dim == action_dim by truncating or zero-padding.
    Expected shape: [T, D] (or anything where last dim is D).
    """
    actions = np.asarray(actions)
    if actions.ndim < 2:
        raise ValueError(f"Expected actions with shape [T, D], got {actions.shape}")

    d = actions.shape[-1]
    if d == action_dim:
        return actions

    if d > action_dim:
        return actions[..., :action_dim]

    pad_width = [(0, 0)] * actions.ndim
    pad_width[-1] = (0, action_dim - d)
    return np.pad(actions, pad_width=pad_width, mode="constant", constant_values=0.0)


@dataclasses.dataclass(frozen=True)
class BridgeInputs(transforms.DataTransformFn):
    """
    Convert dataset dict -> model inputs (training + inference).

    Expected keys after your RepackTransform:
      - "observation/primary_image"  (uint8 HWC or float CHW/HWC)
      - "observation/state"
      - "prompt" (optional but typical)
      - "actions" (training only)
    """

    action_dim: int
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/primary_image"])

        # Bridge dataset (as you repack) usually doesn't provide wrist images.
        # We pad them with zeros so the model interface remains consistent.
        left_wrist = np.zeros_like(base_image)
        right_wrist = np.zeros_like(base_image)

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist,
                "right_wrist_0_rgb": right_wrist,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                # same masking rule as libero_policy:
                # - For PI0_FAST: keep masks True even for padded images
                # - For PI0 (non-fast): mask padded images False
                "left_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # Actions are only present in training; pad/truncate to action_dim
        if "actions" in data:
            inputs["actions"] = _pad_or_truncate_actions(data["actions"], self.action_dim)

        # Pass prompt through if present
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        # Always add cot_reasoning to maintain consistent dict structure for batching
        # Use empty bytes if not present (will be padded by TokenizeCoTReasoning)
        inputs["cot_reasoning"] = data.get("cot_reasoning", b"")

        return inputs


@dataclasses.dataclass(frozen=True)
class BridgeOutputs(transforms.DataTransformFn):
    """
    Convert model outputs -> dataset-specific format (inference only).

    By default, return first 7 dims (typical 7-DoF action). If your Bridge actions
    are a different dim, change default_action_out_dim accordingly, or pass a different value.
    """

    default_action_out_dim: int = 7

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        return {"actions": actions[:, : self.default_action_out_dim]}
