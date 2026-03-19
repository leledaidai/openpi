#!/usr/bin/env python3
"""
Pi0-FAST-Implicit inference test script.

Pipeline:
  1. Encode images + prefix-only prompt
  2. Generate K latent steps
  3. Decode each latent step into readable reasoning text with decoder_llm
  4. Decode autoregressive FAST action tokens
  5. Convert FAST action tokens into continuous actions

Usage:
  python test_pi0_fast_implicit.py --checkpoint /inspire/hdd/global_user/gongjingjing-25039/zhdai/openpi/checkpoints/pi0_fast_bridge_rlds_implicit_cot/pi_latent_cot/70000 --image test_obs.png --instruction "place the watermelon on the towel"

  # Or using exp-name + step:
  python test_pi0_fast_implicit.py --exp my_exp --step 29999
"""

import argparse
from pathlib import Path
import sys
import textwrap
import time

import cv2
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

sys.path.insert(0, str(Path(__file__).parent / "src"))


COT_SECTION_LABELS = [
    "TASK",
    "PLAN",
    "VISIBLE OBJECTS",
    "SUBTASK REASONING",
    "SUBTASK",
    "MOVE REASONING",
    "MOVE",
    "GRIPPER POSITION",
    "ACTION",
]


def decode_reasoning_sections(pg_tokenizer, reasoning_tokens: np.ndarray) -> list[str]:
    """Decode one sample's latent reasoning sections from token ids to text."""
    sections = []
    for section_tokens in np.asarray(reasoning_tokens, dtype=np.int32):
        valid_tokens = [int(t) for t in section_tokens.tolist() if int(t) > 0]
        sections.append(pg_tokenizer.decode(valid_tokens) if valid_tokens else "")
    return sections


def decode_action_tokens(
    fast_tokenizer,
    action_tokens: np.ndarray,
    *,
    action_horizon: int,
    action_dim: int,
) -> np.ndarray:
    """Decode FAST action tokens into continuous actions."""
    return fast_tokenizer.extract_actions(
        np.asarray(action_tokens, dtype=np.int32),
        action_horizon,
        action_dim,
    )


def resolve_checkpoint_path(path: Path) -> Path:
    """Accept either a step directory or a params directory."""
    if path.name == "params":
        return path
    return path / "params"


def _section_label(index: int, total_sections: int) -> str:
    if total_sections == len(COT_SECTION_LABELS) and index < len(COT_SECTION_LABELS):
        return COT_SECTION_LABELS[index]
    return f"LATENT {index}"


def make_text_panel(sections: list[tuple[str, str]], width: int, height: int, font_size: int = 13) -> np.ndarray:
    base = Image.fromarray(np.full((height, width, 3), 255, dtype=np.uint8))
    draw = ImageDraw.Draw(base)
    try:
        font_header = ImageFont.load_default(size=font_size + 1)
        font_body = ImageFont.load_default(size=font_size)
    except TypeError:
        font_header = ImageFont.load_default()
        font_body = font_header

    x, y = 12, 10
    line_h = font_size + 4
    max_chars = max((width - 2 * x) // (font_size // 2 + 1), 20)

    for header, body in sections:
        if y + line_h > height - 4:
            break
        draw.text((x, y), header, fill=(0, 60, 160), font=font_header)
        y += line_h
        body_clean = body.strip().replace("\n", " ")
        for line in (textwrap.wrap(body_clean, width=max_chars) or ["(empty)"]):
            if y + line_h > height - 4:
                draw.text((x, y), "...", fill=(100, 100, 100), font=font_body)
                break
            draw.text((x + 6, y), line, fill=(30, 30, 30), font=font_body)
            y += line_h
        y += 4

    return np.array(base, dtype=np.uint8)[:, :, ::-1]


def build_annotated_image(
    raw_uint8: np.ndarray,
    reasoning_sections: list[str],
    actions_np: np.ndarray,
    instruction: str,
    elapsed: float,
    token_count: int,
    display_w: int = 480,
    display_h: int = 360,
) -> np.ndarray:
    obs_bgr = cv2.resize(raw_uint8[:, :, ::-1].astype(np.uint8), (display_w, display_h))

    cv2.rectangle(obs_bgr, (0, 0), (display_w, 22), (0, 0, 0), -1)
    cv2.putText(
        obs_bgr,
        f"Task: {instruction}",
        (6, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.rectangle(obs_bgr, (0, display_h - 22), (display_w, display_h), (0, 0, 0), -1)
    cv2.putText(
        obs_bgr,
        f"Gen: {elapsed:.2f}s  action_tokens={token_count}  shape={actions_np.shape}",
        (6, display_h - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    sections = [
        (f"{_section_label(i, len(reasoning_sections))}:", text.strip() or "(empty)")
        for i, text in enumerate(reasoning_sections)
    ]
    if len(actions_np) > 0:
        a0 = "  ".join(f"{v:+.3f}" for v in actions_np[0])
        sections.append(("Action t=0:", a0))

    text_panel = make_text_panel(sections, display_w, display_h)
    return np.concatenate([obs_bgr, text_panel], axis=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pi0-FAST-Implicit inference test: latent reasoning decode + FAST action decode."
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint params directory.")
    parser.add_argument("--exp", type=str, default=None, help="Experiment name under checkpoints/<config>/")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step number.")
    parser.add_argument(
        "--config",
        type=str,
        default="pi0_fast_bridge_rlds_implicit_cot",
        help="Training config name from openpi.training.config",
    )
    parser.add_argument("--image", type=str, default="test_obs.png", help="Observation image path.")
    parser.add_argument("--instruction", type=str, required=True, help="Language instruction.")
    parser.add_argument(
        "--max-decoding-steps",
        type=int,
        default=80,
        help="Maximum FAST action token decoding steps.",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Action token sampling temperature.")
    parser.add_argument("--output", type=str, default="pi0_fast_implicit_result.png", help="Output visualization path.")
    return parser.parse_args()


def main() -> None:
    from flax import nnx
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec
    import orbax.checkpoint as ocp
    from orbax.checkpoint import ArrayRestoreArgs

    from openpi.models import model as _model
    from openpi.models import pi0_fast_implicit as pi0_fast_implicit_model
    from openpi.models import tokenizer as _tokenizer
    from openpi.training import config as train_config

    args = parse_args()
    w = 80

    print("=" * w)
    print("  PI0-FAST-IMPLICIT INFERENCE TEST")
    print("  Latent reasoning decode via decoder_llm + FAST action token decode")
    print("=" * w)
    print(f"\nJAX devices : {jax.devices()}")

    print(f"\n{'-'*w}")
    print("STEP 1 / 5  |  Configuration")
    print(f"{'-'*w}")
    cfg = train_config.get_config(args.config)
    mc = cfg.model
    assert isinstance(mc, pi0_fast_implicit_model.Pi0FASTImplicitConfig), (
        f"Expected Pi0FASTImplicitConfig, got {type(mc)}"
    )

    print(f"  config name              : {args.config}")
    print(f"  action_dim               : {mc.action_dim}")
    print(f"  action_horizon           : {mc.action_horizon}")
    print(f"  max_prefix_len           : {mc.max_prefix_len}")
    print(f"  max_action_token_len     : {mc.max_action_token_len}")
    print(f"  num_latent               : {mc.num_latent}")
    print(f"  max_cot_step_len         : {mc.max_cot_step_len}")
    print(f"  max_decoding_steps       : {args.max_decoding_steps}")
    print(f"  temperature              : {args.temperature}")

    print(f"\n{'-'*w}")
    print("STEP 2 / 5  |  Initialize model")
    print(f"{'-'*w}")
    t0 = time.time()
    rngs = nnx.Rngs(0)
    model = pi0_fast_implicit_model.Pi0FASTImplicit(mc, rngs=rngs)
    print(f"  Initialized in {time.time() - t0:.1f}s")

    print(f"\n{'-'*w}")
    print("STEP 3 / 5  |  Load checkpoint")
    print(f"{'-'*w}")
    if args.checkpoint:
        ckpt_path = resolve_checkpoint_path(Path(args.checkpoint).resolve())
    elif args.exp is not None and args.step is not None:
        ckpt_path = resolve_checkpoint_path(Path(f"checkpoints/{args.config}/{args.exp}/{args.step}").resolve())
    else:
        print(
            "  ERROR: supply --checkpoint <path> OR both --exp <name> --step <n>\n"
            f"  Expected: checkpoints/{args.config}/<exp>/<step>/params"
        )
        sys.exit(1)

    print(f"  Path: {ckpt_path}")
    if not ckpt_path.exists():
        print(f"  ERROR: not found: {ckpt_path}")
        sys.exit(1)

    devices = jax.devices()
    mesh = Mesh(devices, axis_names=("replica",))
    sharding = NamedSharding(mesh, PartitionSpec())
    handler = ocp.PyTreeCheckpointHandler()
    state = nnx.state(model)
    restore_args = jax.tree.map(lambda _: ArrayRestoreArgs(sharding=sharding), state)

    t0 = time.time()
    restored = handler.restore(
        ckpt_path,
        item={"params": state},
        restore_args={"params": restore_args},
    )
    nnx.update(model, restored["params"])
    model.eval()
    print(f"  Loaded in {time.time() - t0:.1f}s  ({len(devices)} device(s))")

    print(f"\n{'-'*w}")
    print("STEP 4 / 5  |  Prepare observation")
    print(f"{'-'*w}")
    img_path = Path(args.image)
    if not img_path.exists():
        print(f"  WARNING: '{img_path}' not found - using random noise.")
        raw_uint8 = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    else:
        raw_uint8 = np.array(Image.open(img_path).convert("RGB").resize((224, 224)), dtype=np.uint8)

    img_float = raw_uint8.astype(np.float32) / 127.5 - 1.0
    dummy = np.zeros((224, 224, 3), dtype=np.float32)

    tokenizer_cls = _tokenizer.FASTTokenizer if mc.fast_model_tokenizer is None else mc.fast_model_tokenizer
    tokenizer_kwargs = {} if mc.fast_model_tokenizer_kwargs is None else mc.fast_model_tokenizer_kwargs
    tok = tokenizer_cls(mc.max_token_len, **tokenizer_kwargs)
    robot_state = np.zeros(mc.action_dim, dtype=np.float32)
    tok_out = tok.tokenize_implicit(
        args.instruction,
        robot_state,
        actions=None,
        cot_reasoning=None,
        num_latent=mc.num_latent,
        max_prefix_len=mc.max_prefix_len,
        max_action_token_len=mc.max_action_token_len,
        max_cot_step_len=mc.max_cot_step_len,
    )

    n_prefix = int(tok_out["tokenized_prefix_mask"].sum())
    prefix_decoded = tok._paligemma_tokenizer.decode(tok_out["tokenized_prefix"][:n_prefix].tolist())  # noqa: SLF001
    print(f"  Image           : {img_path}  shape={raw_uint8.shape}")
    print(f'  Instruction     : "{args.instruction}"')
    print(f"  State           : zeros({mc.action_dim},)  [placeholder]")
    print(f"  Prefix tokens   : {n_prefix} / {mc.max_prefix_len}")
    print(f"  Prefix text     : {prefix_decoded!r}")

    obs = _model.Observation(
        images={
            "base_0_rgb": jnp.array(img_float[None]),
            "base_1_rgb": jnp.array(dummy[None]),
            "wrist_0_rgb": jnp.array(dummy[None]),
        },
        image_masks={
            "base_0_rgb": jnp.array([True]),
            "base_1_rgb": jnp.array([True]),
            "wrist_0_rgb": jnp.array([True]),
        },
        state=jnp.array(robot_state[None]),
        tokenized_prefix=jnp.array(tok_out["tokenized_prefix"][None], dtype=jnp.int32),
        tokenized_prefix_mask=jnp.array(tok_out["tokenized_prefix_mask"][None]),
        prefix_ar_mask=jnp.array(tok_out["prefix_ar_mask"][None], dtype=jnp.int32),
        tokenized_action_tokens=jnp.zeros((1, mc.max_action_token_len), dtype=jnp.int32),
        tokenized_action_mask=jnp.zeros((1, mc.max_action_token_len), dtype=jnp.bool_),
        cot_step_tokens=jnp.zeros((1, mc.num_latent, mc.max_cot_step_len), dtype=jnp.int32),
        cot_step_masks=jnp.zeros((1, mc.num_latent, mc.max_cot_step_len), dtype=jnp.bool_),
        ref_answer_position=jnp.array([n_prefix], dtype=jnp.int32),
    )

    print(f"\n{'-'*w}")
    print("STEP 5 / 5  |  Inference")
    print(f"{'-'*w}")
    key = jax.random.PRNGKey(0)
    t0 = time.time()
    action_tokens, reasoning_tokens = model.sample_actions(
        key,
        obs,
        max_decoding_steps=args.max_decoding_steps,
        temperature=args.temperature,
        return_reasoning=True,
    )
    elapsed = time.time() - t0

    action_tokens_np = np.asarray(action_tokens[0], dtype=np.int32)
    reasoning_tokens_np = np.asarray(reasoning_tokens[0], dtype=np.int32)
    reasoning_sections = decode_reasoning_sections(tok._paligemma_tokenizer, reasoning_tokens_np)  # noqa: SLF001
    actions_np = decode_action_tokens(
        tok,
        action_tokens_np,
        action_horizon=mc.action_horizon,
        action_dim=mc.action_dim,
    )
    n_action_tokens = int(np.count_nonzero(action_tokens_np > 0))

    print(f"  Done in {elapsed:.2f}s")
    print(f"  reasoning_tokens shape : {reasoning_tokens_np.shape}")
    print(f"  action_tokens shape    : {action_tokens_np.shape}")
    print(f"  actions shape          : {actions_np.shape}")

    print(f"\n{'='*w}")
    print("  LATENT REASONING SECTIONS")
    print(f"{'='*w}")
    for i, section_text in enumerate(reasoning_sections):
        print(f"\n  [{_section_label(i, len(reasoning_sections))}]")
        for line in textwrap.wrap(section_text.strip() or "(empty)", width=w - 6):
            print(f"    {line}")

    print(f"\n{'='*w}")
    print("  ACTION TOKENS  (raw FAST output)")
    print(f"{'='*w}")
    print("  " + " ".join(str(int(t)) for t in action_tokens_np.tolist()))

    print(f"\n{'='*w}")
    print("  DECODED ACTIONS  (continuous)")
    print(f"{'='*w}")
    print(f"  horizon={mc.action_horizon}  dim={mc.action_dim}")
    print()
    dim_labels = [f"d{d:<3}" for d in range(mc.action_dim)]
    header = "  t  |  " + "  ".join(dim_labels)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for t_idx, action in enumerate(actions_np):
        row = "  ".join(f"{v:+.4f}" for v in action)
        print(f"  {t_idx:2d} |  {row}")

    print(f"\n{'='*w}")
    print("  ANNOTATED VISUALIZATION")
    print(f"{'='*w}")
    viz_bgr = build_annotated_image(
        raw_uint8,
        reasoning_sections,
        actions_np,
        args.instruction,
        elapsed,
        n_action_tokens,
        display_w=480,
        display_h=360,
    )
    out_path = Path(args.output)
    cv2.imwrite(str(out_path), viz_bgr)
    print(f"  Saved  : {out_path.resolve()}")
    print(f"  Size   : {viz_bgr.shape[1]}x{viz_bgr.shape[0]} px")

    print(f"\n{'='*w}")
    print("  SUMMARY")
    print(f"{'='*w}")
    print(f"  Config              : {args.config}")
    print(f"  Checkpoint          : {ckpt_path}")
    print(f'  Instruction         : "{args.instruction}"')
    print(f"  Latent sections     : {len(reasoning_sections)}")
    print(f"  Action tokens       : {n_action_tokens} / {args.max_decoding_steps}")
    print(f"  Decoded actions     : {actions_np.shape}")
    print(f"  Output image        : {out_path.resolve()}")
    print(f"{'='*w}")
    print("  TEST COMPLETE")
    print(f"{'='*w}")


if __name__ == "__main__":
    main()
