#!/usr/bin/env python3
"""
PiCOT inference test script.

Architecture: dual-stream transformer (PaliGemma LLM + action expert with adaRMS).
Inference pipeline:
  1. Embed prefix (images + text prompt, bidirectional)
  2. Autoregressive CoT generation  → "TASK: ..." text
  3. Autoregressive FAST token generation → "Action: ..." tokens
  4. 10-step flow matching denoising → continuous actions

Key difference from pi0_fast: actions come from flow matching (continuous),
not from FAST decoding (discrete). FAST tokens are generated but only used
as conditioning context for the action expert.

Usage:
  python test_pi_cot.py \\
      --checkpoint <path/to/params> \\
      --image test_obs.png \\
      --instruction "place the watermelon on the towel"

  # Or using exp-name + step:
  python test_pi_cot.py --exp my_exp --step 29999
"""

import argparse
import dataclasses
import enum
import sys
import textwrap
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent / "src"))


# ─────────────────────────────────────────────────────────────────────────────
# CoT parsing utilities  (same logic as Example.ipynb)
# ─────────────────────────────────────────────────────────────────────────────

class CotTag(enum.Enum):
    TASK               = "TASK:"
    PLAN               = "PLAN:"
    VISIBLE_OBJECTS    = "VISIBLE OBJECTS:"
    SUBTASK_REASONING  = "SUBTASK REASONING:"
    SUBTASK            = "SUBTASK:"
    MOVE_REASONING     = "MOVE REASONING:"
    MOVE               = "MOVE:"
    GRIPPER_POSITION   = "GRIPPER POSITION:"
    ACTION             = "ACTION:"


def get_cot_tags_list():
    return [t.value for t in CotTag]


def split_reasoning(text: str, tags: list) -> dict:
    parts = {None: text}
    for tag in tags:
        new_parts = {}
        for k, v in parts.items():
            if tag in v:
                s = v.split(tag, 1)
                new_parts[k] = s[0]
                new_parts[tag] = s[1]
            else:
                new_parts[k] = v
        parts = new_parts
    return parts


def parse_metadata(reasoning: dict) -> dict:
    metadata = {"gripper": [[0, 0]], "bboxes": {}}

    gripper_key = f" {CotTag.GRIPPER_POSITION.value}"
    if gripper_key in reasoning:
        raw = reasoning[gripper_key]
        raw = raw.split("[")[-1].split("]")[0]
        try:
            nums = [int(x.strip()) for x in raw.split(",") if x.strip()]
            metadata["gripper"] = [
                (nums[2 * i], nums[2 * i + 1]) for i in range(len(nums) // 2)
            ]
        except (ValueError, IndexError):
            pass

    objects_key = f" {CotTag.VISIBLE_OBJECTS.value}"
    if objects_key in reasoning:
        for chunk in reasoning[objects_key].split("]"):
            if "[" not in chunk:
                continue
            name_part, coord_part = chunk.split("[", 1)
            name = name_part.strip().lstrip(",").strip()
            if not name:
                continue
            try:
                coords = [int(x.strip()) for x in coord_part.split(",") if x.strip()]
                if len(coords) >= 4:
                    metadata["bboxes"][name] = coords[:4]
            except ValueError:
                pass

    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# Drawing utilities
# ─────────────────────────────────────────────────────────────────────────────

def name_to_color(name: str) -> tuple:
    h = hash(name)
    return ((h >> 0) & 0xFF, (h >> 8) & 0xFF, (h >> 16) & 0xFF)


def scale_pos(pos: tuple, img_w: int, img_h: int) -> tuple:
    return (int(pos[0] * img_w) // 256, int(pos[1] * img_h) // 256)


def draw_gripper(img_bgr: np.ndarray, gripper_positions: list) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    for i, pos in enumerate(reversed(gripper_positions)):
        px, py = scale_pos(pos, w, h)
        alpha = 255 - int(200 * i / max(len(gripper_positions), 1))
        cv2.circle(img_bgr, (px, py), 7, (0, 0, 0), -1)
        cv2.circle(img_bgr, (px, py), 6, (alpha, alpha, 255), -1)
    return img_bgr


def draw_bboxes(img_bgr: np.ndarray, bboxes: dict) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    for name, (x1, y1, x2, y2) in bboxes.items():
        p1 = scale_pos((x1, y1), w, h)
        p2 = scale_pos((x2, y2), w, h)
        color = name_to_color(name)
        cv2.rectangle(img_bgr, p1, p2, color, 2)
        cv2.putText(
            img_bgr, name.strip(),
            (p1[0], max(p1[1] - 4, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return img_bgr


def make_text_panel(sections: list, width: int, height: int, font_size: int = 13) -> np.ndarray:
    base = Image.fromarray(np.full((height, width, 3), 255, dtype=np.uint8))
    draw = ImageDraw.Draw(base)
    try:
        font_header = ImageFont.load_default(size=font_size + 1)
        font_body   = ImageFont.load_default(size=font_size)
    except TypeError:
        font_header = ImageFont.load_default()
        font_body   = font_header

    x, y     = 12, 10
    line_h   = font_size + 4
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

    arr = np.array(base, dtype=np.uint8)
    return arr[:, :, ::-1]  # RGB -> BGR


def build_annotated_image(
    raw_uint8: np.ndarray,
    cot_text: str,
    actions_np: np.ndarray,
    instruction: str,
    elapsed_flow: float,
    num_steps: int,
    display_w: int = 480,
    display_h: int = 360,
) -> np.ndarray:
    tags      = [f" {t}" for t in get_cot_tags_list()]
    reasoning = split_reasoning(cot_text, tags)
    metadata  = parse_metadata(reasoning)

    bboxes = {
        k.lstrip(",").strip(): v
        for k, v in metadata["bboxes"].items()
        if k.lstrip(",").strip()
    }

    # Left panel: annotated observation
    obs_bgr = cv2.resize(raw_uint8[:, :, ::-1].astype(np.uint8), (display_w, display_h))
    obs_bgr = draw_gripper(obs_bgr, metadata["gripper"])
    obs_bgr = draw_bboxes(obs_bgr, bboxes)

    cv2.rectangle(obs_bgr, (0, 0), (display_w, 22), (0, 0, 0), -1)
    cv2.putText(
        obs_bgr, f"Task: {instruction}", (6, 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA,
    )
    cv2.rectangle(obs_bgr, (0, display_h - 22), (display_w, display_h), (0, 0, 0), -1)
    cv2.putText(
        obs_bgr,
        f"FlowMatch: {elapsed_flow:.2f}s  steps={num_steps}  shape={actions_np.shape}",
        (6, display_h - 7),
        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA,
    )

    # Right panel: CoT sections
    DISPLAY_ORDER = [
        (f" {CotTag.TASK.value}",              "TASK"),
        (f" {CotTag.PLAN.value}",              "PLAN"),
        (f" {CotTag.VISIBLE_OBJECTS.value}",   "VISIBLE OBJECTS"),
        (f" {CotTag.SUBTASK_REASONING.value}", "SUBTASK REASONING"),
        (f" {CotTag.SUBTASK.value}",           "SUBTASK"),
        (f" {CotTag.MOVE_REASONING.value}",    "MOVE REASONING"),
        (f" {CotTag.MOVE.value}",              "MOVE"),
        (f" {CotTag.GRIPPER_POSITION.value}",  "GRIPPER POSITION"),
        (f" {CotTag.ACTION.value}",            "ACTION (CoT)"),
    ]
    sections = [
        (label + ":", reasoning[key].strip())
        for key, label in DISPLAY_ORDER
        if key in reasoning
    ]
    if not sections:
        sections = [("Raw CoT output:", cot_text.strip() or "(none)")]

    # Append first action row as a summary
    a0 = "  ".join(f"{v:+.3f}" for v in actions_np[0])
    sections.append(("Action t=0 (flow):", a0))

    text_panel = make_text_panel(sections, display_w, display_h)
    return np.concatenate([obs_bgr, text_panel], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PiCOT inference test: CoT reasoning + flow-matching actions."
    )
    parser.add_argument("--checkpoint", type=str, default=None,
        help="Absolute path to checkpoint params dir.")
    parser.add_argument("--exp",  type=str, default=None,
        help="Experiment name (used to build checkpoint path).")
    parser.add_argument("--step", type=int, default=None,
        help="Checkpoint step (used with --exp).")
    parser.add_argument("--image", type=str, default="test_obs.png",
        help="Path to test observation image.")
    parser.add_argument("--instruction", type=str,
        default="place the watermelon on the towel",
        help="Language instruction for the robot.")
    parser.add_argument("--num-steps", type=int, default=10,
        help="Flow matching denoising steps.")
    parser.add_argument("--cot-temperature", type=float, default=0.0,
        help="CoT sampling temperature. 0=greedy.")
    parser.add_argument("--output", type=str, default="pi_cot_result.png",
        help="Path to save the annotated visualization.")
    parser.add_argument("--config", type=str,
        default="pi_cot_bridge_rlds_finetune_cot_test",
        help="TrainConfig name.")
    parser.add_argument("--max-cot-tokens", type=int, default=None,
        help="Override max_cot_tokens at inference time (smaller = faster JIT + faster run). "
             "Must be <= config value. E.g. --max-cot-tokens 128")
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp
    from flax import nnx
    from jax.sharding import Mesh, NamedSharding, PartitionSpec
    from orbax.checkpoint import ArrayRestoreArgs
    import orbax.checkpoint as ocp
    from openpi.models import model as _model
    from openpi.models import tokenizer as _tokenizer
    import openpi.models.pi_cot as pi_cot_model
    from openpi.training import config as train_config

    W = 80

    print("=" * W)
    print("  PICOT INFERENCE TEST")
    print("  CoT reasoning (AR) + Flow-matching actions (action expert blocked from FAST)")
    print("=" * W)
    print(f"\nJAX devices : {jax.devices()}")

    # ── STEP 1: Config ────────────────────────────────────────────────────────
    print(f"\n{'─'*W}")
    print("STEP 1 / 5  |  Configuration")
    print(f"{'─'*W}")

    cfg = train_config.get_config(args.config)
    mc  = cfg.model

    assert isinstance(mc, pi_cot_model.PiCOTConfig), (
        f"Expected PiCOTConfig, got {type(mc)}"
    )

    print(f"  config name              : {args.config}")
    print(f"  action_dim               : {mc.action_dim}")
    print(f"  action_horizon           : {mc.action_horizon}")
    print(f"  max_token_len (prefix)   : {mc.max_token_len}")
    print(f"  max_cot_tokens           : {mc.max_cot_tokens}")
    print(f"  max_fast_tokens          : {mc.max_fast_tokens}")
    print(f"  flow_matching_loss_weight: {mc.flow_matching_loss_weight}")
    print(f"  paligemma_variant        : {mc.paligemma_variant}")
    print(f"  action_expert_variant    : {mc.action_expert_variant}")
    print(f"  dtype                    : {mc.dtype}")
    print(f"  batch_size               : {cfg.batch_size}")
    print(f"  num_train_steps          : {cfg.num_train_steps}")
    print(f"  --- inference ---")
    print(f"  num_steps (flow)         : {args.num_steps}")
    print(f"  cot_temperature          : {args.cot_temperature}")

    # Allow overriding max_cot_tokens at inference time without retraining.
    # The model's for-loop is unrolled to exactly max_cot_tokens steps at JIT time,
    # so a smaller value directly reduces both compile time and per-step runtime.
    if args.max_cot_tokens is not None:
        assert args.max_cot_tokens <= mc.max_cot_tokens, (
            f"--max-cot-tokens {args.max_cot_tokens} > config {mc.max_cot_tokens}"
        )
        mc = dataclasses.replace(mc, max_cot_tokens=args.max_cot_tokens)
        print(f"  max_cot_tokens (override): {mc.max_cot_tokens}  "
              f"(was {cfg.model.max_cot_tokens})")

    # ── STEP 2: Build model ───────────────────────────────────────────────────
    print(f"\n{'─'*W}")
    print("STEP 2 / 5  |  Initialize PiCOT model")
    print(f"{'─'*W}")

    t0    = time.time()
    rng   = nnx.Rngs(0)
    model = pi_cot_model.PiCOT(mc, rngs=rng)
    print(f"  Initialized in {time.time()-t0:.1f}s")

    # ── STEP 3: Load checkpoint ───────────────────────────────────────────────
    print(f"\n{'─'*W}")
    print("STEP 3 / 5  |  Load checkpoint")
    print(f"{'─'*W}")

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint).resolve()
    elif args.exp is not None and args.step is not None:
        ckpt_path = Path(
            f"checkpoints/{args.config}/{args.exp}/{args.step}/params"
        ).resolve()
    else:
        print(
            "  ERROR: supply --checkpoint <path>  OR  both --exp <name> --step <n>\n"
            f"  Expected: checkpoints/{args.config}/<exp>/<step>/params"
        )
        sys.exit(1)

    print(f"  Path: {ckpt_path}")
    if not ckpt_path.exists():
        print(f"  ERROR: not found: {ckpt_path}")
        sys.exit(1)

    devices  = jax.devices()
    mesh     = Mesh(devices, axis_names=("replica",))
    sharding = NamedSharding(mesh, PartitionSpec())
    handler  = ocp.PyTreeCheckpointHandler()
    state    = nnx.state(model)
    r_args   = jax.tree.map(lambda _: ArrayRestoreArgs(sharding=sharding), state)

    t0 = time.time()
    restored = handler.restore(
        ckpt_path,
        item={"params": state},
        restore_args={"params": r_args},
    )
    nnx.update(model, restored["params"])
    model.eval()
    print(f"  Loaded in {time.time()-t0:.1f}s  ({len(devices)} device(s))")

    # ── STEP 4: Prepare observation ───────────────────────────────────────────
    print(f"\n{'─'*W}")
    print("STEP 4 / 5  |  Prepare observation")
    print(f"{'─'*W}")

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"  WARNING: '{img_path}' not found — using random noise.")
        raw_uint8 = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    else:
        raw_uint8 = np.array(
            Image.open(img_path).convert("RGB").resize((224, 224)), dtype=np.uint8
        )

    img_float = raw_uint8.astype(np.float32) / 127.5 - 1.0
    dummy     = np.zeros((224, 224, 3), dtype=np.float32)

    # PiCOT uses tokenize_picot(): separate prefix / CoT / FAST arrays
    tok = _tokenizer.FASTTokenizer(
        max_len=mc.max_token_len,
        fast_tokenizer_path=mc.fast_tokenizer_path,
    )
    robot_state = np.zeros(mc.action_dim, dtype=np.float32)

    tok_out = tok.tokenize_picot(
        args.instruction,
        robot_state,
        actions=None,
        cot_reasoning=None,
        max_cot_tokens=mc.max_cot_tokens,
        max_fast_tokens=mc.max_fast_tokens,
    )

    n_prefix = int(tok_out["tokenized_prompt_mask"].sum())
    prefix_decoded = tok._paligemma_tokenizer.decode(
        tok_out["tokenized_prompt"][:n_prefix].tolist()
    )
    print(f"  Image           : {img_path}  shape={raw_uint8.shape}")
    print(f"  Instruction     : \"{args.instruction}\"")
    print(f"  State           : zeros({mc.action_dim},)  [placeholder]")
    print(f"  Prefix tokens   : {n_prefix} / {mc.max_token_len}")
    print(f"  Prefix text     : {prefix_decoded!r}")

    obs = _model.Observation(
        images={
            "base_0_rgb":        jnp.array(img_float[None]),
            "left_wrist_0_rgb":  jnp.array(dummy[None]),
            "right_wrist_0_rgb": jnp.array(dummy[None]),
        },
        image_masks={
            "base_0_rgb":        jnp.array([True]),
            "left_wrist_0_rgb":  jnp.array([True]),
            "right_wrist_0_rgb": jnp.array([True]),
        },
        state=jnp.array(robot_state[None]),
        tokenized_prompt=jnp.array(tok_out["tokenized_prompt"][None],       dtype=jnp.int32),
        tokenized_prompt_mask=jnp.array(tok_out["tokenized_prompt_mask"][None]),
        # CoT and FAST arrays are zero-padded; model generates them autoregressively
        tokenized_cot_reasoning=jnp.zeros((1, mc.max_cot_tokens),       dtype=jnp.int32),
        tokenized_cot_reasoning_mask=jnp.zeros((1, mc.max_cot_tokens),  dtype=jnp.bool_),
        # tokenized_fast_actions=jnp.zeros((1, mc.max_fast_tokens),       dtype=jnp.int32),
        # tokenized_fast_actions_mask=jnp.zeros((1, mc.max_fast_tokens),  dtype=jnp.bool_),
    )

    # ── STEP 5: Inference ─────────────────────────────────────────────────────
    print(f"\n{'─'*W}")
    print("STEP 5 / 5  |  Inference")
    print(f"{'─'*W}")

    key = jax.random.PRNGKey(0)
    obs_pre = _model.preprocess_observation(None, obs, train=False)

    # Phase A: embed prefix
    print("  [A] Embedding prefix ...")
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(obs_pre)
    print(f"      prefix_tokens shape : {prefix_tokens.shape}")

    # Phase B: autoregressive CoT generation
    print(f"  [B] Generating CoT (max={mc.max_cot_tokens} tokens, temp={args.cot_temperature}) ...")
    key, cot_rng = jax.random.split(key)
    t_cot = time.time()
    cot_emb, cot_mask, cot_ar_mask, cot_ids = model.generate_cot_tokens(
        cot_rng, prefix_tokens, prefix_mask, prefix_ar_mask,
        temperature=args.cot_temperature,
    )
    cot_elapsed = time.time() - t_cot
    cot_ids_np  = np.array(cot_ids[0], dtype=np.int32).tolist()
    cot_text    = tok._paligemma_tokenizer.decode([t for t in cot_ids_np if t > 0])
    n_cot_valid = int(cot_mask[0].sum())
    print(f"      Done in {cot_elapsed:.2f}s  ({n_cot_valid} valid tokens)")

    # Phase C: flow matching denoising (conditioned on prefix + CoT only, NOT FAST).
    # sample_actions internally re-generates CoT with its own rng split, then runs
    # the denoising loop. Action expert is blocked from attending to FAST tokens
    # (consistent with training attention mask), so no FAST generation is needed here.
    print(f"  [C] Flow matching denoising ({args.num_steps} steps) ...")
    key, flow_rng = jax.random.split(key)
    t_flow = time.time()
    actions = model.sample_actions(
        flow_rng, obs,
        num_steps=args.num_steps,
        cot_temperature=args.cot_temperature,
    )
    elapsed_flow = time.time() - t_flow
    actions_np   = np.array(actions[0])  # [action_horizon, action_dim]
    print(f"      Done in {elapsed_flow:.2f}s  shape={actions_np.shape}")

    # ── Print CoT text ─────────────────────────────────────────────────────────
    print(f"\n{'='*W}")
    print("  CHAIN-OF-THOUGHT REASONING  (raw, full text)")
    print(f"{'='*W}")
    if cot_text:
        for line in textwrap.wrap(cot_text, width=W):
            print(f"  {line}")
    else:
        print("  (empty)")

    # ── Print structured CoT sections ─────────────────────────────────────────
    DISPLAY_ORDER = [
        (f" {CotTag.TASK.value}",              "TASK"),
        (f" {CotTag.PLAN.value}",              "PLAN"),
        (f" {CotTag.VISIBLE_OBJECTS.value}",   "VISIBLE OBJECTS"),
        (f" {CotTag.SUBTASK_REASONING.value}", "SUBTASK REASONING"),
        (f" {CotTag.SUBTASK.value}",           "SUBTASK"),
        (f" {CotTag.MOVE_REASONING.value}",    "MOVE REASONING"),
        (f" {CotTag.MOVE.value}",              "MOVE"),
        (f" {CotTag.GRIPPER_POSITION.value}",  "GRIPPER POSITION"),
        (f" {CotTag.ACTION.value}",            "ACTION (CoT tag)"),
    ]

    if cot_text:
        tags      = [f" {t}" for t in get_cot_tags_list()]
        reasoning = split_reasoning(cot_text, tags)
        found_any = False
        for tag_key, label in DISPLAY_ORDER:
            if tag_key in reasoning:
                found_any = True
                body = reasoning[tag_key].strip()
                print(f"\n  [{label}]")
                for line in textwrap.wrap(body, width=W - 6) or ["(empty)"]:
                    print(f"    {line}")
        if not found_any:
            print("  (no structured CoT tags found — raw CoT text above)")
    else:
        print("  (model generated no CoT text)")

    # ── Print flow-matching actions ────────────────────────────────────────────
    print(f"\n{'='*W}")
    print("  FLOW-MATCHING ACTIONS  (continuous, from denoising)")
    print(f"{'='*W}")
    print(f"  horizon={mc.action_horizon}  dim={mc.action_dim}")
    print()

    dim_labels = [f"d{d:<3}" for d in range(mc.action_dim)]
    header = "  t  |  " + "  ".join(dim_labels)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for t_idx, a in enumerate(actions_np):
        row = "  ".join(f"{v:+.4f}" for v in a)
        print(f"  {t_idx:2d} |  {row}")

    # ── Build and save annotated image ────────────────────────────────────────
    print(f"\n{'='*W}")
    print("  ANNOTATED VISUALIZATION")
    print(f"{'='*W}")

    viz_bgr = build_annotated_image(
        raw_uint8, cot_text, actions_np, args.instruction,
        elapsed_flow, args.num_steps,
        display_w=480, display_h=360,
    )

    out_path = Path(args.output)
    cv2.imwrite(str(out_path), viz_bgr)
    print(f"  Saved  : {out_path.resolve()}")
    print(f"  Size   : {viz_bgr.shape[1]}x{viz_bgr.shape[0]} px")
    print(f"  Layout : left=annotated observation  right=CoT sections + action t=0")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*W}")
    print("  SUMMARY")
    print(f"{'='*W}")
    print(f"  Config              : {args.config}")
    print(f"  Checkpoint          : {ckpt_path}")
    print(f"  Instruction         : \"{args.instruction}\"")
    print(f"  CoT generation      : {cot_elapsed:.2f}s  (~{n_cot_valid} tokens)")
    print(f"  Flow matching       : {elapsed_flow:.2f}s  ({args.num_steps} steps)")
    print(f"  Actions shape       : {actions_np.shape}")
    print(f"  Output image        : {out_path.resolve()}")
    print(f"{'='*W}")
    print("  TEST COMPLETE")
    print(f"{'='*W}")


if __name__ == "__main__":
    main()
