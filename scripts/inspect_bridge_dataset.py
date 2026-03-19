#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
from pathlib import Path

from openpi.training import bridge_dataset_inspector


DEFAULT_RLDS_ROOT = "/inspire/hdd/global_user/gongjingjing-25039/zhdai/openpi_dataset_bridge"
DEFAULT_DATASET_NAME = "bridge_orig"
DEFAULT_VERSION = "1.0.0"
DEFAULT_REASONING_PATH = (
    "/inspire/hdd/global_user/gongjingjing-25039/zhdai/hf_cache/hub/"
    "datasets--Embodied-CoT--embodied_features_bridge/snapshots/"
    "854ee59c7c76868d63fac37c33e0f031ed678014/embodied_features_bridge.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a local Bridge RLDS dataset and compare it with the published dataset statistics."
    )
    parser.add_argument("--rlds-root", default=DEFAULT_RLDS_ROOT, help="Root TFDS data dir passed to tfds.builder().")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME, help="TFDS dataset name to inspect.")
    parser.add_argument("--version", default=DEFAULT_VERSION, help="TFDS dataset version.")
    parser.add_argument(
        "--reasoning-dataset-path",
        default=DEFAULT_REASONING_PATH,
        help="Path to embodied_features_bridge.json.",
    )
    parser.add_argument(
        "--skip-tfds-scan",
        action="store_true",
        help="Skip the slower TFDS trajectory scan and only read dataset_info.json + reasoning JSON.",
    )
    parser.add_argument(
        "--tfds-limit",
        type=int,
        default=None,
        help="Optional cap on trajectories scanned from TFDS, useful for smoke tests.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    dataset_root = Path(args.rlds_root) / args.dataset_name / args.version
    reasoning_path = Path(args.reasoning_dataset_path)

    print("Bridge dataset inspection")
    print(f"  rlds_root: {args.rlds_root}")
    print(f"  dataset_name: {args.dataset_name}")
    print(f"  version: {args.version}")
    print(f"  dataset_root: {dataset_root}")
    print(f"  reasoning_dataset_path: {reasoning_path}")
    print()

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not reasoning_path.exists():
        raise FileNotFoundError(f"Reasoning dataset path not found: {reasoning_path}")

    dataset_info = bridge_dataset_inspector.load_dataset_info(dataset_root)
    dataset_summary = bridge_dataset_inspector.summarize_dataset_info(dataset_info)
    reasoning_dataset = bridge_dataset_inspector.load_reasoning_dataset(reasoning_path)
    reasoning_summary = bridge_dataset_inspector.summarize_reasoning_records(reasoning_dataset)
    del reasoning_dataset
    gc.collect()

    print("Dataset info summary")
    print(f"  total_trajectories_from_dataset_info: {dataset_summary['total_trajectories']}")
    for split_name, count in dataset_summary["split_counts"].items():
        shards = dataset_summary["split_shards"][split_name]
        print(f"  split[{split_name}]: trajectories={count}, shards={shards}")
    print()

    print("Reasoning summary")
    print(f"  files: {reasoning_summary['files']}")
    print(f"  episodes: {reasoning_summary['episodes']}")
    print(f"  episodes_with_reasoning: {reasoning_summary['episodes_with_reasoning']}")
    print(f"  episodes_without_reasoning: {reasoning_summary['episodes_without_reasoning']}")
    print(f"  frames_with_reasoning: {reasoning_summary['frames_with_reasoning']}")
    print(f"  environments_from_reasoning_paths: {reasoning_summary['environments']}")
    print(f"  skills_from_reasoning_paths: {reasoning_summary['skills']}")
    print(f"  normalized_environments: {reasoning_summary['normalized_environments']}")
    print(f"  normalized_skills: {reasoning_summary['normalized_skills']}")
    print(f"  files_with_unparsed_paths: {reasoning_summary['files_with_unparsed_paths']}")
    print("  top_environments:")
    for environment, count in reasoning_summary["environment_counts"].most_common(10):
        print(f"    {environment}: {count}")
    print("  top_normalized_environments:")
    for environment, count in reasoning_summary["normalized_environment_counts"].most_common(10):
        print(f"    {environment}: {count}")
    print("  top_skills:")
    for skill, count in reasoning_summary["skill_counts"].most_common(15):
        print(f"    {skill}: {count}")
    print("  top_normalized_skills:")
    for skill, count in reasoning_summary["normalized_skill_counts"].most_common(15):
        print(f"    {skill}: {count}")
    print()

    tfds_summary = None
    if args.skip_tfds_scan:
        print("TFDS scan skipped by request.")
        print()
    else:
        print("Scanning TFDS trajectories. This can take a while...")
        tfds_summary = bridge_dataset_inspector.scan_tfds_bridge(
            data_dir=args.rlds_root,
            dataset_name=args.dataset_name,
            version=args.version,
            split="train+val",
            limit=args.tfds_limit,
        )
        print("TFDS scan summary")
        print(f"  scanned_trajectories: {tfds_summary['scanned_trajectories']}")
        print(f"  unique_instructions: {tfds_summary['unique_instructions']}")
        print(f"  environments_from_tfds_paths: {tfds_summary['environments']}")
        print(f"  skills_from_tfds_paths: {tfds_summary['skills']}")
        print(f"  normalized_environments_from_tfds_paths: {tfds_summary['normalized_environments']}")
        print(f"  normalized_skills_from_tfds_paths: {tfds_summary['normalized_skills']}")
        print(f"  unknown_collection_type_trajectories: {tfds_summary['unknown_collection_type_trajectories']}")
        if tfds_summary["teleoperated_demonstrations"] is None:
            print("  teleoperated_demonstrations: unable to infer from file_path naming")
            print("  scripted_rollouts: unable to infer from file_path naming")
        else:
            print(f"  teleoperated_demonstrations: {tfds_summary['teleoperated_demonstrations']}")
            print(f"  scripted_rollouts: {tfds_summary['scripted_rollouts']}")
        print("  top_instructions:")
        for instruction, count in tfds_summary["top_instructions"][:10]:
            print(f"    {instruction}: {count}")
        print()

    observed = {
        "total_trajectories": dataset_summary["total_trajectories"],
        "teleoperated_demonstrations": None if tfds_summary is None else tfds_summary["teleoperated_demonstrations"],
        "scripted_rollouts": None if tfds_summary is None else tfds_summary["scripted_rollouts"],
        "environments": reasoning_summary["normalized_environments"],
        "skills": reasoning_summary["normalized_skills"],
    }
    comparison = bridge_dataset_inspector.compare_with_official_stats(observed)

    print("Comparison with official Bridge statistics")
    print("  note: environment/skill comparison uses normalized counts, not raw directory-level counts")
    for key, result in comparison.items():
        actual = "unknown" if result.actual is None else result.actual
        delta = "unknown" if result.delta is None else f"{result.delta:+d}"
        status = "MATCH" if result.matches else "MISMATCH"
        print(f"  {key}: actual={actual}, expected={result.expected}, delta={delta}, {status}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
