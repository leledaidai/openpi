from __future__ import annotations

from collections import Counter
import dataclasses
import json
from pathlib import Path
import re
from typing import Any


OFFICIAL_BRIDGE_STATS = {
    "total_trajectories": 60_096,
    "teleoperated_demonstrations": 50_365,
    "scripted_rollouts": 9_731,
    "environments": 24,
    "skills": 13,
}


@dataclasses.dataclass(frozen=True)
class ParsedBridgePath:
    dataset_variant: str
    site: str | None
    environment: str
    skill: str
    collection_id: str | None
    split: str


@dataclasses.dataclass(frozen=True)
class StatComparison:
    expected: int
    actual: int | None
    matches: bool
    delta: int | None


def parse_bridge_source_path(path: str) -> ParsedBridgePath | None:
    parts = Path(path).parts
    for i, part in enumerate(parts):
        if not part.startswith("bridge_data_v"):
            continue
        if i + 4 >= len(parts):
            return None
        maybe_split = parts[i + 4]
        if maybe_split not in {"train", "val", "test"}:
            return None
        maybe_collection_id = parts[i + 3]
        if maybe_collection_id.isdigit():
            return ParsedBridgePath(
                dataset_variant=part,
                site=None,
                environment=parts[i + 1],
                skill=parts[i + 2],
                collection_id=maybe_collection_id,
                split=maybe_split,
            )
        return ParsedBridgePath(
            dataset_variant=part,
            site=parts[i + 1],
            environment=parts[i + 2],
            skill=parts[i + 3],
            collection_id=None,
            split=maybe_split,
        )
    return None


def load_dataset_info(dataset_root: str | Path) -> dict[str, Any]:
    dataset_info_path = Path(dataset_root) / "dataset_info.json"
    with dataset_info_path.open() as f:
        return json.load(f)


def summarize_dataset_info(dataset_info: dict[str, Any]) -> dict[str, Any]:
    split_counts = {}
    split_shards = {}
    for split in dataset_info["splits"]:
        shard_lengths = [int(x) for x in split["shardLengths"]]
        split_counts[split["name"]] = sum(shard_lengths)
        split_shards[split["name"]] = len(shard_lengths)

    return {
        "split_counts": split_counts,
        "split_shards": split_shards,
        "total_trajectories": sum(split_counts.values()),
    }


def load_reasoning_dataset(reasoning_dataset_path: str | Path) -> dict[str, Any]:
    with Path(reasoning_dataset_path).open() as f:
        return json.load(f)


def summarize_reasoning_records(reasoning_dataset: dict[str, Any]) -> dict[str, Any]:
    environment_counter: Counter[str] = Counter()
    skill_counter: Counter[str] = Counter()
    normalized_environment_counter: Counter[str] = Counter()
    normalized_skill_counter: Counter[str] = Counter()
    split_counter: Counter[str] = Counter()
    files_with_unparsed_paths = 0
    episodes = 0
    episodes_with_reasoning = 0
    frames_with_reasoning = 0

    for file_path, episode_map in reasoning_dataset.items():
        parsed = parse_bridge_source_path(file_path)
        if parsed is None:
            files_with_unparsed_paths += 1
        else:
            environment_counter[parsed.environment] += 1
            skill_counter[parsed.skill] += 1
            normalized_environment_counter[normalize_environment_name(parsed.environment)] += 1
            normalized_skill_counter[normalize_skill_name(parsed.skill)] += 1
            split_counter[parsed.split] += 1

        for episode_data in episode_map.values():
            episodes += 1
            reasoning = episode_data.get("reasoning")
            if reasoning:
                episodes_with_reasoning += 1
                frames_with_reasoning += len(reasoning)

    return {
        "files": len(reasoning_dataset),
        "episodes": episodes,
        "episodes_with_reasoning": episodes_with_reasoning,
        "episodes_without_reasoning": episodes - episodes_with_reasoning,
        "frames_with_reasoning": frames_with_reasoning,
        "environments": len(environment_counter),
        "skills": len(skill_counter),
        "environment_counts": environment_counter,
        "skill_counts": skill_counter,
        "normalized_environments": len(normalized_environment_counter),
        "normalized_skills": len(normalized_skill_counter),
        "normalized_environment_counts": normalized_environment_counter,
        "normalized_skill_counts": normalized_skill_counter,
        "split_counts": split_counter,
        "files_with_unparsed_paths": files_with_unparsed_paths,
    }


def compare_with_official_stats(observed: dict[str, int | None]) -> dict[str, StatComparison]:
    comparison = {}
    for key, expected in OFFICIAL_BRIDGE_STATS.items():
        actual = observed.get(key)
        comparison[key] = StatComparison(
            expected=expected,
            actual=actual,
            matches=actual == expected,
            delta=None if actual is None else actual - expected,
        )
    return comparison


def scan_tfds_bridge(
    data_dir: str,
    dataset_name: str = "bridge_orig",
    version: str = "1.0.0",
    split: str = "train+val",
    limit: int | None = None,
) -> dict[str, Any]:
    import tensorflow as tf
    import tensorflow_datasets as tfds

    tf.config.set_visible_devices([], "GPU")

    builder = tfds.builder(dataset_name, data_dir=data_dir, version=version)
    dataset = builder.as_dataset(split=split)

    trajectory_count = 0
    instruction_counter: Counter[str] = Counter()
    environment_counter: Counter[str] = Counter()
    skill_counter: Counter[str] = Counter()
    normalized_environment_counter: Counter[str] = Counter()
    normalized_skill_counter: Counter[str] = Counter()
    split_counter: Counter[str] = Counter()
    teleop_count = 0
    scripted_count = 0
    unknown_collection_type_count = 0

    iterator = tfds.as_numpy(dataset)
    for episode in iterator:
        trajectory_count += 1
        steps = list(episode["steps"])
        if steps:
            instruction = _decode_scalar(steps[0].get("language_instruction", b""))
            if instruction:
                instruction_counter[instruction] += 1

        episode_metadata = episode.get("episode_metadata", {})
        file_path = _decode_scalar(episode_metadata.get("file_path", b""))
        parsed = parse_bridge_source_path(file_path) if file_path else None
        if parsed is not None:
            environment_counter[parsed.environment] += 1
            skill_counter[parsed.skill] += 1
            normalized_environment_counter[normalize_environment_name(parsed.environment)] += 1
            normalized_skill_counter[normalize_skill_name(parsed.skill)] += 1
            split_counter[parsed.split] += 1
            collection_kind = infer_collection_type_from_path(file_path)
            if collection_kind == "teleop":
                teleop_count += 1
            elif collection_kind == "scripted":
                scripted_count += 1
            else:
                unknown_collection_type_count += 1
        else:
            unknown_collection_type_count += 1

        if limit is not None and trajectory_count >= limit:
            break

    teleop_value: int | None = teleop_count if (teleop_count + scripted_count) == trajectory_count else None
    scripted_value: int | None = scripted_count if (teleop_count + scripted_count) == trajectory_count else None

    return {
        "scanned_trajectories": trajectory_count,
        "unique_instructions": len(instruction_counter),
        "top_instructions": instruction_counter.most_common(20),
        "environments": len(environment_counter),
        "skills": len(skill_counter),
        "environment_counts": environment_counter,
        "skill_counts": skill_counter,
        "normalized_environments": len(normalized_environment_counter),
        "normalized_skills": len(normalized_skill_counter),
        "normalized_environment_counts": normalized_environment_counter,
        "normalized_skill_counts": normalized_skill_counter,
        "split_counts": split_counter,
        "teleoperated_demonstrations": teleop_value,
        "scripted_rollouts": scripted_value,
        "unknown_collection_type_trajectories": unknown_collection_type_count,
    }


def infer_collection_type_from_path(path: str) -> str | None:
    lowered = path.lower()
    if any(token in lowered for token in ("scripted", "rollout", "policy")):
        return "scripted"
    if any(token in lowered for token in ("teleop", "demo", "demonstration")):
        return "teleop"
    return None


def normalize_environment_name(environment: str) -> str:
    normalized = environment.lower()
    normalized = re.sub(r"^(datacol\d+_|deepthought_|rss_|minsky_)", "", normalized)
    normalized = re.sub(r"_room\d+$", "", normalized)
    normalized = re.sub(r"_bww$", "", normalized)
    normalized = normalized.replace("__", "_")
    return normalized


def normalize_skill_name(skill: str) -> str:
    normalized = skill.lower().strip()

    explicit_mappings = {
        "many_skills": "composite",
        "test": "composite",
        "stack_blocks": "stack_blocks",
        "fold_cloth": "fold_cloth",
        "fold_cloth_pnp": "fold_cloth",
        "fold_cloth_in_half": "fold_cloth",
        "pnp_push_sweep": "sweep_push",
        "pnp_sweep": "sweep_push",
        "sweep_granular": "sweep_push",
        "drawer_pnp": "drawer",
        "zip_zipper_bag": "bag_open_close",
        "unzip_zipper_bag": "bag_open_close",
        "right_pepper_shaker": "reorient",
        "lift_bowl": "pick_place",
        "move_drying_rack_out_of_sink": "pick_place",
        "move_light_switch_to_the_right": "articulated",
        "move_faucet_front_to_left": "articulated",
    }
    if normalized in explicit_mappings:
        return explicit_mappings[normalized]

    if normalized.startswith(("open_", "close_")):
        return "open_close"
    if "lever" in normalized or "knob" in normalized:
        return "articulated"
    if normalized.startswith(("turn_", "flip_")) and any(
        token in normalized for token in ("lever", "knob", "switch", "faucet", "handle")
    ):
        return "articulated"
    if normalized.startswith("fold_"):
        return "fold_cloth"
    if "laundry_machine" in normalized or "clothes" in normalized:
        return "laundry"
    if "wipe" in normalized or "sponge" in normalized:
        return "wipe"
    if any(token in normalized for token in ("upright", "topple")) or normalized.startswith("flip_"):
        return "reorient"
    if normalized.startswith(("pour_",)):
        return "pour"
    if "drawer" in normalized or "box" in normalized:
        return "drawer"
    if any(token in normalized for token in ("sweep", "push")):
        return "sweep_push"
    if normalized.startswith("stack_"):
        return "stack_blocks"
    if normalized.startswith(("put_", "pick_", "take_")):
        return "pick_place"
    return normalized


def _decode_scalar(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)
