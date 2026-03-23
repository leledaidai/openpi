"""Chain-of-Thought utilities for OpenPI."""

import enum


class CotTag(enum.Enum):
    """Enumeration of Chain-of-Thought tags used in reasoning."""

    TASK = "TASK:"
    PLAN = "PLAN:"
    VISIBLE_OBJECTS = "VISIBLE OBJECTS:"
    SUBTASK_REASONING = "SUBTASK REASONING:"
    SUBTASK = "SUBTASK:"
    MOVE_REASONING = "MOVE REASONING:"
    MOVE = "MOVE:"
    GRIPPER_POSITION = "GRIPPER POSITION:"
    ACTION = "ACTION:"


def abbreviate_tag(tag: str) -> str:
    """Abbreviate a tag for compact logging.

    Args:
        tag: The tag string (e.g., "TASK:", "PLAN:")

    Returns:
        Abbreviated tag (e.g., "T:", "P:")
    """
    return tag[0] + tag[-2]


def get_cot_tags_list() -> list[str]:
    """Get list of all CoT tags in order.

    Returns:
        List of tag strings (e.g., ["TASK:", "PLAN:", ...])
    """
    return [
        CotTag.TASK.value,
        CotTag.PLAN.value,
        CotTag.VISIBLE_OBJECTS.value,
        CotTag.SUBTASK_REASONING.value,
        CotTag.SUBTASK.value,
        CotTag.MOVE_REASONING.value,
        CotTag.MOVE.value,
        CotTag.GRIPPER_POSITION.value,
        CotTag.ACTION.value,
    ]


def get_implicit_cot_tags_list() -> list[str]:
    """Get the fixed implicit-CoT step tags in order."""
    return get_cot_tags_list()[:-1]


def get_cot_database_keys() -> dict[str, str]:
    """Get mapping from CoT tags to database keys.

    Returns:
        Dictionary mapping tag strings to JSON field names
    """
    return {
        CotTag.TASK.value: "task",
        CotTag.PLAN.value: "plan",
        CotTag.VISIBLE_OBJECTS.value: "bboxes",
        CotTag.SUBTASK_REASONING.value: "subtask_reason",
        CotTag.SUBTASK.value: "subtask",
        CotTag.MOVE_REASONING.value: "move_reason",
        CotTag.MOVE.value: "move",
        CotTag.GRIPPER_POSITION.value: "gripper",
        CotTag.ACTION.value: "action",
    }


def format_reasoning_dict_to_string(reasoning_dict: dict[str, str]) -> str:
    """Convert a reasoning dictionary to a formatted string.

    Args:
        reasoning_dict: Dictionary with keys matching get_cot_database_keys() values

    Returns:
        Formatted string like "TASK: ... PLAN: ... MOVE REASONING: ..."
    """
    tags = get_implicit_cot_tags_list()
    database_keys = get_cot_database_keys()
    reasoning_parts = []

    for tag in tags:
        db_key = database_keys[tag]
        if db_key in reasoning_dict and reasoning_dict[db_key]:
            reasoning_parts.append(f"{tag} {reasoning_dict[db_key]}")

    return " ".join(reasoning_parts)


def parse_reasoning_string(reasoning_str: str) -> dict[str, str]:
    """Parse a formatted reasoning string back into a dictionary.

    Args:
        reasoning_str: Formatted string like "TASK: ... PLAN: ..."

    Returns:
        Dictionary with tag keys and their content
    """
    result = {}
    tags = get_cot_tags_list()

    # Split by tags
    for i, tag in enumerate(tags):
        if tag not in reasoning_str:
            continue

        # Find the start of this tag
        start_idx = reasoning_str.find(tag)
        if start_idx == -1:
            continue

        # Find the end (start of next tag or end of string)
        end_idx = len(reasoning_str)
        for next_tag in tags[i + 1 :]:
            next_idx = reasoning_str.find(next_tag, start_idx + len(tag))
            if next_idx != -1:
                end_idx = min(end_idx, next_idx)

        # Extract content
        content = reasoning_str[start_idx + len(tag) : end_idx].strip()
        result[tag] = content

    return result


def extract_implicit_cot_steps(reasoning_str: str) -> list[str]:
    """Split a reasoning string into the fixed implicit-CoT step slots.

    Missing steps are kept as empty strings. The returned step text keeps the
    original tag prefix, e.g. ``"TASK: place the mug"``.
    """
    if not reasoning_str or not reasoning_str.strip():
        return [""] * len(get_implicit_cot_tags_list())

    parsed = parse_reasoning_string(reasoning_str)
    steps = []
    for tag in get_implicit_cot_tags_list():
        content = parsed.get(tag, "").strip()
        steps.append(f"{tag} {content}".strip() if content else "")
    return steps


def format_visible_reasoning_without_action(reasoning_str: str) -> str:
    """Keep only the visible reasoning steps and drop the ACTION field entirely."""
    return " ".join(step for step in extract_implicit_cot_steps(reasoning_str) if step)
