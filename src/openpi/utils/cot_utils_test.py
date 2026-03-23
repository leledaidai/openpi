from openpi.utils import cot_utils


def test_extract_implicit_cot_steps_excludes_action_and_keeps_slots():
    reasoning = (
        "TASK: place the mug "
        "PLAN: move to the mug "
        "SUBTASK: pick up the mug "
        "MOVE: approach forward "
        "ACTION: leaked action tokens should not appear"
    )

    steps = cot_utils.extract_implicit_cot_steps(reasoning)

    assert len(steps) == 8
    assert steps[0] == "TASK: place the mug"
    assert steps[1] == "PLAN: move to the mug"
    assert steps[4] == "SUBTASK: pick up the mug"
    assert steps[6] == "MOVE: approach forward"
    assert steps[2] == ""
    assert steps[3] == ""
    assert steps[5] == ""
    assert steps[7] == ""
    assert all("ACTION:" not in step for step in steps)


def test_format_visible_reasoning_from_steps_omits_action():
    reasoning = (
        "TASK: place the mug "
        "PLAN: move to the mug "
        "VISIBLE OBJECTS: mug [1, 2, 3, 4] "
        "ACTION: should be stripped"
    )

    visible_reasoning = cot_utils.format_visible_reasoning_without_action(reasoning)

    assert visible_reasoning == (
        "TASK: place the mug PLAN: move to the mug VISIBLE OBJECTS: mug [1, 2, 3, 4]"
    )


def test_extract_implicit_cot_steps_empty_reasoning_returns_empty_slots():
    assert cot_utils.extract_implicit_cot_steps("") == [""] * 8
