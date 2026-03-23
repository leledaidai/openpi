import numpy as np

from openpi.models import tokenizer as _tokenizer


def test_tokenize():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=10)
    tokens, masks = tokenizer.tokenize("Hello, world!")

    assert tokens.shape == (10,)
    assert masks.shape == (10,)


def test_fast_tokenizer():
    prompt = "Hello, world!"
    state = np.random.rand(5).astype(np.float32)
    action = np.random.rand(3, 2).astype(np.float32)
    tokenizer = _tokenizer.FASTTokenizer(max_len=256)
    tokens, token_masks, ar_masks, loss_masks = tokenizer.tokenize(prompt, state, action)

    assert tokens.shape == (256,)
    assert token_masks.shape == (256,)
    assert ar_masks.shape == (256,)
    assert loss_masks.shape == (256,)

    act = tokenizer.extract_actions(tokens, 3, 2)
    assert act.shape == (3, 2)


def test_fast_tokenizer_implicit_cot():
    prompt = "Move the mug."
    state = np.random.rand(5).astype(np.float32)
    action = np.random.rand(3, 2).astype(np.float32)
    reasoning = (
        "TASK: move the mug PLAN: grasp then place "
        "VISIBLE OBJECTS: mug [1, 2, 3, 4] ACTION: this must be removed"
    )

    tokenizer = _tokenizer.FASTTokenizer(max_len=64)
    result = tokenizer.tokenize_implicit_cot(
        prompt,
        state,
        action,
        cot_reasoning=reasoning,
        max_step_tokens=12,
    )

    assert result["tokenized_prompt"].shape == (64,)
    assert result["tokenized_prompt_mask"].shape == (64,)
    assert result["tokenized_teacher_cot"].ndim == 1
    assert result["tokenized_teacher_cot_mask"].shape == result["tokenized_teacher_cot"].shape
    assert result["tokenized_action_postfix"].ndim == 1
    assert result["tokenized_action_postfix_mask"].shape == result["tokenized_action_postfix"].shape
    assert result["tokenized_implicit_cot_steps"].shape == (8, 12)
    assert result["tokenized_implicit_cot_steps_mask"].shape == (8, 12)
    assert bool(result["has_cot"])
