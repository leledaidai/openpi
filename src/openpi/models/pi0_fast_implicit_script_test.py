import importlib.util
from pathlib import Path

import numpy as np


def _load_script_module():
    script_path = Path(__file__).resolve().parents[3] / "test_pi0_fast_implicit.py"
    spec = importlib.util.spec_from_file_location("test_pi0_fast_implicit", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_decode_reasoning_sections_decodes_each_latent_independently():
    module = _load_script_module()

    class _FakePGTokenizer:
        def decode(self, tokens):
            return "|".join(str(t) for t in tokens)

    reasoning_tokens = np.array(
        [
            [
                [11, 12, 0, 0],
                [21, 22, 23, 0],
                [0, 0, 0, 0],
            ]
        ],
        dtype=np.int32,
    )

    sections = module.decode_reasoning_sections(_FakePGTokenizer(), reasoning_tokens[0])

    assert sections == ["11|12", "21|22|23", ""]


def test_decode_action_tokens_uses_fast_tokenizer_decoder():
    module = _load_script_module()

    class _FakeFastTokenizer:
        def __init__(self):
            self.calls = []

        def extract_actions(self, tokens, action_horizon, action_dim):
            self.calls.append((tokens.copy(), action_horizon, action_dim))
            return np.full((action_horizon, action_dim), 3.5, dtype=np.float32)

    fast_tokenizer = _FakeFastTokenizer()
    action_tokens = np.array([101, 102, 0, 0], dtype=np.int32)

    actions = module.decode_action_tokens(
        fast_tokenizer,
        action_tokens,
        action_horizon=2,
        action_dim=3,
    )

    assert len(fast_tokenizer.calls) == 1
    called_tokens, called_horizon, called_dim = fast_tokenizer.calls[0]
    np.testing.assert_array_equal(called_tokens, action_tokens)
    assert called_horizon == 2
    assert called_dim == 3
    np.testing.assert_allclose(actions, 3.5)


def test_resolve_checkpoint_path_accepts_step_directory():
    module = _load_script_module()

    step_dir = Path("/tmp/example/70000")
    resolved = module.resolve_checkpoint_path(step_dir)

    assert resolved == step_dir / "params"
