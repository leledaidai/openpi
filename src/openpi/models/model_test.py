import jax
import numpy as np

from openpi import transforms as _transforms
from openpi.models import pi0_config
from openpi.models import pi0_fast
from openpi.models import pi0_fast_implicit
from openpi.models import pi_cot
from openpi.shared import nnx_utils


def test_pi0_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


# def test_pi0_lora_model():
#     key = jax.random.key(0)
#     config = pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
#     model = config.create(key)

#     batch_size = 2
#     obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

#     loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
#     assert loss.shape == (batch_size, config.action_horizon)

#     actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
#     assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi0_fast_model():
    key = jax.random.key(0)
    config = pi0_fast.Pi0FASTConfig()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size,)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs)
    assert actions.shape == (batch_size, 256)


# def test_pi0_fast_lora_model():
#     key = jax.random.key(0)
#     config = pi0_fast.Pi0FASTConfig(paligemma_variant="gemma_2b_lora")
#     model = config.create(key)

#     batch_size = 2
#     obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

#     loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
#     assert loss.shape == (batch_size,)

#     actions = nnx_utils.module_jit(model.sample_actions)(key, obs)
#     assert actions.shape == (batch_size, 256)

#     lora_filter = nnx_utils.PathRegex(".*lora.*")
#     model_state = nnx.state(model)

#     lora_state_elems = list(model_state.filter(lora_filter))
#     assert len(lora_state_elems) > 0


# @pytest.mark.manual
# def test_model_restore():
#     key = jax.random.key(0)
#     config = pi0_config.Pi0Config()

#     batch_size = 2
#     obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

#     model = config.load(
#         _model.restore_params(download.maybe_download("gs://openpi-assets/checkpoints/pi0_base/params"))
#     )

#     loss = model.compute_loss(key, obs, act)
#     assert loss.shape == (batch_size, config.action_horizon)

#     actions = model.sample_actions(key, obs, num_steps=10)
#     assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi_cot_model():
    """Smoke test: verify shapes and that jax.lax.scan-based CoT generation works correctly.

    Uses dummy gemma variant (tiny model) so no GPU/checkpoint needed.
    max_cot_tokens=4 keeps the test fast while still exercising the scan loop.
    """
    key = jax.random.key(0)
    config = pi_cot.PiCOTConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=8,
        action_horizon=4,
        max_token_len=16,
        max_cot_tokens=4,
        max_fast_tokens=4,
    )
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    # Test compute_loss
    loss, aux = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    # Test sample_actions (exercises the full scan-based CoT generation path)
    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=2)
    assert actions.shape == (batch_size, config.action_horizon, config.action_dim)


def test_pi0_fast_implicit_inference_generates_tokens_then_decodes_actions():
    key = jax.random.key(0)
    config = pi0_fast_implicit.Pi0FASTImplicitConfig(
        paligemma_variant="dummy",
        action_dim=4,
        action_horizon=3,
        max_token_len=32,
        num_latent=2,
        max_prefix_len=8,
        max_action_token_len=6,
        max_cot_step_len=4,
    )
    model = config.create(key)
    obs = config.fake_obs(batch_size=1)

    action_tokens = nnx_utils.module_jit(model.sample_actions)(key, obs, max_decoding_steps=5)

    assert action_tokens.shape == (1, 5)
    assert np.issubdtype(np.asarray(action_tokens).dtype, np.integer)

    class _FakeFastTokenizer:
        def __init__(self):
            self.calls = []

        def extract_actions(self, tokens, action_horizon, action_dim):
            self.calls.append(tokens.copy())
            return np.full((action_horizon, action_dim), 7.0, dtype=np.float32)

    fake_tokenizer = _FakeFastTokenizer()
    extractor = _transforms.ExtractFASTActions(
        fake_tokenizer,
        action_horizon=config.action_horizon,
        action_dim=config.action_dim,
    )

    decoded = extractor({"actions": np.asarray(action_tokens[0])})

    assert len(fake_tokenizer.calls) == 1
    np.testing.assert_array_equal(fake_tokenizer.calls[0], np.asarray(action_tokens[0], dtype=np.int32))
    assert decoded["actions"].shape == (config.action_horizon, config.action_dim)
    np.testing.assert_allclose(decoded["actions"], 7.0)
