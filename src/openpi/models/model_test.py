from flax import nnx
import jax
import pytest

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import pi0_fast
from openpi.models import pi_cot
from openpi.shared import download
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

