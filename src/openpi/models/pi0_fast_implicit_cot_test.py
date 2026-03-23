import flax.nnx as nnx
import jax

from openpi.models import pi0_fast_implicit_cot as _implicit


def test_pi0_fast_implicit_cot_lora_freezes_only_main_llm():
    config = _implicit.Pi0FASTImplicitCoTConfig(
        paligemma_variant="gemma_2b_lora",
        action_dim=8,
        action_horizon=4,
        max_token_len=16,
        implicit_max_step_tokens=8,
    )
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))
    frozen = nnx.state(abstract_model, nnx.All(nnx.Param, config.get_freeze_filter())).flat_state()

    assert len(frozen) > 0
    assert all("lora" not in path for path in frozen)
    assert all("main_llm" in path for path in frozen)
    assert all("decoder_llm" not in path for path in frozen)


def test_pi0_fast_implicit_cot_model_smoke():
    key = jax.random.key(0)
    config = _implicit.Pi0FASTImplicitCoTConfig(
        paligemma_variant="dummy",
        decoder_variant="dummy",
        action_dim=8,
        action_horizon=4,
        max_token_len=16,
        implicit_max_step_tokens=4,
        latent_decode_max_tokens=4,
        prj_dim=16,
    )
    model = config.create(key)

    obs, act = config.fake_obs(batch_size=2), config.fake_act(batch_size=2)

    loss, aux = model.compute_loss(key, obs, act)
    assert loss.shape == (2,)
    assert set(aux) >= {"student_action_loss", "teacher_ce_loss", "decoder_loss", "distill_loss"}

    action_tokens = model.sample_actions(key, obs, max_decoding_steps=8)
    assert action_tokens.shape == (2, 8)

    debug = model.sample_actions_with_debug(key, obs, max_decoding_steps=8, latent_decode_max_tokens=4)
    assert debug["actions"].shape == (2, 8)
    assert debug["latent_step_token_ids"].shape == (2, 8, 4)
