import jax
import jax.numpy as jnp

from openpi.models import gemma_fast


def _make_dummy_module(*, scan: bool) -> gemma_fast.Module:
    config = gemma_fast.get_config("dummy").to_dict()
    config["scan"] = scan
    return gemma_fast.Module(**config, embed_dtype="float32", cache_dtype="float32")


def test_gemma_fast_scan_surfaces_per_layer_hidden_states():
    module = _make_dummy_module(scan=True)
    tokens = jnp.arange(10, dtype=jnp.int32).reshape(2, 5)

    (_, _, out), variables = module.init_with_output(
        jax.random.key(0),
        tokens=tokens,
        return_hidden_states=True,
    )

    hidden_states = out["hidden_states"]

    assert len(hidden_states) == module.depth + 2
    assert all(hidden.shape == (2, 5, module.width) for hidden in hidden_states)
    assert jnp.allclose(hidden_states[-1], out["pre_logits"])

    embedded = module.apply(variables, tokens=tokens, embed_only=True)
    assert jnp.allclose(hidden_states[0], embedded)


def test_gemma_fast_scan_keeps_layer_param_layout():
    module = _make_dummy_module(scan=True)
    tokens = jnp.arange(10, dtype=jnp.int32).reshape(2, 5)

    _, variables = module.init_with_output(jax.random.key(0), tokens=tokens)

    assert set(variables["params"]["layers"].keys()) == {
        "attn",
        "mlp",
        "pre_attention_norm",
        "pre_ffw_norm",
    }
