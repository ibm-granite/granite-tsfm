from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import set_seed

from tsfm_public import TinyTimeMixerConfig, TinyTimeMixerForPrediction, TSPulseConfig, TSPulseForReconstruction


def assert_models_equal(model_a_from_pretrained, model_b_safetensors, msg=None):
    state_a = model_a_from_pretrained.state_dict()
    state_b = model_b_safetensors

    msg_ = f"[{msg}] " if msg else ""
    assert set(state_a.keys()) == set(state_b.keys()), f"{msg_}Models have different state_dict keys"

    for key in state_a:
        torch.testing.assert_close(
            state_a[key],
            state_b[key],
            msg=f"[{msg}] Mismatch at: {key}",
        )


def test_ttm_from_pretrained():
    path = Path(__file__).parent

    def ttm_dummy_model(conf_options={}, seed=42):
        set_seed(seed)
        conf = TinyTimeMixerConfig(**conf_options)
        model = TinyTimeMixerForPrediction(conf)
        return model

    m = ttm_dummy_model({"post_init": False})
    m_4 = load_file(path / "init_stability_test" / "ttm" / "no_post_init" / "model.safetensors")

    assert_models_equal(m, m_4, "no post init")

    m_post_init = ttm_dummy_model({"post_init": True})
    m_4_post_init = load_file(path / "init_stability_test" / "ttm" / "post_init" / "model.safetensors")

    assert_models_equal(m_post_init, m_4_post_init, "with post init")

    # In Transformers 4.x, models were saved with:

    # m = ttm_dummy_model({"post_init": False})
    # m.save_pretrained(path / "init_stability_test/ttm/no_post_init")
    # m_with_init = ttm_dummy_model({"post_init": True})
    # m_with_init.save_pretrained(path / "init_stability_test/ttm/post_init")


def test_tspulse_from_pretrained():
    path = Path(__file__).parent

    def tspulse_dummy_model(conf_options={}, seed=42):
        set_seed(seed)
        conf = TSPulseConfig(**conf_options)
        model = TSPulseForReconstruction(conf)
        return model

    if True:
        # In Transformers 4.x, models were saved with:

        m = tspulse_dummy_model({"post_init": False})
        m.save_pretrained(path / "init_stability_test/tspulse/no_post_init")
        m_with_init = tspulse_dummy_model({"post_init": True})
        m_with_init.save_pretrained(path / "init_stability_test/tspulse/post_init")
        return

    m = tspulse_dummy_model({"post_init": False})
    m_4 = load_file(path / "init_stability_test" / "tspulse" / "no_post_init" / "model.safetensors")

    assert_models_equal(m, m_4, "no post init")

    m_post_init = tspulse_dummy_model({"post_init": True})
    m_4_post_init = load_file(path / "init_stability_test" / "tspulse" / "post_init" / "model.safetensors")

    assert_models_equal(m_post_init, m_4_post_init, "with post init")
