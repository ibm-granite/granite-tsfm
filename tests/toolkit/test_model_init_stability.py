from pathlib import Path

import pytest
import torch
import transformers
from safetensors.torch import load_file
from transformers import set_seed

from tsfm_public import (
    TinyTimeMixerConfig,
    TinyTimeMixerForPrediction,
    TSPulseConfig,
    TSPulseForClassification,
    TSPulseForReconstruction,
)


TRANSFORMERS_MAJOR_VERSION = int(transformers.__version__.split(".")[0])

TSPULSE_TASK_MODEL_MAP = {
    "reconstruction": TSPulseForReconstruction,
    "classification": TSPulseForClassification,
}

TSPULSE_TASK_CONF_OVERRIDES = {
    "reconstruction": {},
    "classification": {"loss": "cross_entropy"},
}


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


@pytest.mark.parametrize("post_init", [False, True])
def test_ttm_from_pretrained(post_init):
    path = Path(__file__).parent
    post_init_dir = "post_init" if post_init else "no_post_init"

    def ttm_dummy_model(seed=42):
        set_seed(seed)
        conf = TinyTimeMixerConfig(post_init=post_init)
        model = TinyTimeMixerForPrediction(conf)
        return model

    if False and TRANSFORMERS_MAJOR_VERSION < 5:
        # In Transformers 4.x, models were saved with:

        m = ttm_dummy_model()
        m.save_pretrained(path / f"init_stability_test/ttm/{post_init_dir}")
        return

    m = ttm_dummy_model()
    m_saved = load_file(path / "init_stability_test" / "ttm" / post_init_dir / "model.safetensors")

    assert_models_equal(m, m_saved, f"{post_init_dir}")


@pytest.mark.parametrize("post_init", [False, True])
@pytest.mark.parametrize("task", ["reconstruction", "classification"])
def test_tspulse_from_pretrained(task, post_init):
    path = Path(__file__).parent
    model_cls = TSPULSE_TASK_MODEL_MAP[task]
    conf_overrides = TSPULSE_TASK_CONF_OVERRIDES[task]
    post_init_dir = "post_init" if post_init else "no_post_init"

    def tspulse_dummy_model(seed=42):
        set_seed(seed)
        conf = TSPulseConfig(**conf_overrides, post_init=post_init)
        model = model_cls(conf)
        return model

    if True and TRANSFORMERS_MAJOR_VERSION < 5:
        # In Transformers 4.x, models were saved with:
        m = tspulse_dummy_model()
        m.save_pretrained(path / f"init_stability_test/tspulse/{task}/{post_init_dir}")
        return

    m = tspulse_dummy_model()
    m_saved = load_file(path / "init_stability_test" / "tspulse" / task / post_init_dir / "model.safetensors")

    assert_models_equal(m, m_saved, f"{task} / {post_init_dir}")
