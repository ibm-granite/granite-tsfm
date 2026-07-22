import tempfile
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


SAVE_MODELS = True

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
        try:
            torch.testing.assert_close(
                state_a[key],
                state_b[key],
                msg=f"[{msg}] Mismatch at: {key}",
            )
        except Exception as e:
            error_keys.append(key)
            last_error = e

    if last_error:
        print("*** Error keys ***")
        print(error_keys)
        raise last_error


@pytest.mark.parametrize("decoder_mode", ["common_channel", "mix_channel"])
@pytest.mark.parametrize("post_init", [False, True])
def test_ttm_from_pretrained(post_init, decoder_mode):
    if decoder_mode == "mix_channel" and post_init:
        # do not test this case
        return

    path = Path(__file__).parent
    post_init_dir = "post_init" if post_init else "no_post_init"

    def ttm_dummy_model(seed=42):
        set_seed(seed)
        conf = TinyTimeMixerConfig(post_init=post_init)
        model = TinyTimeMixerForPrediction(conf)
        return model

    if SAVE_MODELS and TRANSFORMERS_MAJOR_VERSION < 5:
        # In Transformers 4.x, models were saved with:

        if decoder_mode == "common_channel":
            m = ttm_dummy_model()
            m.save_pretrained(path / f"init_stability_test/ttm/{decoder_mode}/{post_init_dir}")
        elif decoder_mode == "mix_channel" and not post_init:
            with tempfile.TemporaryDirectory() as base_tmp:
                m = ttm_dummy_model()
                m.save_pretrained(base_tmp)
                del m

                set_seed(999)
                m = TinyTimeMixerForPrediction.from_pretrained(base_tmp, decoder_mode=decoder_mode)
                m.save_pretrained(path / f"init_stability_test/ttm/{decoder_mode}/{post_init_dir}")
        else:
            raise ValueError("Unknown decoder mode")
        return

    m = ttm_dummy_model()

    if decoder_mode == "mix_channel" and not post_init:
        with tempfile.TemporaryDirectory() as base_tmp:
            m = ttm_dummy_model()
            m.save_pretrained(base_tmp)
            del m

            set_seed(999)
            m = TinyTimeMixerForPrediction.from_pretrained(base_tmp, decoder_mode=decoder_mode)

    m_saved = load_file(path / "init_stability_test" / "ttm" / decoder_mode / post_init_dir / "model.safetensors")

    assert_models_equal(m, m_saved, f"{decoder_mode}/{post_init_dir}")


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

    if SAVE_MODELS and TRANSFORMERS_MAJOR_VERSION < 5:
        # In Transformers 4.x, models were saved with:
        m = tspulse_dummy_model()
        m.save_pretrained(path / f"init_stability_test/tspulse/{task}/{post_init_dir}")
        return

    m = tspulse_dummy_model()
    m_saved = load_file(path / "init_stability_test" / "tspulse" / task / post_init_dir / "model.safetensors")

    assert_models_equal(m, m_saved, f"{task} / {post_init_dir}")
