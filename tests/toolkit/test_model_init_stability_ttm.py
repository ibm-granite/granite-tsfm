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
)


SAVE_MODELS = True

TRANSFORMERS_MAJOR_VERSION = int(transformers.__version__.split(".")[0])


def assert_models_equal(model, reference_state, msg=None):
    """Compare a model state_dict with a safetensors state dictionary."""
    model_state = model.state_dict()
    msg_prefix = f"[{msg}] " if msg else ""

    assert set(model_state.keys()) == set(reference_state.keys()), f"{msg_prefix}Models have different state_dict keys"

    error_keys = []
    last_error = None

    for key in model_state:
        try:
            torch.testing.assert_close(
                model_state[key],
                reference_state[key],
                msg=f"{msg_prefix}Mismatch at: {key}",
            )
        except Exception as error:
            error_keys.append(key)
            last_error = error

    if last_error is not None:
        print("*** Error keys ***")
        print(error_keys)
        raise last_error


def clone_model_state(model):
    """Clone the model state so later model mutations cannot affect it."""
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def assert_loaded_weights_preserved(
    loaded_model,
    checkpoint_state,
    msg=None,
):
    """
    Verify that every tensor present in the original checkpoint remains
    bitwise identical after loading a model with additional modules.
    """
    loaded_state = loaded_model.state_dict()
    msg_prefix = f"[{msg}] " if msg else ""

    missing_original_keys = set(checkpoint_state) - set(loaded_state)
    assert not missing_original_keys, (
        f"{msg_prefix}Original checkpoint keys disappeared: " f"{sorted(missing_original_keys)}"
    )

    modified_keys = []

    for key, expected in checkpoint_state.items():
        actual = loaded_state[key].detach().cpu()

        if not torch.equal(actual, expected):
            modified_keys.append(key)

    assert not modified_keys, f"{msg_prefix}Checkpoint-loaded parameters were modified: " f"{modified_keys}"


def build_ttm_model(post_init, seed=42):
    """
    Build a common-channel TTM model.

    post_init=False:
        Fresh parameters use native PyTorch constructor initialization.

    post_init=True:
        Fresh parameters are reinitialized by TTM's custom _init_weights().
    """
    set_seed(seed)

    config = TinyTimeMixerConfig(
        post_init=post_init,
        decoder_mode="common_channel",
    )

    return TinyTimeMixerForPrediction(config)


def load_ttm_with_missing_channel_mixer(
    post_init,
    initialization_seed=999,
):
    """
    Save a common-channel checkpoint and load it as mix-channel.

    The channel_feature_mixer parameters are absent from the checkpoint and
    must therefore be initialized according to config.post_init:

      post_init=False -> native PyTorch defaults
      post_init=True  -> TTM custom initialization

    Parameters already present in the checkpoint must remain unchanged.
    """
    with tempfile.TemporaryDirectory() as base_tmp:
        base_model = build_ttm_model(
            post_init=post_init,
            seed=42,
        )

        checkpoint_state = clone_model_state(base_model)
        base_model.save_pretrained(base_tmp)
        del base_model

        load_config = TinyTimeMixerConfig.from_pretrained(base_tmp)
        load_config.post_init = post_init
        load_config.decoder_mode = "mix_channel"

        set_seed(initialization_seed)

        loaded_model = TinyTimeMixerForPrediction.from_pretrained(
            base_tmp,
            config=load_config,
        )

    return loaded_model, checkpoint_state


@pytest.mark.parametrize("post_init", [False, True])
def test_ttm_fresh_model_init_stability(post_init):
    """
    Verify fresh-model initialization parity between Transformers 4 and 5.

    post_init=False:
        Native PyTorch constructor initialization.

    post_init=True:
        TTM custom _init_weights() initialization.
    """
    path = Path(__file__).parent
    post_init_dir = "post_init" if post_init else "no_post_init"

    reference_dir = path / "init_stability_test" / "ttm" / "common_channel" / post_init_dir

    model = build_ttm_model(
        post_init=post_init,
        seed=42,
    )

    if SAVE_MODELS and TRANSFORMERS_MAJOR_VERSION < 5:
        model.save_pretrained(reference_dir)
        return

    reference_state = load_file(reference_dir / "model.safetensors")

    assert_models_equal(
        model,
        reference_state,
        f"fresh/common_channel/{post_init_dir}",
    )


@pytest.mark.parametrize("post_init", [False, True])
def test_ttm_missing_module_init_stability(post_init):
    """
    Verify from_pretrained() when the target architecture contains modules
    that are missing from the checkpoint.

    post_init=False:
        Missing modules use native PyTorch initialization.

    post_init=True:
        Missing modules use TTM custom initialization.

    Both modes:
        Checkpoint-loaded parameters remain bitwise unchanged.
    """
    path = Path(__file__).parent
    post_init_dir = "post_init" if post_init else "no_post_init"

    reference_dir = path / "init_stability_test" / "ttm" / "mix_channel" / post_init_dir

    model, checkpoint_state = load_ttm_with_missing_channel_mixer(
        post_init=post_init,
        initialization_seed=999,
    )

    loaded_keys = set(model.state_dict())
    checkpoint_keys = set(checkpoint_state)
    new_keys = loaded_keys - checkpoint_keys

    assert new_keys, "Expected mix_channel loading to introduce missing/new parameters"

    assert any("channel_feature_mixer" in key for key in new_keys), (
        "Expected newly introduced channel_feature_mixer parameters, " f"but found: {sorted(new_keys)}"
    )

    assert_loaded_weights_preserved(
        model,
        checkpoint_state,
        f"mix_channel/{post_init_dir}",
    )

    if SAVE_MODELS and TRANSFORMERS_MAJOR_VERSION < 5:
        model.save_pretrained(reference_dir)
        return

    reference_state = load_file(reference_dir / "model.safetensors")

    assert_models_equal(
        model,
        reference_state,
        f"missing_modules/mix_channel/{post_init_dir}",
    )
