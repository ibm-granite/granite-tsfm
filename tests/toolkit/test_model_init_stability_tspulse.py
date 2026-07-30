import tempfile
from pathlib import Path

import pytest
import torch
import transformers
from safetensors.torch import load_file
from transformers import set_seed

from tsfm_public import (
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

TSPULSE_TASK_CONFIG_OVERRIDES = {
    "reconstruction": {},
    "classification": {
        "loss": "cross_entropy",
    },
}


def assert_models_equal(model, reference_state, msg=None):
    """Compare a model state_dict with a safetensors state dictionary."""
    model_state = model.state_dict()
    msg_prefix = f"[{msg}] " if msg else ""

    model_keys = set(model_state)
    reference_keys = set(reference_state)

    assert model_keys == reference_keys, (
        f"{msg_prefix}Models have different state_dict keys. "
        f"Only in model: {sorted(model_keys - reference_keys)}. "
        f"Only in reference: {sorted(reference_keys - model_keys)}."
    )

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


def get_channel_feature_mixers(model):
    """
    Return all TSPulseChannelFeatureMixerBlock modules without importing the
    internal modeling class directly.
    """
    return [
        (name, module)
        for name, module in model.named_modules()
        if module.__class__.__name__ == "TSPulseChannelFeatureMixerBlock"
    ]


def expected_half_identity(linear):
    """Return the exact 0.5 * identity matrix expected by free-channel-flow."""
    return (
        torch.eye(
            linear.out_features,
            linear.in_features,
            dtype=linear.weight.dtype,
            device=linear.weight.device,
        )
        * 0.5
    )


def assert_identity_channel_mixers(model):
    """
    Verify the exact free-channel-flow identity-style parameter initialization.

    Expected initialization:
      fc1.weight = 0.5 * I
      fc2.weight = 0.5 * I
      fc1.bias   = 0
      fc2.bias   = 0

    When gated attention is present:
      attention weight = 0
      attention bias   = 0
    """
    mixers = get_channel_feature_mixers(model)

    assert mixers, "No TSPulseChannelFeatureMixerBlock modules were found"

    for module_name, module in mixers:
        mlp = module.mlp

        torch.testing.assert_close(
            mlp.fc1.weight,
            expected_half_identity(mlp.fc1),
            rtol=0,
            atol=0,
            msg=f"{module_name}.mlp.fc1.weight is not 0.5 * identity",
        )

        torch.testing.assert_close(
            mlp.fc2.weight,
            expected_half_identity(mlp.fc2),
            rtol=0,
            atol=0,
            msg=f"{module_name}.mlp.fc2.weight is not 0.5 * identity",
        )

        if mlp.fc1.bias is not None:
            torch.testing.assert_close(
                mlp.fc1.bias,
                torch.zeros_like(mlp.fc1.bias),
                rtol=0,
                atol=0,
                msg=f"{module_name}.mlp.fc1.bias is not zero",
            )

        if mlp.fc2.bias is not None:
            torch.testing.assert_close(
                mlp.fc2.bias,
                torch.zeros_like(mlp.fc2.bias),
                rtol=0,
                atol=0,
                msg=f"{module_name}.mlp.fc2.bias is not zero",
            )

        if hasattr(module, "gating_block"):
            attn_layer = module.gating_block.attn_layer

            torch.testing.assert_close(
                attn_layer.weight,
                torch.zeros_like(attn_layer.weight),
                rtol=0,
                atol=0,
                msg=f"{module_name}.gating_block.attn_layer.weight is not zero",
            )

            if attn_layer.bias is not None:
                torch.testing.assert_close(
                    attn_layer.bias,
                    torch.zeros_like(attn_layer.bias),
                    rtol=0,
                    atol=0,
                    msg=f"{module_name}.gating_block.attn_layer.bias is not zero",
                )


def assert_custom_channel_mixer_init(model):
    """
    Verify the selected TSPulse custom Linear initialization for channel
    mixers when free_channel_flow=False and post_init=True.

    The test config uses init_linear='normal', whose observable deterministic
    property is that all Linear biases are initialized to zero.
    """
    mixers = get_channel_feature_mixers(model)

    assert mixers, "No TSPulseChannelFeatureMixerBlock modules were found"

    for module_name, module in mixers:
        linear_layers = [
            ("mlp.fc1", module.mlp.fc1),
            ("mlp.fc2", module.mlp.fc2),
        ]

        if hasattr(module, "gating_block"):
            linear_layers.append(("gating_block.attn_layer", module.gating_block.attn_layer))

        for relative_name, linear in linear_layers:
            if linear.bias is not None:
                torch.testing.assert_close(
                    linear.bias,
                    torch.zeros_like(linear.bias),
                    rtol=0,
                    atol=0,
                    msg=(
                        f"{module_name}.{relative_name}.bias was not initialized "
                        "by the TSPulse custom Linear policy"
                    ),
                )

        fc1_is_identity = torch.equal(
            module.mlp.fc1.weight,
            expected_half_identity(module.mlp.fc1),
        )
        fc2_is_identity = torch.equal(
            module.mlp.fc2.weight,
            expected_half_identity(module.mlp.fc2),
        )

        assert not (fc1_is_identity and fc2_is_identity), (
            f"{module_name} unexpectedly received free-channel-flow identity "
            "initialization while free_channel_flow=False"
        )


def assert_pytorch_default_channel_mixer_init(model):
    """
    Verify that channel mixers did not receive TSPulse custom or identity
    initialization when both conditions are disabled:

      free_channel_flow=False
      post_init=False

    Native PyTorch Linear initialization produces non-zero random biases for
    at least one tested Linear layer with the deterministic seeds used here.
    """
    mixers = get_channel_feature_mixers(model)

    assert mixers, "No TSPulseChannelFeatureMixerBlock modules were found"

    bias_tensors = []

    for _, module in mixers:
        linear_layers = [
            module.mlp.fc1,
            module.mlp.fc2,
        ]

        if hasattr(module, "gating_block"):
            linear_layers.append(module.gating_block.attn_layer)

        for linear in linear_layers:
            if linear.bias is not None:
                bias_tensors.append(linear.bias)

    assert bias_tensors, "No channel-mixer Linear biases were found"

    assert any(torch.count_nonzero(bias).item() > 0 for bias in bias_tensors), (
        "All channel-mixer biases are zero. This suggests TSPulse custom "
        "initialization was applied even though post_init=False."
    )

    for module_name, module in mixers:
        fc1_is_identity = torch.equal(
            module.mlp.fc1.weight,
            expected_half_identity(module.mlp.fc1),
        )
        fc2_is_identity = torch.equal(
            module.mlp.fc2.weight,
            expected_half_identity(module.mlp.fc2),
        )

        assert not (fc1_is_identity and fc2_is_identity), (
            f"{module_name} unexpectedly received identity initialization " "while free_channel_flow=False"
        )


def assert_channel_mixer_policy(
    model,
    post_init,
    free_channel_flow,
):
    """Assert the expected channel-mixer initialization policy."""
    if free_channel_flow:
        # Architecture-specific policy takes priority over post_init.
        assert_identity_channel_mixers(model)
    elif post_init:
        # Global TSPulse custom initialization.
        assert_custom_channel_mixer_init(model)
    else:
        # Native PyTorch constructor/reset_parameters initialization.
        assert_pytorch_default_channel_mixer_init(model)


def build_tspulse_model(
    task,
    post_init,
    free_channel_flow,
    mode="mix_channel",
    decoder_mode="mix_channel",
    seed=42,
):
    """
    Build a TSPulse task model with an explicit initialization policy.

    Policy under test:
      free_channel_flow=True
          Relevant channel mixers use identity-style initialization,
          regardless of post_init.

      free_channel_flow=False and post_init=False
          Parameters use native PyTorch initialization.

      free_channel_flow=False and post_init=True
          Parameters use TSPulse custom initialization.
    """
    set_seed(seed)

    config_kwargs = {
        **TSPULSE_TASK_CONFIG_OVERRIDES[task],
        "post_init": post_init,
        "mode": mode,
        "decoder_mode": decoder_mode,
        "free_channel_flow": free_channel_flow,
        "channel_mix_init": "identity",
        "gated_attn": True,
        # Make the custom Linear policy explicit and easy to validate.
        "init_linear": "normal",
        "init_std": 0.02,
    }

    config = TSPulseConfig(**config_kwargs)
    model_cls = TSPULSE_TASK_MODEL_MAP[task]

    return model_cls(config)


def load_tspulse_with_missing_channel_mixers(
    task,
    post_init,
    free_channel_flow,
    initialization_seed=999,
):
    """
    Save a common-channel checkpoint and load it as mix-channel.

    The channel_feature_mixer modules are absent from the checkpoint, so this
    exercises missing-module initialization.

    Expected missing-module policy:
      free_channel_flow=True
          Identity-style initialization, regardless of post_init.

      free_channel_flow=False and post_init=False
          Native PyTorch initialization.

      free_channel_flow=False and post_init=True
          TSPulse custom initialization.

    Parameters already present in the checkpoint must remain unchanged.
    """
    with tempfile.TemporaryDirectory() as base_tmp:
        base_model = build_tspulse_model(
            task=task,
            post_init=post_init,
            free_channel_flow=free_channel_flow,
            mode="common_channel",
            decoder_mode="common_channel",
            seed=42,
        )

        checkpoint_state = clone_model_state(base_model)
        base_model.save_pretrained(base_tmp)
        del base_model

        load_config = TSPulseConfig.from_pretrained(base_tmp)
        load_config.post_init = post_init
        load_config.mode = "mix_channel"
        load_config.decoder_mode = "mix_channel"
        load_config.free_channel_flow = free_channel_flow
        load_config.channel_mix_init = "identity"
        load_config.gated_attn = True
        load_config.init_linear = "normal"
        load_config.init_std = 0.02

        set_seed(initialization_seed)

        model_cls = TSPULSE_TASK_MODEL_MAP[task]
        loaded_model = model_cls.from_pretrained(
            base_tmp,
            config=load_config,
        )

    return loaded_model, checkpoint_state


def reference_directory(
    task,
    scenario,
    post_init,
    free_channel_flow,
):
    """Return the reference checkpoint directory for one test combination."""
    path = Path(__file__).parent
    post_init_dir = "post_init" if post_init else "no_post_init"
    flow_dir = "free_channel_flow" if free_channel_flow else "standard_channel_flow"

    return path / "init_stability_test" / "tspulse" / task / scenario / flow_dir / post_init_dir


@pytest.mark.parametrize(
    "task",
    ["reconstruction", "classification"],
)
@pytest.mark.parametrize("post_init", [False, True])
@pytest.mark.parametrize("free_channel_flow", [False, True])
def test_tspulse_fresh_model_init_stability(
    task,
    post_init,
    free_channel_flow,
):
    """
    Verify all fresh-model initialization policies and v4/v5 parity.

    Cases:
      free_channel_flow=False, post_init=False
          Native PyTorch initialization.

      free_channel_flow=False, post_init=True
          TSPulse custom initialization.

      free_channel_flow=True, post_init=False
          Identity-style channel-mixer initialization.

      free_channel_flow=True, post_init=True
          Identity-style channel-mixer initialization.
    """
    reference_dir = reference_directory(
        task=task,
        scenario="fresh_mix_channel",
        post_init=post_init,
        free_channel_flow=free_channel_flow,
    )

    model = build_tspulse_model(
        task=task,
        post_init=post_init,
        free_channel_flow=free_channel_flow,
        mode="mix_channel",
        decoder_mode="mix_channel",
        seed=42,
    )

    assert_channel_mixer_policy(
        model,
        post_init=post_init,
        free_channel_flow=free_channel_flow,
    )

    if SAVE_MODELS and TRANSFORMERS_MAJOR_VERSION < 5:
        model.save_pretrained(reference_dir)
        return

    reference_state = load_file(reference_dir / "model.safetensors")

    assert_models_equal(
        model,
        reference_state,
        (f"fresh/{task}/" f"post_init={post_init}/" f"free_channel_flow={free_channel_flow}"),
    )


@pytest.mark.parametrize(
    "task",
    ["reconstruction", "classification"],
)
@pytest.mark.parametrize("post_init", [False, True])
@pytest.mark.parametrize("free_channel_flow", [False, True])
def test_tspulse_missing_module_init_stability(
    task,
    post_init,
    free_channel_flow,
):
    """
    Verify all missing-module initialization policies and v4/v5 parity.

    A common-channel checkpoint is loaded as mix-channel, introducing new
    channel_feature_mixer modules.

    In every case:
      - New channel mixers follow the selected initialization policy.
      - All checkpoint-loaded tensors remain bitwise unchanged.
    """
    reference_dir = reference_directory(
        task=task,
        scenario="missing_mix_channel",
        post_init=post_init,
        free_channel_flow=free_channel_flow,
    )

    model, checkpoint_state = load_tspulse_with_missing_channel_mixers(
        task=task,
        post_init=post_init,
        free_channel_flow=free_channel_flow,
        initialization_seed=999,
    )

    loaded_keys = set(model.state_dict())
    checkpoint_keys = set(checkpoint_state)
    new_keys = loaded_keys - checkpoint_keys

    assert new_keys, "Expected mix-channel loading to introduce missing/new parameters"

    channel_mixer_keys = sorted(key for key in new_keys if "channel_feature_mixer" in key)

    assert channel_mixer_keys, (
        "Expected newly introduced channel_feature_mixer parameters, " f"but found: {sorted(new_keys)}"
    )

    assert_loaded_weights_preserved(
        model,
        checkpoint_state,
        (f"{task}/" f"post_init={post_init}/" f"free_channel_flow={free_channel_flow}"),
    )

    assert_channel_mixer_policy(
        model,
        post_init=post_init,
        free_channel_flow=free_channel_flow,
    )

    if SAVE_MODELS and TRANSFORMERS_MAJOR_VERSION < 5:
        model.save_pretrained(reference_dir)
        return

    reference_state = load_file(reference_dir / "model.safetensors")

    assert_models_equal(
        model,
        reference_state,
        (f"missing_modules/{task}/" f"post_init={post_init}/" f"free_channel_flow={free_channel_flow}"),
    )
