"""Functions for generating model-specific parameter payloads"""

# Standard
import importlib
import inspect
import logging
from types import NoneType
from typing import Optional, Set, Union, get_args

import pydantic

# Third Party
from pydantic import BaseModel, ConfigDict, Field


LOGGER = logging.getLogger(__file__)


def get_pydantic_parameters(
    base_name: str,
    config_class_path: str,
    include_set: Optional[Set[str]] = None,
    exclude_set: Optional[Set[str]] = None,
) -> pydantic.BaseModel:
    """Creates a pydantic parameter model for a given HF config class.

    Only one of `include_set` or `exclude_set` should be set. If None they have no effect.

    Args:
        base_name (str): Base configuration object name used to indentify configuration object.
        config_class_path (str): Complete path to module containing configuration class.
        include_set (Optional[Set[str]], optional): Set of parameters to include. Defaults to None.
        exclude_set (Optional[Set[str]], optional): Set of parameters to exclude. Defaults to None.

    Returns:
        pydantic.BaseModel: Resulting parameter model
    """

    if include_set is not None and exclude_set is not None:
        raise ValueError("Only one of `include_set` or `exclude_set` should be specified.")

    config_class_name = f"{base_name}Config"

    module = importlib.import_module(config_class_path)
    config_class = getattr(module, config_class_name)

    params = _create_pydantic_model_from_hf_config(
        config_class,
        f"{base_name}Parameters",
        include_set=include_set,
        exclude_set=exclude_set,
    )
    return params


def _create_pydantic_model_from_hf_config(
    config_class: type,
    model_name: str,
    include_set: Optional[Set[str]] = None,
    exclude_set: Optional[Set[str]] = None,
) -> pydantic.BaseModel:
    """Helper function to create a pydantic model from a HuggingFace config class.

    Args:
        config_class (type): HF config class.
        model_name (str): Name of the resulting model.

    Returns:
        type[BaseModel]: A pydantic parameter model
    """
    field_definitions = {}

    sig = inspect.signature(config_class.__init__)
    for name, param in sig.parameters.items():
        if (
            name in ["self", "kwargs"]
            or ((include_set is not None) and (name not in include_set))
            or ((exclude_set is not None) and (name in exclude_set))
        ):
            continue

        if NoneType in get_args(param.annotation):
            # the magic happens
            LOGGER.debug(f"Rewriting optional parameter {name}")

            new_args = tuple([arg for arg in get_args(param.annotation) if arg is not NoneType])
            # to do: reset defaults appropriately
            if len(new_args) > 1:
                field_definitions[name] = (Union[new_args], param.default)
            else:
                field_definitions[name] = (new_args[0], param.default)
        else:
            field_definitions[name] = (param.annotation, param.default)

    return pydantic.create_model(model_name, **field_definitions)


include_set = {
    "distribution_output",
    "loss",
    "attention_dropout",
    "dropout",
    "positional_dropout",
    "path_dropout",
    "ff_dropout",
    "pooling_type",
    "head_dropout",
    "output_range",
    "num_parallel_samples",
}

PatchTSTParameters = get_pydantic_parameters(
    base_name="PatchTST",
    config_class_path="transformers.models.patchtst.configuration_patchtst",
    include_set=include_set,
)


include_set = {
    "num_parallel_samples",
    "dropout",
    "loss",
    "head_dropout",
    "distribution_output",
    "prediction_length",
    "output_range",
    "pooling_type",
}

PatchTSMixerParameters: BaseModel = get_pydantic_parameters(
    base_name="PatchTSMixer",
    config_class_path="transformers.models.patchtsmixer.configuration_patchtsmixer",
    include_set=include_set,
)


# include_set = {
#     "dropout",
#     "loss",
#     "head_dropout",
#     "distribution_output",
#     "num_parallel_samples",
#     # decoder parameters
#     "decoder_num_layers",
#     "decoder_d_model_scale",
#     "decoder_adaptive_patching_levels",
#     "decoder_raw_residual",
#     "decoder_mode",
#     "use_decoder",
#     # forecast channel mixing wit exog support
#     "enable_forecast_channel_mixing",
#     "fcm_gated_attn",
#     "fcm_context_length",
#     "fcm_use_mixer",
#     "fcm_mix_layers",
# }

# TinyTimeMixerParameters: BaseModel = get_pydantic_parameters(
#     base_name="TinyTimeMixer",
#     config_class_path="tsfm.models.tinytimemixer.configuration_tinytimemixer",
#     include_set=include_set,
# )


class TinyTimeMixerParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dropout: float = Field(default=0.2, description="Dropout")
    head_dropout: float = Field(default=0.2, description="Head dropout")
    # check
    loss: str = Field(default="mse", description="Loss")
    # check
    distribution_output: str = Field(
        default="student_t", description="Distribution output specification"
    )  # student_t or normal or negative_binomial
    num_parallel_samples: int = Field(
        default=100,
        description="Number of parallel samples when sampling from the output distribution",
    )
    # check
    decoder_num_layers: int = Field(default=8, description="Number of decoder layers")
    decoder_adaptive_patching_levels: int = Field(default=0, description="Number of adaptive patching levels")
    decoder_raw_residual: bool = Field(
        default=False,
        description="Flag to enable merging of raw embedding with encoder embedding for decoder input",
    )
    decoder_mode: str = Field(
        default="mix_channel",
        description="Use `common_channel` for channel-independent modeling and `mix_channel` for channel-mixing modeling",
    )
    use_decoder: bool = Field(default=True, description="Enable use of the decoder")
    enable_forecast_channel_mixing: bool = Field(
        default=True,
        description="Use forecast channel mixing when forecasting, useful when exogenous features are present",
    )
    fcm_gated_attn: bool = Field(default=True, description="Enable gated attention in the forecast channel mixer")
    fcm_context_length: int = Field(default=1, description="The context length for forecast channel mixing")
    fcm_use_mixer: bool = Field(default=True, description="Use the forecast channel mixer")
    fcm_mix_layers: int = Field(default=2, description="Number of mixing layers for the forecast channel mixer")
    fcm_prepend_past: bool = Field(default=True, description="Prepend last context for forecast reconciliation")
