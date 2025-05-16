"""Service handler utils for TSFM models"""

import importlib
import logging
import pathlib
from typing import Any, Dict, Optional, Union

import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    PretrainedConfig,
    PreTrainedModel,
)

import tsfm_public
from tsfm_public.toolkit.hf_util import register_config

from . import TSFM_ALLOW_LOAD_FROM_HF_HUB


LOCAL_FILES_ONLY = not TSFM_ALLOW_LOAD_FROM_HF_HUB


LOGGER = logging.getLogger(__file__)


def load_config(
    model_path: Union[str, pathlib.Path],
    model_type: Optional[str] = None,
    model_config_name: Optional[str] = None,
    module_path: Optional[str] = None,
    **extra_config_kwargs: Dict[str, Any],
) -> PretrainedConfig:
    """Load configuration

    Attempts to load the configuration, if it is not loadable, then we register it with the AutoConfig mechanism.

    Args:
        model_path (pathlib.Path): The path from which to load the config.
        model_type (Optional[str], optional): The type of the model, from the model implementation. Defaults to None.
        model_config_name (Optional[str], optional): The name of configuration class for the model. Defaults to None.
        module_path (Optional[str], optional): Python module path that can be used to load the
            config/model. Defaults to None.

    Returns:
        PretrainedConfig: The configuration object corresponding to the pretrained model.
    """
    # load config first try autoconfig, if not then we register and load
    try:
        conf = AutoConfig.from_pretrained(model_path, local_files_only=LOCAL_FILES_ONLY, **extra_config_kwargs)
    except (KeyError, ValueError) as exc:  # determine error raised by autoconfig
        if model_type is None or model_config_name is None or module_path is None:
            raise ValueError("model_type, model_config_name, and module_path should be specified.") from exc

        register_config(model_type, model_config_name, module_path)
        conf = AutoConfig.from_pretrained(model_path, local_files_only=LOCAL_FILES_ONLY, **extra_config_kwargs)

    return conf


def _get_model_class(config: PretrainedConfig, module_path: Optional[str] = None) -> type:
    """Helper to find model class based on config object

    First the module_path will be checked if it can be loaded in the current environment. If not
    then the transformers library will be used.

    Args:
        config (PretrainedConfig): HF configuration for the model.
        module_path (Optional[str], optional): Python module path that can be used to load the
            config/model. Defaults to None.

    Raises:
        AttributeError: Raised if the module at module_path cannot be loaded.
        AttributeError: If the architecture provided by the config cannot be loaded from
            the module.

    Returns:
        type: The class for the model.
    """
    if module_path is not None:
        try:
            mods = [importlib.import_module(module_path)]
        except ModuleNotFoundError as exc:
            raise AttributeError("Could not load module '{module_path}'.") from exc
    else:
        mods = [transformers, tsfm_public]

    # get architecture from model config
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        model_class = None
        for mod in mods:
            model_class = getattr(mod, arch, None)
            if model_class is not None:
                return model_class
        raise AttributeError(f"Could not load model class for architecture '{arch}'.")

        # try:
        #     model_class = getattr(mod, arch)
        #     return model_class
        # except AttributeError as exc:
        #     # catch specific error import error or attribute error
        #     raise AttributeError(f"Could not load model class for architecture '{arch}'.") from exc


def load_model(
    model_path: Union[str, pathlib.Path],
    config: Optional[PretrainedConfig] = None,
    module_path: Optional[str] = None,
) -> PreTrainedModel:
    """Load a pretrained model.
    If module_path is provided, load the model using the provided module path.

    Args:
        model_path (Union[str, pathlib.Path]): Path to a location where the model can be loaded.
        config (Optional[PretrainedConfig], optional): HF Configuration object. Defaults to None.
        module_path (Optional[str], optional): Python module path that can be used to load the
            config/model. Defaults to None.

    Raises:
        ValueError: Raised if loading from a module_path and a configuration object is not provided.

    Returns:
        PreTrainedModel: The loaded pretrained model.
    """

    if module_path is not None and config is None:
        return ValueError("Config must be provided when loading from a custom module_path")

    if config is not None:
        model_class = _get_model_class(config, module_path=module_path)
        LOGGER.info(f"Found model class: {model_class.__name__}")
        return model_class.from_pretrained(
            model_path,
            config=config,
            local_files_only=LOCAL_FILES_ONLY,
        )

    return AutoModel.from_pretrained(
        model_path,
        local_files_only=LOCAL_FILES_ONLY,
    )
