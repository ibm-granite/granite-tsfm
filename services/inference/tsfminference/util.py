# Copyright contributors to the TSFM project
#
"""Utilities to support custom time series models"""

import importlib
import logging
import pathlib
from typing import Optional, Union

import transformers
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel


LOGGER = logging.getLogger(__file__)


def register_config(model_type: str, model_config_name: str, module_path: str) -> None:
    """Register a configuration for a particular model architecture

    Args:
        model_type (Optional[str], optional): The type of the model, from the model implementation. Defaults to None.
        model_config_name (Optional[str], optional): The name of configuration class for the model. Defaults to None.
        module_path (Optional[str], optional): Python module path that can be used to load the
            config/model. Defaults to None.

    Raises:
        RuntimeError: Raised when the module cannot be imported from the provided module path.
    """
    # example
    # model_type: "tinytimemixer"
    # model_config_name: "TinyTimeMixerConfig"
    # module_path: "tsfm"  # place where config should be importable

    # AutoConfig.register("tinytimemixer", TinyTimeMixerConfig)
    try:
        mod = importlib.import_module(module_path)
        conf_class = getattr(mod, model_config_name, None)
    except ModuleNotFoundError as exc:  # modulenot found, key error ?
        raise RuntimeError(f"Could not load {model_config_name} from {module_path}") from exc

    if conf_class is not None:
        AutoConfig.register(model_type, conf_class)
    else:
        # issue warning?
        pass


def load_config(
    model_path: Union[str, pathlib.Path],
    model_type: Optional[str] = None,
    model_config_name: Optional[str] = None,
    module_path: Optional[str] = None,
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
        conf = AutoConfig.from_pretrained(model_path)
    except (KeyError, ValueError) as exc:  # determine error raised by autoconfig
        if model_type is None or model_config_name is None or module_path is None:
            raise ValueError("model_type, model_config_name, and module_path should be specified.") from exc

        register_config(model_type, model_config_name, module_path)
        conf = AutoConfig.from_pretrained(model_path)

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
            mod = importlib.import_module(module_path)
        except ModuleNotFoundError as exc:
            raise AttributeError("Could not load module '{module_path}'.") from exc
    else:
        mod = transformers

    # get architecture from model config
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        try:
            model_class = getattr(mod, arch)
            return model_class
        except AttributeError as exc:
            # catch specific error import error or attribute error
            raise AttributeError("Could not load model class for architecture '{arch}'.") from exc


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
        raise ValueError("Config must be provided when loading from a custom module_path")

    if config is not None:
        model_class = _get_model_class(config, module_path=module_path)
        LOGGER.info(f"Found model class: {model_class.__name__}")
        model = model_class.from_pretrained(model_path, config=config)
        return model

    model = AutoModel.from_pretrained(model_path)
    LOGGER.info(f"Found model class: {model.__class__.__name__}")
    return model
