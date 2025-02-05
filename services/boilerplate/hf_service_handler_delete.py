"""Service handler for HuggingFace models"""

import importlib
import logging
import pathlib
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    PretrainedConfig,
    PreTrainedModel,
)

from tsfm_public import TimeSeriesPreprocessor

from .inference_payloads import BaseParameters, ForecastingParameters
from .service_handler import (
    ForecastingServiceHandler,
    ServiceHandlerBase,
)
from .tsfm_config import TSFMConfig


LOGGER = logging.getLogger(__file__)


class HuggingFaceHandler(ServiceHandlerBase):
    """Handler for HuggingFace-like models

    Underlying assumption is that the model makes use of PretrainedConfig/PreTrainedModel
    conventions. During init we register the config using HF AutoConfig.

    Args:
        model_id (str): ID of the model
        model_path (Union[str, Path]): Full path to the model folder.
        handler_config (TSFMConfig): Configuration for the service handler

    """

    def __init__(
        self,
        model_id: str,
        model_path: Union[str, Path],
        handler_config: TSFMConfig,
    ):
        super().__init__(model_id=model_id, model_path=model_path, handler_config=handler_config)

        if (
            getattr(handler_config, "model_type", None)
            and getattr(handler_config, "model_config_name", None)
            and getattr(handler_config, "module_path", None)
        ):
            register_config(
                handler_config.model_type,
                handler_config.model_config_name,
                handler_config.module_path,
            )
            LOGGER.info(f"registered {handler_config.model_type}")

        self.config = None
        self.model = None
        self.preprocessor = None

    def load_preprocessor(self, model_path: str) -> TimeSeriesPreprocessor:
        """Load TSFM preprocessor

        Args:
            model_path (str): Local path or HF Hub path to the preprocessor.
        Raises:
            Exception if there is an issue loading the preprocessor. OSError is caught to allow
            loading models when there is no preprocessor.

        Returns:
            TimeSeriesPreprocessor: the loaded preprocessor.
        """

        try:
            preprocessor = TimeSeriesPreprocessor.from_pretrained(model_path)
            LOGGER.info("Successfully loaded preprocessor")
        except OSError:
            preprocessor = None
            LOGGER.info("No preprocessor found")
        except Exception as ex:
            raise ex

        return preprocessor

    def load_hf_config(self, model_path: str, **extra_config_kwargs: Dict[str, Any]) -> PretrainedConfig:
        """Load a HuggingFace config

        Args:
            model_path (str): Local path or HF Hub path for the config.
            extra_config_kwarg: Extra keyword arguments that are passed while loading the config.

        Returns:
            PretrainedConfig: The loaded config.
        """
        # load config, separate from load model, since we may need to inspect config first
        conf = load_config(model_path, **extra_config_kwargs)

        return conf

    def load_hf_model(self, model_path: str, config: PretrainedConfig) -> PreTrainedModel:
        """Load a HuggingFace model from the HF Hub

        Uses the module_path from the handler_config if available to identify the proper class to load
        the model.

        Args:
            model_path (str): Local path or HF Hub path for the model.
            config (PretrainedConfig): A pretrained config object.

        Returns:
            PreTrainedModel: The loaded model.
        """
        model = load_model(
            model_path,
            config=config,
            module_path=self.handler_config.module_path,
        )

        LOGGER.info("Successfully loaded model")
        return model

    @abstractmethod
    def _get_config_kwargs(
        self,
        parameters: Optional[BaseParameters] = None,
        preprocessor: Optional[TimeSeriesPreprocessor] = None,
    ) -> Dict[str, Any]:
        """Helper function to return additional configuration arguments that are used during config load.
        Can be overridden in a subclass to allow specialized model functionality.

        Args:
            parameters (Optional[ForecastingParameters], optional): Request parameters. Defaults to None.
            preprocessor (Optional[TimeSeriesPreprocessor], optional): Time seres preprocessor. Defaults to None.

        Returns:
            Dict[str, Any]: Dictionary of additional arguments that are used later as keyword arguments to the config.
        """
        ...


class ForecastingHuggingFaceHandler(ForecastingServiceHandler, HuggingFaceHandler):
    """Handler for HuggingFace-like models

    Underlying assumption is that the model makes use of PretrainedConfig/PreTrainedModel
    conventions. During init we register the config using HF AutoConfig.

    Args:
        model_id (str): ID of the model
        model_path (Union[str, Path]): Full path to the model folder.
        handler_config (TSFMConfig): Configuration for the service handler

    """

    def __init__(
        self,
        model_id: str,
        model_path: Union[str, Path],
        handler_config: TSFMConfig,
    ):
        self.training_label_names = ["future_values"]

        # !!! Double check which init
        super().__init__(model_id=model_id, model_path=model_path, handler_config=handler_config)

    def _get_config_kwargs(
        self,
        parameters: Optional[ForecastingParameters] = None,
        preprocessor: Optional[TimeSeriesPreprocessor] = None,
    ) -> Dict[str, Any]:
        """Helper function to return additional configuration arguments that are used during config load.
        Can be overridden in a subclass to allow specialized model functionality.

        Args:
            parameters (Optional[ForecastingParameters], optional): Request parameters. Defaults to None.
            preprocessor (Optional[TimeSeriesPreprocessor], optional): Time seres preprocessor. Defaults to None.

        Returns:
            Dict[str, Any]: Dictionary of additional arguments that are used later as keyword arguments to the config.
        """
        return {"num_input_channels": preprocessor.num_input_channels}


class TinyTimeMixerForecastingHandler(ForecastingHuggingFaceHandler):
    """Service handler for the tiny time mixer model"""

    def _get_config_kwargs(
        self,
        parameters: Optional[ForecastingParameters] = None,
        preprocessor: Optional[TimeSeriesPreprocessor] = None,
    ) -> Dict[str, Any]:
        config_kwargs = {
            "num_input_channels": preprocessor.num_input_channels,
            "prediction_filter_length": parameters.prediction_length,
            "exogenous_channel_indices": preprocessor.exogenous_channel_indices,
            "prediction_channel_indices": preprocessor.prediction_channel_indices,
        }
        return config_kwargs


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
        conf = AutoConfig.from_pretrained(model_path, **extra_config_kwargs)
    except (KeyError, ValueError) as exc:  # determine error raised by autoconfig
        if model_type is None or model_config_name is None or module_path is None:
            raise ValueError("model_type, model_config_name, and module_path should be specified.") from exc

        register_config(model_type, model_config_name, module_path)
        conf = AutoConfig.from_pretrained(model_path, **extra_config_kwargs)

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
            raise AttributeError(f"Could not load model class for architecture '{arch}'.") from exc


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
        return model_class.from_pretrained(model_path, config=config)

    return AutoModel.from_pretrained(model_path)
