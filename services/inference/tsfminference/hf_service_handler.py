"""Service handler for HuggingFace models"""

import copy
import importlib
import logging
import pathlib
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
import transformers
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel

from tsfm_public import TimeSeriesForecastingPipeline, TimeSeriesPreprocessor

from .inference_payloads import (
    ForecastingMetadataInput,
    ForecastingParameters,
)
from .service_handler import ServiceHandler


LOGGER = logging.getLogger(__file__)


class HuggingFaceHandler(ServiceHandler):
    def __init__(
        self,
        model_id: Union[str, Path],
        tsfm_config: Dict[str, Any],
    ):
        super().__init__(model_id=model_id, tsfm_config=tsfm_config)

        if "model_type" in tsfm_config and "model_config_name" in tsfm_config and "module_path" in tsfm_config:
            register_config(
                tsfm_config["model_type"],
                tsfm_config["model_config_name"],
                tsfm_config["module_path"],
            )
            LOGGER.info(f"registered {tsfm_config['model_type']}")

    def load_preprocessor(self, model_path: str) -> Union[Optional[TimeSeriesPreprocessor], Optional[Exception]]:
        # load preprocessor
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
        # load config, separate from load model, since we may need to inspect config first
        conf = load_config(model_path, **extra_config_kwargs)

        return conf

    def load_hf_model(
        self, model_path: str, config: PretrainedConfig
    ) -> Union[Optional[PreTrainedModel], Optional[Exception]]:
        model = load_model(
            model_path,
            config=config,
            module_path=self.tsfm_config.get("module_path", None),
        )

        LOGGER.info("Successfully loaded model")
        return model

    def _get_config_kwargs(
        self,
        parameters: Optional[ForecastingParameters] = None,
        preprocessor: Optional[TimeSeriesPreprocessor] = None,
    ):
        return {}

    def _prepare(
        self,
        schema: Optional[ForecastingMetadataInput] = None,
        parameters: Optional[ForecastingParameters] = None,
    ) -> "HuggingFaceHandler":
        # to do: use config parameters below
        # issue: may need to know data length to set parameters upon model load (multst)

        preprocessor_params = copy.deepcopy(schema.model_dump())
        preprocessor_params["prediction_length"] = parameters.prediction_length

        LOGGER.info(f"Preprocessor params: {preprocessor_params}")

        preprocessor = self.load_preprocessor(self.model_id)

        if preprocessor is None:
            preprocessor = TimeSeriesPreprocessor(
                **preprocessor_params,
                scaling=False,
                encode_categorical=False,
            )
            # we don't set context length or prediction length above because it is not needed for inference

        model_config_kwargs = self._get_config_kwargs(
            parameters=parameters,
            preprocessor=preprocessor,
        )
        LOGGER.info(f"model_config_kwargs: {model_config_kwargs}")
        model_config = self.load_hf_config(self.model_id, **model_config_kwargs)

        model = self.load_hf_model(self.model_id, config=model_config)

        self.config = model_config
        self.model = model
        self.preprocessor = preprocessor

        return self

    def _run(
        self,
        data: pd.DataFrame,
        future_data: Optional[pd.DataFrame] = None,
        schema: Optional[ForecastingMetadataInput] = None,
        parameters: Optional[ForecastingParameters] = None,
    ) -> pd.DataFrame:
        # tbd, can this be moved to the HFWrapper?
        # error checking once data available
        if self.preprocessor.freq is None:
            # train to estimate freq if not available
            self.preprocessor.train(data)
            LOGGER.info(f"Data frequency determined: {self.preprocessor.freq}")

        # warn if future data is not provided, but is needed by the model
        if self.preprocessor.exogenous_channel_indices and future_data is None:
            ValueError(
                "Future data should be provided for exogenous columns where the future is known (`control_columns` and `observable_columns`)"
            )

        forecast_pipeline = TimeSeriesForecastingPipeline(
            model=self.model,
            explode_forecasts=True,
            feature_extractor=self.preprocessor,
            add_known_ground_truth=False,
            freq=self.preprocessor.freq,
        )
        forecasts = forecast_pipeline(data, future_time_series=future_data, inverse_scale_outputs=True)

        return forecasts

    def _train(
        self,
    ): ...


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
        return None, ValueError("Config must be provided when loading from a custom module_path")

    try:
        if config is not None:
            model_class = _get_model_class(config, module_path=module_path)
            LOGGER.info(f"Found model class: {model_class.__name__}")
            model = model_class.from_pretrained(model_path, config=config)
            return model, None

        model = AutoModel.from_pretrained(model_path)
        return model, None
    except Exception as e:
        return None, e

    LOGGER.info(f"Found model class: {model.__class__.__name__}")
    return model, None
