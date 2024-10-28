"""Base serivce handlers"""

import datetime
import importlib
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
from transformers import PretrainedConfig, PreTrainedModel

from tsfm_public import TimeSeriesPreprocessor

from .hfutil import load_config, load_model, register_config
from .inference_payloads import (
    ForecastingMetadataInput,
    ForecastingParameters,
    PredictOutput,
)


LOGGER = logging.getLogger(__file__)


class ServiceHandler(ABC):
    def __init__(
        self,
        model_id: Union[str, Path],
        tsfm_config: Dict[str, Any],
    ):
        """_summary_

        Args:
            tsfm_config (Dict[str, Any]): TSFM Service configuration
        """
        self.model_id = model_id
        self.tsfm_config = tsfm_config
        self.prepared = False

    @classmethod
    def load(cls, model_id: Union[str, Path]) -> Union[Optional["ServiceHandler"], Optional[Exception]]:
        """Load the tsfm_config  -- the tsfm service config for this model

        tsfm_config_path is expected to point to a folder containing the tsfm_config.json file
        to do: can we make this work with HF Hub?

        tsfm_config.json contents:


        module_path: tsfm_public
        model_type: tinytimemixer
        model_config_name: TinyTimeMixerConfig
        model_class_name: TinyTimeMixerForPrediction

        # do we need a list of capabilities, like:
        # supports variable prediction length
        # supports variable context length, etc.?

        """

        tsfm_config_path = Path(model_id) if isinstance(model_id, str) else model_id

        try:
            with open((tsfm_config_path / "tsfm_config.json").as_posix(), "r", encoding="utf-8") as reader:
                text = reader.read()
            config = json.loads(text)
        except FileNotFoundError:
            LOGGER.info("TSFM Config file not found.")
            config = {}

        try:
            wrapper_class = get_service_model_class(config)
            return wrapper_class(model_id=model_id, tsfm_config=config), None
        except Exception as e:
            return None, e

    def prepare(
        self,
        schema: Optional[ForecastingMetadataInput] = None,
        parameters: Optional[ForecastingParameters] = None,
    ) -> Union[Optional["ServiceHandler"], Optional[Exception]]:
        """Prepare the wrapper by loading all the components needed to use the model."""

        try:
            self._prepare(schema=schema, parameters=parameters)
            self.prepared = True
            return self, None
        except Exception as e:
            return self, e

    @abstractmethod
    def _prepare(
        self,
        schema: Optional[ForecastingMetadataInput] = None,
        parameters: Optional[ForecastingParameters] = None,
    ) -> "ServiceHandler":
        """Prepare implementation to be implemented in derived class"""
        ...

    def run(
        self,
        data: pd.DataFrame,
        future_data: Optional[pd.DataFrame] = None,
        schema: Optional[ForecastingMetadataInput] = None,
        parameters: Optional[ForecastingParameters] = None,
    ) -> Union[Optional[PredictOutput], Optional[Exception]]:
        """Perform an inference request on a loaded model"""

        if not self.prepared:
            return None, RuntimeError("Service wrapper has not yet been prepared; run `model.prepare()` first.")

        try:
            result = self._run(
                data,
                future_data=future_data,
                schema=schema,
                parameters=parameters,
            )
            return PredictOutput(
                model_id=str(self.model_id),
                created_at=datetime.datetime.now().isoformat(),
                results=[result.to_dict(orient="list")],
            ), None

        except Exception as e:
            return None, e

    @abstractmethod
    def _run(
        self,
        data: pd.DataFrame,
        future_data: Optional[pd.DataFrame] = None,
        schema: Optional[ForecastingMetadataInput] = None,
        parameters: Optional[ForecastingParameters] = None,
    ) -> pd.DataFrame:
        """Abstract method to be implemented by model owner"""
        ...

    def train(
        self,
    ):
        """comming soon"""
        ...

    @abstractmethod
    def _train(
        self,
    ):
        """Abstract method to be implemented by model owner"""
        ...


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
        # load model
        # TO DO: needs cleanup
        model, e = load_model(
            model_path,
            config=config,
            module_path=self.tsfm_config.get("module_path", None),
        )

        if e is not None:
            raise (e)
        LOGGER.info("Successfully loaded model")
        return model


def get_service_model_class(config: Dict[str, Any]):
    if "service_handler_module_path" in config and "service_handler_class_name" in config:
        module = importlib.import_module(config["service_handler_module_path"])
        my_class = getattr(module, config["service_handler_class_name"])

    else:
        from .tsfm_service_handler import DefaultHandler

        my_class = DefaultHandler

    return my_class
