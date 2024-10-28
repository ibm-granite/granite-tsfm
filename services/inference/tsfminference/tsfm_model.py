import copy
import datetime
import importlib
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
from transformers import PretrainedConfig, PreTrainedModel

from tsfm_public import TimeSeriesForecastingPipeline, TimeSeriesPreprocessor

from .hfutil import load_config, load_model, register_config
from .inference_payloads import (
    ForecastingMetadataInput,
    ForecastingParameters,
    PredictOutput,
)


LOGGER = logging.getLogger(__file__)

# tsfm service wrapper


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

        with open((tsfm_config_path / "tsfm_config.json").as_posix(), "r", encoding="utf-8") as reader:
            text = reader.read()
        config = json.loads(text)

        wrapper_class = get_service_model_class(config)
        try:
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
            module_path=self.tsfm_config["module_path"],
        )

        if e is not None:
            raise (e)
        LOGGER.info("Successfully loaded model")
        return model


class TinyTimeMixerHandler(HuggingFaceHandler):
    def _prepare(
        self,
        schema: Optional[ForecastingMetadataInput] = None,
        parameters: Optional[ForecastingParameters] = None,
    ) -> "TinyTimeMixerHandler":
        # to do: use config parameters below
        # issue: _load may need to know data length to set parameters upon model load (multst)

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

        model_config_kwargs = {
            "prediction_filter_length": parameters.prediction_length,
            "num_input_channels": preprocessor.num_input_channels,
        }
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


def get_service_model_class(config: Dict[str, Any]):
    module = importlib.import_module(config["service_wrapper_module_path"])
    my_class = getattr(module, config["service_wrapper_class_name"])

    return my_class
