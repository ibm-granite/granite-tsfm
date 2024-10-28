import copy
import datetime
import importlib
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import pandas as pd
from transformers import PretrainedConfig, PreTrainedModel

from tsfm_public import TimeSeriesForecastingPipeline, TimeSeriesPreprocessor
from tsfm_public.toolkit.util import select_by_index

from .hfutil import load_config, load_model
from .inference import decode_data
from .inference_payloads import (
    ForecastingInferenceInput,
    ForecastingMetadataInput,
    ForecastingParameters,
    PredictOutput,
)


LOGGER = logging.getLogger(__file__)

# tsfm service wrapper


class TSFMWrapperBase(ABC):
    def __init__(self, config: Dict[str, Any], model: Any):
        self.model = model
        self.config = config

    @classmethod
    def load(
        cls,
        model_id: str,
        schema: Optional[ForecastingMetadataInput] = None,
        parameters: Optional[ForecastingParameters] = None,
    ) -> Union[Optional["TSFMWrapperBase"], Optional[Exception]]:
        """Load all the components needed to use the model."""

        config = cls.load_config(model_id)
        wrapper_class = get_service_model_class(config)
        # service_class.load(service_wrapper_config)

        try:
            return wrapper_class._load(model_id, config=config, schema=schema, parameters=parameters), None
        except Exception as e:
            return None, e

    @classmethod
    @abstractmethod
    def _load(
        cls,
        model_id: str,
        config: Dict[str, Any] = {},
        schema: Optional[ForecastingMetadataInput] = None,
        parameters: Optional[ForecastingParameters] = None,
    ) -> "TSFMWrapperBase":
        """Load implementation to be implemented in derived class"""
        ...

    @classmethod
    def load_config(cls, tsfm_config_path: str):
        """Load the configuration of the service wrapper for tsfmservices


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
        with open(tsfm_config_path, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)

    def run(
        self,
        input_payload: ForecastingInferenceInput,
    ) -> Union[Optional[PredictOutput], Optional[Exception]]:
        """Perform an inference request on a loaded model"""

        schema = input_payload.schema.model_dump()

        data = decode_data(input_payload.data, schema)
        future_data = decode_data(input_payload.future_data, schema)

        # collect and check underlying time series lengths
        if self.config.minimum_context_length:
            if schema["id_columns"]:
                data_lengths = data.groupby(schema["id_columns"]).apply(len)
                min_len_index = data_lengths.argmin()
                max_len_index = data_lengths.argmax()
                min_data_length = data_lengths.iloc[min_len_index]
                max_data_length = data_lengths.iloc[max_len_index]
            else:
                data_length = len(data)
            LOGGER.info(f"Data length recieved {len(data)}, minimum series length: {data_length}")

            if data_length < self.config.minimum_context_length:
                err_str = "Data should have time series of length that is at least the required model context length. "
                if schema.id_columns:
                    err_str += f"Received {min_data_length} time points for id {data_lengths.index[min_len_index]}, but model requires {self.coonfig.minimum_context_length} time points"
                else:
                    err_str += f"Received {min_data_length} time points, but model requires {self.coonfig.minimum_context_length} time points"

                return None, ValueError(err_str)

        # truncate data length
        if self.config.max_context_length:
            if max_data_length > self.config.max_context_length:
                data = select_by_index(
                    data, id_columns=schema["id_columns"], start_index=-self.config.max_context_length
                )

        try:
            result = self._run(
                data,
                future_data=future_data,
                schema=input_payload.schema,
                parameters=input_payload.parameters,
            )
            return PredictOutput(
                model_id=input_payload.model_id,
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


class HFWrapper(TSFMWrapperBase):
    def __init__(self, config: Dict[str, Any], model: PreTrainedModel, preprocessor: TimeSeriesPreprocessor):
        super().__init__(config=config, model=model)
        self.preprocessor = preprocessor

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
        model = load_model(
            model_path,
            config=config,
            module_path=self.model_to_module_map.get(config.__class__.__name__, None),
        )

        LOGGER.info("Successfully loaded model")
        return model


class TinyTimeMixerModel(HFWrapper):
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

    def _load(
        cls,
        model_path: str,
        config: Dict[str, any] = {},
        schema: Optional[ForecastingMetadataInput] = None,
        parameters: Optional[ForecastingParameters] = None,
    ) -> "TinyTimeMixerModel":
        # to do: use config parameters below
        # issue: _load may need to know data length to set parameters upon model load (multst)

        preprocessor_params = copy.deepcopy(schema.model_dump())
        preprocessor_params["prediction_length"] = parameters.prediction_length

        LOGGER.info(f"Preprocessor params: {preprocessor_params}")

        preprocessor = cls.load_preprocessor(model_path)

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
        model_config = cls.load_hf_config(model_path, **model_config_kwargs)

        model = cls.load_hf_model(model_path, config=model_config)

        return TinyTimeMixerModel(config=config, model=model, preprocessor=preprocessor)


def get_service_model_class(config: Dict[str, Any]):
    module = importlib.import_module(config["service_wraper_module_path"])
    my_class = getattr(module, config["service_class_name"])

    return my_class
