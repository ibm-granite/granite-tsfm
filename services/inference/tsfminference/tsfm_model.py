import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import pandas as pd
from transformers import PretrainedConfig, PreTrainedModel

from tsfm_public import TimeSeriesPreprocessor

from .hfutil import load_config, load_model
from .inference import decode_data
from .inference_payloads import (
    ForecastingInferenceInput,
    ForecastingMetadataInput,
    ForecastingParameters,
    PredictOutput,
)


LOGGER = logging.getLogger(__file__)


class TSFMWrapperBase(ABC):
    def __init__(self, model: Any):
        self.model = model

    def run(
        self,
        input_payload: ForecastingInferenceInput,
    ) -> Union[Optional[PredictOutput], Optional[Exception]]:
        # to be implemented using service implementation

        schema_params = input_payload.schema.model_dump()

        data = decode_data(input_payload.data, schema_params)
        future_data = decode_data(input_payload.future_data, schema_params)

        try:
            return self._run(
                data,
                future_data=future_data,
                schema=input_payload.schema,
                parameters=input_payload.parameters,
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
    ) -> Union[Optional[PredictOutput], Optional[Exception]]:
        """Abstract method to be implemented by model owner"""
        ...

    @classmethod
    def load_config(cls, tsfm_config_path: str):
        """Load the configuration of the service wrapper for tsfmservices


        tsfm_config_path is expected to point to a folder containing the tsfm_config.json file

        """
        with open(tsfm_config_path, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)

    @classmethod
    def load(
        cls,
        model_id: str,
        schema: Optional[ForecastingMetadataInput] = None,
        parameters: Optional[ForecastingParameters] = None,
    ) -> Union[Optional["TSFMWrapperBase"], Optional[Exception]]:
        try:
            return cls._load(model_id, schema=schema, parameters=parameters), None
        except Exception as e:
            return None, e

    @classmethod
    @abstractmethod
    def _load(
        cls,
        model_id: str,
        schema: Optional[ForecastingMetadataInput] = None,
        parameters: Optional[ForecastingParameters] = None,
    ) -> "TSFMWrapperBase": ...


class HFWrapper(TSFMWrapperBase):
    def __init__(self, model: PreTrainedModel, preprocessor: TimeSeriesPreprocessor):
        super().__init__(model=model)
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
    ) -> Union[Optional[PredictOutput], Optional[Exception]]:
        # error checking once data available
        if self.preprocess.freq is None:
            # train to estimate freq if not available
            self.preprocessor.train(data)
            LOGGER.info(f"Data frequency determined: {self.preprocessor.freq}")

        # warn if future data is not provided, but is needed by the model
        if self.preprocessor.exogenous_channel_indices and future_data is None:
            ValueError(
                "Future data should be provided for exogenous columns where the future is known (`control_columns` and `observable_columns`)"
            )

    def _load(
        cls,
        model_path: str,
        schema: Optional[ForecastingMetadataInput] = None,
        parameters: Optional[ForecastingParameters] = None,
    ) -> "TinyTimeMixerModel":
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
        config = cls.load_config(model_path, **model_config_kwargs)

        model = cls.load_model(model_path, config=config)

        return TinyTimeMixerModel(model=model, time_series_preprocessor=preprocessor)
