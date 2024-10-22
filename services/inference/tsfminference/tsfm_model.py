from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import pandas as pd

from tsfm_public import TinyTimeMixerForPrediction

from .inference import decode_data
from .inference_payloads import (
    ForecastingInferenceInput,
    ForecastingMetadataInput,
    ForecastingParameters,
    PredictOutput,
)


class TSFMModel(ABC):
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

        return self._run(
            data,
            future_data=future_data,
            schema=input_payload.schema,
            parameters=input_payload.parameters,
        )

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
    def load(cls, model_id: str) -> "TSFMModel":
        return cls._load(model_id)

    @classmethod
    @abstractmethod
    def _load(cls, model_id: str) -> "TSFMModel": ...


class TinyTimeMixerModel(TSFMModel):
    def _run(
        self,
        data: pd.DataFrame,
        future_data: Optional[pd.DataFrame] = None,
        schema: Optional[ForecastingMetadataInput] = None,
        parameters: Optional[ForecastingParameters] = None,
    ) -> Union[Optional[PredictOutput], Optional[Exception]]: ...

    def _load(cls, model_id: str) -> "TinyTimeMixerModel":


        # assume model_id is a path to a file, or otherwise loadable uri

        # load preprocessor

        # create appropriate config

        model = TinyTimeMixerForPrediction.from_pretrained(model_id)

        tsfm_model = TinyTimeMixerModel(model)


        tsfm_model.preprocessor = 

        return 
