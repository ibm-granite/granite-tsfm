"""Base serivce handler"""

import logging
from abc import abstractmethod
from pathlib import Path
from typing import Tuple, Union

import pandas as pd

from .ftpayloads import TuneOutput
from .inference_payloads import (
    BaseMetadataInput,
    BaseParameters,
    ForecastingMetadataInput,
    ForecastingParameters,
)
from .service_handler import ForecastingServiceHandler, HandlerFunction, ServiceHandlerBase


LOGGER = logging.getLogger(__file__)


class TuningHandler(ServiceHandlerBase):
    @classmethod
    def load(
        cls, model_id: str, model_path: Union[str, Path]
    ) -> Tuple["TuningHandler", None] | Tuple[None, Exception]:
        """Load the handler_config -- the tsfm service config for this model, returning the proper
        handler to use the model.

        model_path is expected to point to a folder containing the tsfm_config.json file. This can be a local folder
        or with a model on the HuggingFace Hub.

        Args:
            model_id (str): A string identifier for the model.
            model_path (Union[str, Path]): The full path to the model, can be a local path or a HuggingFace Hub path.
            handler_function (str): The type of handler, currently supported handlers are defined in the HandlerFunction
                enum.

        """

        return super().load(model_id, model_path, handler_function=HandlerFunction.TUNING.value)

    def train(
        self,
        data: pd.DataFrame,
        schema: BaseMetadataInput,
        parameters: BaseParameters,
        tuned_model_name: str,
        tmp_dir: Path,
    ) -> Tuple[str, None] | Tuple[None, Exception]:
        """Perform a fine-tuning request"""
        if not self.prepared:
            return None, RuntimeError("Service wrapper has not yet been prepared; run `handler.prepare()` first.")

        try:
            result = self._train(
                data, schema=schema, parameters=parameters, tuned_model_name=tuned_model_name, tmp_dir=tmp_dir
            )

            # counts = self._calculate_data_point_counts(
            #     data, output_data=result, schema=schema, parameters=parameters, **kwargs
            # )
            # Does TuneOuput need some info about the request -- for billing purposes
            # return TuneOutput(training_ref=result), None
            return result, None

        except Exception as e:
            return None, e


class ForecastingTuningHandler(ForecastingServiceHandler, TuningHandler):
    def train(
        self,
        data: pd.DataFrame,
        schema: ForecastingMetadataInput,
        parameters: ForecastingParameters,
        tuned_model_name: str,
        tmp_dir: Path,
    ) -> "TuneOutput":
        """Perform a fine-tuning request"""
        return super().train(
            data, schema=schema, parameters=parameters, tuned_model_name=tuned_model_name, tmp_dir=tmp_dir
        )

    @abstractmethod
    def _train(
        self,
        data: pd.DataFrame,
        schema: ForecastingMetadataInput,
        parameters: ForecastingParameters,
        tuned_model_name: str,
        tmp_dir: Path,
    ) -> str:
        """Abstract method for train to be implemented by model owner in derived class"""
        ...
