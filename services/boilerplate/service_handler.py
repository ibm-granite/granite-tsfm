"""Base serivce handler"""

import datetime
import importlib
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from .inference_payloads import (
    BaseMetadataInput,
    BaseParameters,
    ForecastingMetadataInput,
    ForecastingParameters,
    PredictOutput,
)
from .tsfm_config import TSFMConfig


LOGGER = logging.getLogger(__file__)


class ServiceHandler(ABC):
    """Abstraction to enable serving of various models.

    Args:
        model_id (str): A string identifier for the model.
        model_path (Union[str, Path]): The full path to the model, can be a local path or a HuggingFace Hub path.
        handler_config (TSFMConfig): A handler configuration object.
    """

    def __init__(
        self,
        model_id: str,
        model_path: Union[str, Path],
        handler_config: TSFMConfig,
    ):
        self.model_id = model_id
        self.model_path = model_path
        self.handler_config = handler_config
        self.prepared = False

    @classmethod
    def load(
        cls, model_id: str, model_path: Union[str, Path]
    ) -> Tuple["ServiceHandler", None] | Tuple[None, Exception]:
        """Load the handler_config -- the tsfm service config for this model, returning the proper
        handler to use the model.

        model_path is expected to point to a folder containing the tsfm_config.json file. This can be a local folder
        or with a model on the HuggingFace Hub.

        Args:
            model_id (str): A string identifier for the model.
            model_path (Union[str, Path]): The full path to the model, can be a local path or a HuggingFace Hub path.

        """

        handler_config_path = Path(model_path) if isinstance(model_path, str) else model_path
        # try:
        #     with open((handler_config_path / "tsfm_config.json").as_posix(), "r", encoding="utf-8") as reader:
        #         text = reader.read()
        #     config = json.loads(text)
        # except FileNotFoundError:
        #     LOGGER.info("TSFM Config file not found.")
        #     config = {}

        try:
            config = TSFMConfig.from_pretrained(handler_config_path)
        except (FileNotFoundError, OSError):
            LOGGER.info("TSFM Config file not found.")
            config = TSFMConfig()
        try:
            handler_class = get_service_handler_class(config)
            return handler_class(model_id=model_id, model_path=model_path, handler_config=config), None
        except Exception as e:
            return None, e

    def prepare(
        self,
        data: pd.DataFrame,
        schema: Optional[BaseMetadataInput] = None,
        parameters: Optional[BaseParameters] = None,
        **kwargs,
    ) -> Tuple["ServiceHandler", None] | Tuple[None, Exception]:
        """Prepare the wrapper by loading all the components needed to use the model.

        The actual preparation is done in the `_prepare()` method -- which should be overridden for a
        particular model implementation. This is separate from `load()` above because we may need to know
        certain details about the model (learned from the handler config) before we can effectively load
        and configure the model artifacts.

        Args:
            data (pd.DataFrame): A pandas dataframe containing historical data.
            schema (Optional[BaseMetadataInput], optional): Service request schema. Defaults to None.
            parameters (Optional[BaseParameters], optional): Service requst parameters. Defaults to None.

        Returns:
            Tuple(ServiceHandler, None) or Tuple(None, Exception): In the first case, a tuple containing
                a prepared service handler is returned. The prepared service handler contains all the
                necessary artifacts for performing subsequent inference or training tasks. In the second
                case, the tuple contains an error object.
        """
        try:
            self._prepare(data=data, schema=schema, parameters=parameters, **kwargs)
            self.prepared = True
            return self, None
        except Exception as e:
            return self, e

    @abstractmethod
    def _prepare(
        self,
        data: pd.DataFrame,
        schema: Optional[BaseMetadataInput] = None,
        parameters: Optional[BaseParameters] = None,
        **kwargs,
    ) -> "ServiceHandler":
        """Prepare implementation to be implemented by model owner in derived class

        Args:
            data (pd.DataFrame): A pandas dataframe containing historical data.
            schema (Optional[BaseMetadataInput], optional): Service request schema. Defaults to None.
            parameters (Optional[BaseParameters], optional): Service requst parameters. Defaults to None.

        Returns:
            ServiceHandler: The prepared service handler.
        """
        ...

    def run(
        self,
        data: pd.DataFrame,
        schema: Optional[BaseMetadataInput] = None,
        parameters: Optional[BaseParameters] = None,
        **kwargs,
    ) -> Tuple[PredictOutput, None] | Tuple[None, Exception]:
        """Perform an inference request

        Args:
            data (pd.DataFrame): A pandas dataframe containing historical data.
            schema (Optional[BaseMetadataInput], optional): Service request schema. Defaults to None.
            parameters (Optional[BaseParameters], optional): Service requst parameters. Defaults to None.

        Returns:
            Tuple[PredictOutput, None] | Tuple[None, Exception]: If successful, returns a tuple containing a PredictionOutput
                object as the first element. If unsuccessful, a tuple with an exception as the second element will be returned.
        """

        if not self.prepared:
            return None, RuntimeError("Service wrapper has not yet been prepared; run `model.prepare()` first.")

        try:
            result = self._run(data, schema=schema, parameters=parameters, **kwargs)
            encoded_result = encode_dataframe(result)
            counts = self._calculate_data_point_counts(
                data, output_data=result, schema=schema, parameters=parameters, **kwargs
            )
            return PredictOutput(
                model_id=str(self.model_id),
                created_at=datetime.datetime.now().isoformat(),
                results=[encoded_result],
                **counts,
            ), None

        except Exception as e:
            return None, e

    @abstractmethod
    def _run(
        self,
        data: pd.DataFrame,
        schema: Optional[BaseMetadataInput] = None,
        parameters: Optional[BaseParameters] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Abstract method for run to be implemented by model owner in derived class"""
        ...

    def train(
        self,
    ) -> "ServiceHandler":
        """Perform a fine-tuning request"""
        ...

    @abstractmethod
    def _train(
        self,
    ) -> "ServiceHandler":
        """Abstract method for train to be implemented by model owner in derived class"""
        ...

    @abstractmethod
    def _calculate_data_point_counts(
        self,
        data: pd.DataFrame,
        output_data: Optional[pd.DataFrame] = None,
        schema: Optional[ForecastingMetadataInput] = None,
        parameters: Optional[ForecastingParameters] = None,
    ) -> Dict[str, int]:
        """Abstract method for counting datapoints in input and output implemented by model owner in derived class"""
        ...


class ForecastingServiceHandler(ServiceHandler):
    """Abstraction to enable serving of various models.

    Args:
        model_id (str): A string identifier for the model.
        model_path (Union[str, Path]): The full path to the model, can be a local path or a HuggingFace Hub path.
        handler_config (TSFMConfig): A handler configuration object.
    """

    def __init__(
        self,
        model_id: str,
        model_path: Union[str, Path],
        handler_config: TSFMConfig,
    ):
        super().__init__(model_id, model_path, handler_config)

    def prepare(
        self,
        data: pd.DataFrame,
        future_data: Optional[pd.DataFrame] = None,
        schema: Optional[ForecastingMetadataInput] = None,
        parameters: Optional[ForecastingParameters] = None,
        **kwargs,
    ) -> Tuple["ServiceHandler", None] | Tuple[None, Exception]:
        """Prepare the wrapper by loading all the components needed to use the model.

        The actual preparation is done in the `_prepare()` method -- which should be overridden for a
        particular model implementation. This is separate from `load()` above because we may need to know
        certain details about the model (learned from the handler config) before we can effectively load
        and configure the model artifacts.

        Args:
            data (pd.DataFrame): A pandas dataframe containing historical data.
            future_data (Optional[pd.DataFrame], optional): A pandas dataframe containing future data. Defaults to None.
            schema (Optional[ForecastingMetadataInput], optional): Service request schema. Defaults to None.
            parameters (Optional[ForecastingParameters], optional): Service requst parameters. Defaults to None.

        Returns:
            Tuple(ServiceHandler, None) or Tuple(None, Exception): In the first case, a tuple containing
                a prepared service handler is returned. The prepared service handler contains all the
                necessary artifacts for performing subsequent inference or training tasks. In the second
                case, the tuple contains an error object.
        """

        return super().prepare(data=data, future_data=future_data, schema=schema, parameters=parameters, **kwargs)

    @abstractmethod
    def _prepare(
        self,
        data: pd.DataFrame,
        future_data: Optional[pd.DataFrame] = None,
        schema: Optional[ForecastingMetadataInput] = None,
        parameters: Optional[ForecastingParameters] = None,
        **kwargs,
    ) -> "ForecastingServiceHandler":
        """Prepare implementation to be implemented by model owner in derived class

        Args:
            data (pd.DataFrame): A pandas dataframe containing historical data.
            future_data (Optional[pd.DataFrame], optional): A pandas dataframe containing future data. Defaults to None.
            schema (Optional[ForecastingMetadataInput], optional): Service request schema. Defaults to None.
            parameters (Optional[ForecastingParameters], optional): Service requst parameters. Defaults to None.

        Returns:
            ForecastingServiceHandler: The prepared service handler.
        """
        ...

    def run(
        self,
        data: pd.DataFrame,
        future_data: Optional[pd.DataFrame] = None,
        schema: Optional[ForecastingMetadataInput] = None,
        parameters: Optional[ForecastingParameters] = None,
        **kwargs,
    ) -> Tuple[PredictOutput, None] | Tuple[None, Exception]:
        """Perform an inference request

        Args:
            data (pd.DataFrame): A pandas dataframe containing historical data.
            future_data (Optional[pd.DataFrame], optional): A pandas dataframe containing future data. Defaults to None.
            schema (Optional[ForecastingMetadataInput], optional): Service request schema. Defaults to None.
            parameters (Optional[ForecastingParameters], optional): Service requst parameters. Defaults to None.

        Returns:
            Tuple[PredictOutput, None] | Tuple[None, Exception]: If successful, returns a tuple containing a PredictionOutput
                object as the first element. If unsuccessful, a tuple with an exception as the second element will be returned.
        """

        return super().run(data, future_data=future_data, schema=schema, parameters=parameters, **kwargs)

    @abstractmethod
    def _run(
        self,
        data: pd.DataFrame,
        future_data: Optional[pd.DataFrame] = None,
        schema: Optional[ForecastingMetadataInput] = None,
        parameters: Optional[ForecastingParameters] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Abstract method for run to be implemented by model owner in derived class"""
        ...

    def train(
        self,
    ) -> "ForecastingServiceHandler":
        """Perform a fine-tuning request"""
        ...

    @abstractmethod
    def _train(
        self,
    ) -> "ForecastingServiceHandler":
        """Abstract method for train to be implemented by model owner in derived class"""
        ...

    @abstractmethod
    def _calculate_data_point_counts(
        self,
        data: pd.DataFrame,
        future_data: Optional[pd.DataFrame] = None,
        output_data: Optional[pd.DataFrame] = None,
        schema: Optional[ForecastingMetadataInput] = None,
        parameters: Optional[ForecastingParameters] = None,
    ) -> Dict[str, int]:
        """Abstract method for counting datapoints in input and output implemented by model owner in derived class"""
        ...


def get_service_handler_class(config: TSFMConfig) -> "ServiceHandler":
    if config.service_handler_module_path and config.service_handler_class_name:
        module = importlib.import_module(config.service_handler_module_path)
        my_class = getattr(module, config.service_handler_class_name)

    else:
        # Default to forecasting task
        from .hf_service_handler import ForecastingHuggingFaceHandler

        my_class = ForecastingHuggingFaceHandler

    return my_class


def encode_dataframe(result: pd.DataFrame, timestamp_column: str = None) -> Dict[str, List[Any]]:
    if timestamp_column and pd.api.types.is_datetime64_any_dtype(result[timestamp_column]):
        result[timestamp_column] = result[timestamp_column].apply(lambda x: x.isoformat())
    return result.to_dict(orient="list")
