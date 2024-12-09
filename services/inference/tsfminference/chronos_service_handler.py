"""Service handler for Chronos"""

import copy
import importlib
import logging
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import torch

from tsfm_public import TimeSeriesPreprocessor
from tsfm_public.toolkit.time_series_preprocessor import extend_time_series

from .inference_payloads import (
    ForecastingMetadataInput,
    ForecastingParameters,
)
from .service_handler import ForecastingServiceHandler
from .tsfm_config import TSFMConfig


LOGGER = logging.getLogger(__file__)


class ChronosForecastingHandler(ForecastingServiceHandler):
    """Handler for Chronos model family

    Supports chronos-t5-tiny at this point.

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

    def _prepare(
        self,
        data: pd.DataFrame,
        future_data: Optional[pd.DataFrame] = None,
        schema: Optional[ForecastingMetadataInput] = None,
        parameters: Optional[ForecastingParameters] = None,
        **kwargs,
    ) -> "ChronosForecastingHandler":
        """Implementation of _prepare for Chronos family of models. This method loads the model using chronos apis.

        Args:
            data (pd.DataFrame): A pandas dataframe containing historical data.
            future_data (Optional[pd.DataFrame], optional): A pandas dataframe containing future data. Defaults to None.
            schema (Optional[ForecastingMetadataInput], optional): Schema information from the original inference
                request. Includes information about columns and their role. Defaults to None.
            parameters (Optional[ForecastingParameters], optional): Parameters from the original inference
                request. Defaults to None.

        Returns:
            ChronosForecastingHandler: The updated service handler object.
        """

        # load model class
        try:
            mod = importlib.import_module(self.handler_config.module_path)
        except ModuleNotFoundError as exc:
            raise AttributeError("Could not load module '{module_path}'.") from exc

        model_class = getattr(mod, self.handler_config.model_class_name)
        model = model_class.from_pretrained(self.model_path)

        self.model = model
        if hasattr(self.model.model, "model"):  # chronos t5 family
            self.config = model.model.model.config
        else:  # chronos bolt family
            self.config = model.model.config

        self.chronos_config = self.config.chronos_config

        preprocessor_params = copy.deepcopy(schema.model_dump())
        preprocessor_params["prediction_length"] = (
            parameters.prediction_length or self.chronos_config["prediction_length"]
        )

        LOGGER.info("initializing TSFM TimeSeriesPreprocessor")
        preprocessor = TimeSeriesPreprocessor(
            **preprocessor_params,
            scaling=False,
            encode_categorical=False,
        )
        preprocessor.train(data)
        LOGGER.info(f"Data frequency determined: {preprocessor.freq}")
        self.preprocessor = preprocessor

        return self

    def _calculate_data_point_counts(
        self,
        data: pd.DataFrame,
        future_data: Optional[pd.DataFrame] = None,
        output_data: Optional[pd.DataFrame] = None,
        schema: Optional[ForecastingMetadataInput] = None,
        parameters: Optional[ForecastingParameters] = None,
    ) -> Dict[str, int]:
        """Implementation for counting datapoints in input and output

        Assumes data has been truncated
        Future data may not be truncated
        """
        return {}  # to be implemented

    def _run(
        self,
        data: pd.DataFrame,
        future_data: Optional[pd.DataFrame] = None,
        schema: Optional[ForecastingMetadataInput] = None,
        parameters: Optional[ForecastingParameters] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Inference request for Chronos family of models.

        Args:
            data (pd.DataFrame): Input historical time series data.
            future_data (Optional[pd.DataFrame], optional): Input future time series data, useful for
                passing future exogenous if known. Defaults to None.
            schema (Optional[ForecastingMetadataInput], optional): Schema information from the original inference
                request. Includes information about columns and their role. Defaults to None.
            parameters (Optional[ForecastingParameters], optional): Parameters from the original inference
                request. Defaults to None.

        Returns:
            pd.DataFrame: The forecasts produced by the model.
        """

        if self.preprocessor.exogenous_channel_indices or future_data is not None:
            raise ValueError("Chronos does not support or require future exogenous.")

        target_columns = self.preprocessor.target_columns
        prediction_length = self.preprocessor.prediction_length

        additional_params = {}
        if "num_samples" in self.chronos_config:  # chronos t5 family
            additional_params["num_samples"] = self.chronos_config["num_samples"]
            additional_params["temperature"] = self.chronos_config["temperature"]
            additional_params["top_k"] = self.chronos_config["top_k"]
            additional_params["top_p"] = self.chronos_config["top_p"]

        context = torch.tensor(data[target_columns].values).transpose(1, 0)
        LOGGER.info("computing chronos forecasts.")
        forecasts = self.model.predict(
            context,
            prediction_length=prediction_length,
            limit_prediction_length=False,
            **additional_params,
        )
        median_forecast_arr = []
        for i in range(len(target_columns)):
            median_forecast_arr.append(np.quantile(forecasts[i].numpy(), [0.5], axis=0).flatten())

        result = pd.DataFrame(np.array(median_forecast_arr).transpose(), columns=target_columns)
        LOGGER.info("extend the time series.")
        time_series = extend_time_series(
            time_series=data,
            freq=self.preprocessor.freq,
            timestamp_column=schema.timestamp_column,
            grouping_columns=schema.id_columns,
            periods=prediction_length,
        )
        # append time stamp column to the result
        result[schema.timestamp_column] = (
            time_series[schema.timestamp_column].tail(result.shape[0]).reset_index(drop=True)
        )

        return result

    def _train(
        self,
    ) -> "ChronosForecastingHandler": ...
