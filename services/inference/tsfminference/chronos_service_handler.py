"""Service handler for Chronos"""

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline

from .inference_payloads import (
    ForecastingMetadataInput,
    ForecastingParameters,
)
from .service_handler import ForecastingServiceHandler
from .tsfm_config import TSFMConfig


LOGGER = logging.getLogger(__file__)


class ChronosForecastingHandler(ForecastingServiceHandler):
    """Handler for Chronos model family

    Supports chronos-t5-small at this point.

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
        super().__init__(
            model_id=model_id, model_path=model_path, handler_config=handler_config
        )

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

        model = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-tiny",
        )
        self.model = model
        self.config = model.model.model.config
        self.chronos_config = self.config.chronos_config

        return self

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

        target_columns = schema.target_columns or data.columns.tolist()
        prediction_length = (
            parameters.prediction_length or self.chronos_config["prediction_length"]
        )

        num_samples = self.chronos_config["num_samples"]
        temperature = self.chronos_config["temperature"]
        top_k = self.chronos_config["top_k"]
        top_p = self.chronos_config["top_p"]

        context = torch.tensor(data[target_columns].values).transpose(1, 0)
        LOGGER.info("computing chronos forecasts.")
        forecasts = self.model.predict(
            context,
            prediction_length=prediction_length,
            num_samples=num_samples,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            limit_prediction_length=False,
        )
        median_forecast_arr = []
        for i in range(len(target_columns)):
            median_forecast_arr.append(
                np.quantile(forecasts[i].numpy(), [0.5], axis=0).flatten()
            )

        result = pd.DataFrame(
            np.array(median_forecast_arr).transpose(), columns=target_columns
        )
        return result

    def _train(
        self,
    ) -> "ChronosForecastingHandler": ...
