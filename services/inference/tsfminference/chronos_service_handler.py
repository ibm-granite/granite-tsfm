"""Service handler for Chronos"""

import copy
import importlib
import logging
from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd
import torch

from tsfm_public import TimeSeriesPreprocessor
from tsfm_public.toolkit.time_series_preprocessor import (
    create_timestamps,
)

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

        # model specific parameters
        preprocessor_params["num_samples"] = getattr(parameters, "num_samples", None)
        preprocessor_params["temperature"] = getattr(parameters, "temperature", None)
        preprocessor_params["top_k"] = getattr(parameters, "top_k", None)
        preprocessor_params["top_p"] = getattr(parameters, "top_p", None)

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

        predictions = None
        if self.preprocessor.exogenous_channel_indices or future_data is not None:
            raise ValueError("Chronos does not support or require future exogenous.")

        target_columns = self.preprocessor.target_columns
        prediction_length = self.preprocessor.prediction_length
        timestamp_column = self.preprocessor.timestamp_column
        id_columns = self.preprocessor.id_columns

        additional_params = {}
        if "num_samples" in self.chronos_config:  # chronos t5 family
            additional_params["num_samples"] = self.preprocessor.num_samples or self.chronos_config["num_samples"]
            additional_params["temperature"] = self.preprocessor.temperature or self.chronos_config["temperature"]
            additional_params["top_k"] = self.preprocessor.top_k or self.chronos_config["top_k"]
            additional_params["top_p"] = self.preprocessor.top_p or self.chronos_config["top_p"]

        LOGGER.info("model specific params: {}".format(additional_params))

        scoped_cols = [timestamp_column] + id_columns + target_columns

        LOGGER.info("computing chronos forecasts.")

        if not id_columns:
            LOGGER.info("id columns are not provided, proceeding without groups.")
            context = torch.tensor(data[target_columns].values).transpose(1, 0)
            forecasts = self.model.predict(
                context,
                prediction_length=prediction_length,
                limit_prediction_length=False,
                **additional_params,
            )
            median_forecasts = torch.quantile(forecasts, 0.5, dim=1).transpose(1, 0)
            result = pd.DataFrame(median_forecasts, columns=target_columns)
            if timestamp_column:
                result[timestamp_column] = create_timestamps(
                    data[timestamp_column].iloc[-1],
                    freq=self.preprocessor.freq,
                    periods=result.shape[0],
                )
            predictions = result
        else:  # create groups
            LOGGER.info("using id columns {} to create groups.".format(id_columns))
            accumulator = []
            for grp, batch in data[scoped_cols].groupby(id_columns):
                context = torch.tensor(batch[target_columns].values).transpose(1, 0)
                forecasts = self.model.predict(
                    context,
                    prediction_length=prediction_length,
                    limit_prediction_length=False,
                    **additional_params,
                )
                median_forecasts = torch.quantile(forecasts, 0.5, dim=1).transpose(1, 0)
                result = pd.DataFrame(median_forecasts, columns=target_columns)
                if timestamp_column:
                    result[timestamp_column] = create_timestamps(
                        batch[timestamp_column].iloc[-1],
                        freq=self.preprocessor.freq,
                        periods=result.shape[0],
                    )
                if (id_columns is not None) and id_columns:
                    for k, id_col in enumerate(id_columns):
                        result[id_col] = grp[k]
                accumulator.append(result)

            predictions = pd.concat(accumulator, ignore_index=True)

        return predictions[scoped_cols]

    def _train(
        self,
    ) -> "ChronosForecastingHandler": ...
