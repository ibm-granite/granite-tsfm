"""Service handler for HuggingFace models"""

import logging
from typing import Dict, Optional

import pandas as pd
import torch

from tsfm_public import TimeSeriesForecastingPipeline
from tsfm_public.toolkit.time_series_preprocessor import extend_time_series
from tsfm_public.toolkit.util import select_by_index

from .hf_service_handler import ForecastingHuggingFaceHandler, TinyTimeMixerForecastingHandler
from .inference_payloads import ForecastingMetadataInput, ForecastingParameters
from .inference_handler import (
    ForecastingInferenceHandler,
)


LOGGER = logging.getLogger(__file__)


class ForecastingHuggingFaceInferenceHandler(ForecastingHuggingFaceHandler, ForecastingInferenceHandler):
    def _run(
        self,
        data: pd.DataFrame,
        future_data: Optional[pd.DataFrame] = None,
        schema: Optional[ForecastingMetadataInput] = None,
        parameters: Optional[ForecastingParameters] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """_summary_

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

        # warn if future data is not provided, but is needed by the model
        # Remember preprocessor.exogenous_channel_indices are the exogenous for which future data is available
        if self.preprocessor.exogenous_channel_indices and future_data is None:
            raise ValueError(
                "Future data should be provided for exogenous columns where the future is known (`control_columns` and `observable_columns`)"
            )

        if not self.preprocessor.exogenous_channel_indices and future_data is not None:
            raise ValueError("Future data was provided, but the model does not support or require future exogenous.")

        # future_data checks
        if future_data is not None:
            if schema.id_columns:
                data_lengths = future_data.groupby(schema.id_columns)[schema.id_columns].apply(len)
                fd_min_len_index = data_lengths.argmin()
                fd_min_data_length = data_lengths.iloc[fd_min_len_index]
                fd_max_data_length = data_lengths.max()
            else:
                fd_min_data_length = fd_max_data_length = len(future_data)
            LOGGER.info(
                f"Future Data length recieved {len(future_data)}, minimum series length: {fd_min_data_length}, maximum series length: {fd_max_data_length}"
            )

            # if data is too short, raise error
            prediction_length = getattr(self.config, "prediction_filter_length", None)
            has_prediction_filter = prediction_length is not None

            model_prediction_length = self.config.prediction_length

            prediction_length = prediction_length if prediction_length is not None else model_prediction_length
            if fd_min_data_length < prediction_length:
                err_str = (
                    "Future data should have time series of length that is at least the specified prediction length. "
                )
                if schema.id_columns:
                    err_str += f"Received {fd_min_data_length} time points for id {data_lengths.index[fd_min_len_index]}, but expected {prediction_length} time points."
                else:
                    err_str += (
                        f"Received {fd_min_data_length} time points, but expected {prediction_length} time points."
                    )
                raise ValueError(err_str)

            # if data exceeds prediction filter length, truncate
            if fd_max_data_length > model_prediction_length:
                LOGGER.info(f"Truncating future series lengths to {model_prediction_length}")
                future_data = select_by_index(
                    future_data, id_columns=schema.id_columns, end_index=model_prediction_length
                )

            # if provided data is greater than prediction_filter_length, but less than model_prediction_length we extend
            if has_prediction_filter and fd_min_data_length < model_prediction_length:
                LOGGER.info(f"Extending time series to match model prediction length: {model_prediction_length}")
                future_data = extend_time_series(
                    time_series=future_data,
                    freq=self.preprocessor.freq,
                    timestamp_column=schema.timestamp_column,
                    grouping_columns=schema.id_columns,
                    total_periods=model_prediction_length,
                )

        device = "cpu" if not torch.cuda.is_available() else "cuda"
        forecast_pipeline = TimeSeriesForecastingPipeline(
            model=self.model,
            explode_forecasts=True,
            feature_extractor=self.preprocessor,
            add_known_ground_truth=False,
            freq=self.preprocessor.freq,
            device=device,
        )
        forecasts = forecast_pipeline(data, future_time_series=future_data, inverse_scale_outputs=True)

        return forecasts

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

        input_ts_columns = sum(
            [
                len(c)
                for c in [
                    schema.target_columns,
                    schema.conditional_columns,
                    schema.control_columns,
                    schema.observable_columns,
                ]
            ]
        )
        input_ts_columns = input_ts_columns if input_ts_columns != 0 else data.shape[1] - len(schema.id_columns) - 1
        input_static_columns = len(schema.static_categorical_columns)
        num_target_columns = (
            len(schema.target_columns) if schema.target_columns != [] else data.shape[1] - len(schema.id_columns) - 1
        )
        unique_ts = len(data.drop_duplicates(subset=schema.id_columns)) if schema.id_columns else 1
        has_future_data = future_data is not None

        # we don't count the static columns in the future data
        # we only count future data which falls within forecast horizon "causal assumption"
        # note that output_data.shape[0] = unique_ts * prediction_length
        future_data_points = (input_ts_columns - num_target_columns) * output_data.shape[0] if has_future_data else 0

        counts = {
            "input_data_points": input_ts_columns * data.shape[0]
            + input_static_columns * unique_ts
            + future_data_points,
            "output_data_points": output_data.shape[0] * num_target_columns,
        }
        LOGGER.info(f"Data point counts: {counts}")
        return counts


class TinyTimeMixerForecastingInferenceHandler(
    TinyTimeMixerForecastingHandler, ForecastingHuggingFaceInferenceHandler
): ...
