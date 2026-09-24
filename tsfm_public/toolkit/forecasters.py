"""
High level classes for making forecasts. Motivated by need to support ensembles.
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

from tsfm_public import (
    FlowStateForPrediction,
    PatchTSTFMForPrediction,
    TimeSeriesForecastingPipeline,
    TimeSeriesPreprocessor,
    TinyTimeMixerForDecomposedPrediction,
    TinyTimeMixerForPrediction,
    get_model,
)


logger = logging.getLogger(__name__)

DEFAULT_QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class Forecaster(ABC):
    """Abstract class providing a consistent interface to forecasting capabilities.
    Also enables ensemble foreasting.
    """

    @abstractmethod
    def __init__(self):
        """Initialize with info that should be loaded once, such as the model checkpoint and device type."""

    @abstractmethod
    def __call__(self):
        """Returns forecast in default format"""

    @abstractmethod
    def forecast_for_ensemble(self):
        """Returns the forecast info as ndarray needed for ensembling"""


@dataclass
class ForecastResult:
    """Result model for forecasting operations.

    Fields:
        predicted: Point predictions - mean (TTM trained with MSE) or median (FlowState/PatchTST-fm/TTM trained with MAE)
        actuals: Observations aligned with the forecast horizon (if available, otherwise None)
        predicted_quantiles: Quantile predictions (if quantile_levels specified), with quantile_levels in metadata
        cutoff_dates: End of context timestamps for each sample
        metadata: Model configuration, quantile levels, and ensemble aggregation method if applicable
    """

    success: bool
    message: str
    predicted: Optional[List[List[List[float]]]] = None
    # Point predictions: mean (TTM with MSE loss) or median (FlowState/PatchTST-fm)
    # Shape: (n_samples, prediction_length, n_targets)

    actuals: Optional[List[List[List[float]]]] = None
    # Ground truth values, shape: (n_samples, prediction_length, n_targets)

    predicted_quantiles: Optional[List[List[List[List[float]]]]] = None
    # Quantile predictions, shape: (n_samples, prediction_length, n_targets, n_quantiles)
    # Quantile levels specified in metadata["quantile_levels"]

    cutoff_dates: Optional[List[str]] = None
    # End of context timestamps for each sample, shape: (n_samples,)

    metadata: dict = field(default_factory=dict)



class DataFramePipelineForecaster(Forecaster):
    """Forecasts from  PreTrained model using TimeSeriesForecastingPipeline.

    The model is loaded once at construction time. Use:
      - __call__ to get the raw forecast DataFrame
      - forecast_in_forecast_result for a structured ForecastResult
      - forecast_for_ensemble for the quantile array used by ensemble aggregation
    """

    def __init__(
        self,
        model,
        device: str = None,
    ):
        """Load model from checkpoint and set device.

        Args:
            model_checkpoint: HuggingFace model checkpoint path.
            device: Inference device. If None, uses CUDA if available, else CPU.
        """
        self.model = model
        self.device = device if device is not None else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def __call__(
        self,
        data: pd.DataFrame,
        timestamp_column: str,
        target_columns: List[str],
        prediction_length: int,
        context_length: Optional[int] = None,
        id_columns: List[str] = [],
        quantile_levels: List[float] = DEFAULT_QUANTILE_LEVELS,
        batch_size: int = 16,
    ) -> pd.DataFrame:
        """Run inference and return the raw forecast DataFrame.

        Args:
            data: DataFrame containing time series data.
            timestamp_column: Name of the timestamp column.
            target_columns: List of target column names to forecast.
            prediction_length: Number of future timesteps to forecast.
            context_length: Number of historical timesteps to use as context.
                If None, defaults to the model's own context length.
            id_columns: Columns identifying distinct time series. Default: [].
            quantile_levels: Quantile levels for probabilistic forecasting.
                Default: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9].
            batch_size: Inference batch size. Default: 16.

        Returns:
            pd.DataFrame with forecast results from TimeSeriesForecastingPipeline.
        """
        context_length = context_length if context_length is not None else self.model.config.context_length

        assert not data.empty, "Input data is empty"
        assert timestamp_column in data.columns, \
            f"Timestamp column '{timestamp_column}' not found in data"
        for col in target_columns:
            assert col in data.columns, \
                f"Target column '{col}' not found in data"

        fpipe = TimeSeriesForecastingPipeline(
            model=self.model,
            id_columns=id_columns,
            timestamp_column=timestamp_column,
            target_columns=target_columns,
            max_context_length=self.model.config.context_length,
            context_length=context_length,
            prediction_length=prediction_length,
            batch_size=batch_size,
            impute_method=None,
            device=self.device,
            quantile_levels=quantile_levels,
        )

        return fpipe(data)

    def forecast_in_forecast_result(
        self,
        data: pd.DataFrame,
        timestamp_column: str,
        target_columns: List[str],
        prediction_length: int,
        context_length: Optional[int] = None,
        id_columns: List[str] = [],
        quantile_levels: List[float] = DEFAULT_QUANTILE_LEVELS,
        batch_size: int = 16,
    ) -> ForecastResult:
        """Run inference and return a structured ForecastResult.

        Args:
            data: DataFrame containing time series data.
            timestamp_column: Name of the timestamp column.
            target_columns: List of target column names to forecast.
            prediction_length: Number of future timesteps to forecast.
            context_length: Number of historical timesteps to use as context.
                If None, defaults to the model's own context length.
            id_columns: Columns identifying distinct time series. Default: [].
            quantile_levels: Quantile levels for probabilistic forecasting.
                Default: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9].
            batch_size: Inference batch size. Default: 16.

        Returns:
            ForecastResult with predicted, actuals, predicted_quantiles,
            cutoff_dates, and metadata.
        """
        forecast = self.__call__(
            data=data,
            timestamp_column=timestamp_column,
            target_columns=target_columns,
            context_length=context_length,
            prediction_length=prediction_length,
            id_columns=id_columns,
            quantile_levels=quantile_levels,
            batch_size=batch_size,
        )

        predicted_list, actuals_list, predicted_quantiles_list, error_msg = (
            _extract_predictions_and_actuals(
                forecast, target_columns, quantile_levels=quantile_levels
            )
        )

        if error_msg:
            return ForecastResult(success=False, message=error_msg, metadata={})

        cutoff_dates = None
        if timestamp_column in forecast.columns:
            cutoff_dates = forecast[timestamp_column].astype(str).tolist()

        return ForecastResult(
            success=True,
            message="Forecasting completed successfully",
            predicted=predicted_list,
            actuals=actuals_list,
            predicted_quantiles=predicted_quantiles_list,
            cutoff_dates=cutoff_dates,
            metadata={
                "model_checkpoint": self.model.name_or_path,
                "model_type": str(self.model.__class__),
                "context_length": context_length,
                "prediction_length": prediction_length,
                "quantile_levels": quantile_levels,
                "n_targets": len(target_columns),
                "n_samples": len(predicted_list) if predicted_list else 0,
                "device": self.device,
                "batch_size": batch_size,
            },
        )

    def forecast_for_ensemble(
        self,
        data: pd.DataFrame,
        timestamp_column: str,
        target_columns: List[str],
        prediction_length: int,
        context_length: Optional[int] = None,
        id_columns: List[str] = [],
        quantile_levels: List[float] = DEFAULT_QUANTILE_LEVELS,
        batch_size: int = 16,
    ) -> np.ndarray:
        """Run inference and return quantile forecast array for ensemble aggregation.

        Args:
            data: DataFrame containing time series data.
            timestamp_column: Name of the timestamp column.
            target_columns: List of target column names to forecast.
            prediction_length: Number of future timesteps to forecast.
            context_length: Number of historical timesteps to use as context.
                If None, defaults to the model's own context length.
            id_columns: Columns identifying distinct time series. Default: [].
            quantile_levels: Quantile levels for probabilistic forecasting.
                Default: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9].
            batch_size: Inference batch size. Default: 16.

        Returns:
            np.ndarray of shape (n_samples, prediction_length, n_targets, n_quantiles).
        """
        result = self.forecast_in_forecast_result(
            data=data,
            timestamp_column=timestamp_column,
            target_columns=target_columns,
            context_length=context_length,
            prediction_length=prediction_length,
            id_columns=id_columns,
            quantile_levels=quantile_levels,
            batch_size=batch_size,
        )
        return np.array(result.predicted_quantiles)
         




class PatchTSTFMDataFramePipelineForecaster(DataFramePipelineForecaster):

    def __init__(self, 
                 model_checkpoint:str="ibm-research/patchtst-fm-r1",
                 device:str = None):
        model = PatchTSTFMForPrediction.from_pretrained(model_checkpoint)
        super().__init__(model=model, device=device)


class FlowStateDataFramePipelineForecaster(DataFramePipelineForecaster):

    def __init__(self,
                 model_checkpoint:str="ibm-granite/granite-timeseries-flowstate-r1",
                 model_revision="r1.1",
                 device:str = None,
                 scale_factor:float=1,
                 batch_first:bool = True):

        model = FlowStateForPrediction.from_pretrained(
            model_checkpoint, 
            batch_first=batch_first, 
            scale_factor=scale_factor, 
            revision=model_revision
        )
        super().__init__(model=model, device=device)
        


class TinyTimeMixerDataFramePipelineForecaster(DataFramePipelineForecaster):

    def __init__(self,
                 model_checkpoint:str="ibm-granite/granite-timeseries-ttm-r3",
                 model_revision:str=None,
                 device:str=None):

        self.model_checkpoint = model_checkpoint
        self._ttm_model_key = model_revision

        if model_revision:
            model = TinyTimeMixerForPrediction.from_pretrained(model_checkpoint, revision=model_revision)
        else:
            model=None

        super().__init__(model=model,
                         device=device)

    def __call__(self, 
                 data: pd.DataFrame, 
                 timestamp_column: str, 
                 target_columns: List[str], 
                 prediction_length: int, 
                 context_length: Optional[int] = None, 
                 id_columns: List[str] = [], 
                 quantile_levels: List[float] = DEFAULT_QUANTILE_LEVELS, 
                 batch_size: int = 64,
                 scaling:bool=False,
                 scaler_type:str="standard",
                 use_get_model:bool = True,
    ) -> pd.DataFrame:
        """Run inference and return the raw forecast DataFrame.

        Args:
            data: DataFrame containing time series data.
            timestamp_column: Name of the timestamp column.
            target_columns: List of target column names to forecast.
            prediction_length: Number of future timesteps to forecast.
            context_length: Number of historical timesteps to use as context.
                If None, defaults to the model's own context length.
            id_columns: Columns identifying distinct time series. Default: [].
            quantile_levels: Quantile levels for probabilistic forecasting.
                Default: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9].
            batch_size: Inference batch size. Default: 64.
            scaling: 
            scaler_type:
            use_get_model:

        Returns:
            pd.DataFrame with forecast results from TimeSeriesForecastingPipeline.
        """

        # Pick context_length to use for choosing model
        if context_length is None:
            requested_context_length_or_data_length = data.shape[0]
        else:
            requested_context_length_or_data_length = context_length

        if use_get_model:
            model_key = get_model(model_path=self.model_checkpoint, 
                              model_name="ttm", 
                              context_length = requested_context_length_or_data_length, 
                              prediction_length=prediction_length,
                              return_model_key=True)

            if self._ttm_model_key:
                if model_key != self._ttm_model_key:
                    msg = f"TTM model key changed from {self._ttm_model_key} to {model_key}"
                    logging.info(msg)

            
            self.ttm_model_key = model_key
            
            model_class = (
                TinyTimeMixerForDecomposedPrediction
                if "-dec-" in model_key
                else TinyTimeMixerForPrediction
            )
            self.model = model_class.from_pretrained(self.model_checkpoint, revision=model_key)

        # Store context length and prediction length of TTM model
        self._ttm_model_context_length = self.model.config.context_length
        self._ttm_model_prediction_length = self.model.config.prediction_length

        pipeline_context_length = requested_context_length_or_data_length
        if pipeline_context_length > self._ttm_model_context_length:
            if id_columns or len(data) != pipeline_context_length:
                raise ValueError(
                    "Trimming context to the selected TTM model's native length is supported "
                    "only for a single series with one context window."
                )
            # Retain the forecast cutoff while using only the model's native context.
            pipeline_context_length = self._ttm_model_context_length
            data = data.tail(pipeline_context_length)

        # validate the data
        assert not data.empty, "Input data is empty"
        assert timestamp_column in data.columns, \
            f"Timestamp column '{timestamp_column}' not found in data"
        for col in target_columns:
            assert col in data.columns, \
                f"Target column '{col}' not found in data"

        # Setup preprocessor
        tsp = TimeSeriesPreprocessor(
            timestamp_column=timestamp_column,
            target_columns=target_columns,
            prediction_length=self._ttm_model_prediction_length,
            context_length=self._ttm_model_context_length,
            scaling=scaling,
            scaler_type=scaler_type,
        )
        tsp.train(data)


        fpipe = TimeSeriesForecastingPipeline(
            model=self.model,
            id_columns=id_columns,
            timestamp_column=timestamp_column,
            target_columns=target_columns,
            max_context_length=self.model.config.context_length,
            context_length=pipeline_context_length,
            prediction_length=self._ttm_model_prediction_length,
            batch_size=batch_size,
            impute_method=None,
            device=self.device,
            quantile_levels=quantile_levels,
            feature_extractor=tsp,
        )

        forecast:pd.DataFrame = fpipe(data)

        # trim forecasts to requested prediction_length
        if self._ttm_model_prediction_length > prediction_length:
            for idx, row in forecast.iterrows():
                for col in [c for c in forecast.columns if c.startswith(tuple(target_columns))]:
                    value = row[col]
                    # shorten value to the requested prediction_length
                    forecast.at[idx, col] = value[:prediction_length]
             

        return forecast


    def forecast_in_forecast_result(self, data,
                                   timestamp_column, 
                                   target_columns,
                                   prediction_length, 
                                   context_length = None, 
                                   id_columns = [], 
                                   quantile_levels = DEFAULT_QUANTILE_LEVELS, 
                                   batch_size = 64,
                                   scaling:bool=False,
                                   scaler_type:str="standard",
                                   use_get_model:bool = True
    ) -> ForecastResult:

        forecast = self.__call__(
            data=data,
            timestamp_column=timestamp_column,
            target_columns=target_columns,
            context_length=context_length,
            prediction_length=prediction_length,
            id_columns=id_columns,
            quantile_levels=quantile_levels,
            batch_size=batch_size,
            scaling=scaling,
            scaler_type=scaler_type,
            use_get_model=use_get_model
        )

        predicted_list, actuals_list, predicted_quantiles_list, error_msg = (
            _extract_predictions_and_actuals(
                forecast, target_columns, quantile_levels=quantile_levels
            )
        )

        if error_msg:
            return ForecastResult(success=False, message=error_msg, metadata={})

        cutoff_dates = None
        if timestamp_column in forecast.columns:
            cutoff_dates = forecast[timestamp_column].astype(str).tolist()

        return ForecastResult(
            success=True,
            message="Forecasting completed successfully",
            predicted=predicted_list,
            actuals=actuals_list,
            predicted_quantiles=predicted_quantiles_list,
            cutoff_dates=cutoff_dates,
            metadata={
                "model_checkpoint": self.model.name_or_path,
                "model_type": str(self.model.__class__),
                "context_length": context_length,
                "prediction_length": prediction_length,
                "quantile_levels": quantile_levels,
                "n_targets": len(target_columns),
                "n_samples": len(predicted_list) if predicted_list else 0,
                "device": self.device,
                "batch_size": batch_size,
            },
        )


    def forecast_for_ensemble(self, data, 
                                   timestamp_column, 
                                   target_columns,
                                   prediction_length, 
                                   context_length = None, 
                                   id_columns = [], 
                                   quantile_levels = DEFAULT_QUANTILE_LEVELS, 
                                   batch_size = 64,
                                   scaling:bool=False,
                                   scaler_type:str="standard",
                                   use_get_model:bool = True
    ) -> np.array:

        forecast = self.forecast_in_forecast_result(
            data=data,
            timestamp_column=timestamp_column,
            target_columns=target_columns,
            context_length=context_length,
            prediction_length=prediction_length,
            id_columns=id_columns,
            quantile_levels=quantile_levels,
            batch_size=batch_size,
            scaling=scaling,
            scaler_type=scaler_type,
            use_get_model=use_get_model
        )

        return np.array(forecast.predicted_quantiles)




def _extract_predictions_and_actuals(
    forecast: pd.DataFrame,
    target_columns: List[str],
    quantile_levels: Optional[List[float]] = None,
) -> tuple[list, list, Optional[list], Optional[str]]:
    """
    Extract predictions, actuals, and quantiles from forecast DataFrame.

    Helper function to avoid code duplication across forecasting functions.

    Args:
        forecast: DataFrame containing forecast results with prediction and actual columns.
        target_columns: List of target column names.
        quantile_levels: Optional list of quantile levels to extract.

    Returns:
        Tuple of (predicted_list, actuals_list, predicted_quantiles_list, error_message):
            - predicted_list: List of prediction arrays, shape (n_samples, prediction_length, n_targets)
            - actuals_list: List of actual arrays, shape (n_samples, prediction_length, n_targets)
            - predicted_quantiles_list: List of quantile arrays, shape (n_samples, prediction_length, n_targets, n_quantiles)
              or None if quantile_levels is None
            - error_message: Error message if extraction fails, None otherwise
    """
    # Validate columns exist
    for target_col in target_columns:
        pred_col = f"{target_col}_prediction"
        if pred_col not in forecast.columns:
            return (
                [],
                [],
                None,
                f"Prediction column '{pred_col}' not found in forecast output",
            )
        if target_col not in forecast.columns:
            return (
                [],
                [],
                None,
                f"Actual column '{target_col}' not found in forecast output",
            )

    # Extract predictions: shape (n_targets, n_samples, prediction_length)
    # Then transpose to (n_samples, prediction_length, n_targets)
    predicted_array = np.array(
        [
            np.stack(z)
            for z in forecast[[f"{tc}_prediction" for tc in target_columns]].values
        ]
    ).transpose(
        0, 2, 1
    )  # (n_targets, n_samples, pred_len) -> (n_samples, pred_len, n_targets)

    # Extract actuals: shape (n_targets, n_samples, prediction_length)
    # Then transpose to (n_samples, prediction_length, n_targets)
    actuals_array = np.array(
        [np.stack(z) for z in forecast[[tc for tc in target_columns]].values]
    ).transpose(
        0, 2, 1
    )  # (n_targets, n_samples, pred_len) -> (n_samples, pred_len, n_targets)

    # Convert to lists
    predicted_list = predicted_array.tolist()
    actuals_list = actuals_array.tolist()

    # Extract quantiles if requested
    predicted_quantiles_list = None
    if quantile_levels is not None and len(quantile_levels) > 0:
        # Shape: (n_targets, n_quantiles, n_samples, prediction_length)
        # Target shape: (n_samples, prediction_length, n_targets, n_quantiles)
        quantile_arrays = []
        for target_col in target_columns:
            target_quantiles = []
            for q in quantile_levels:
                q_col = f"{target_col}_prediction_q{q}"
                if q_col not in forecast.columns:
                    return (
                        [],
                        [],
                        None,
                        f"Quantile column '{q_col}' not found in forecast output",
                    )
                # q_array = np.array([np.stack(z) for z in forecast[[q_col]].values]).squeeze()
                q_array = np.array(
                    [np.stack(z) for z in forecast[q_col].values]
                )  # n_samples x prediction_lenght
                target_quantiles.append(q_array)  # (n_samples, prediction_length)
            quantile_arrays.append(
                np.stack(target_quantiles, axis=-1)
            )  # (n_samples, prediction_length, n_quantiles)

        # Stack targets: (n_targets, n_samples, prediction_length, n_quantiles)
        # Transpose to: (n_samples, prediction_length, n_targets, n_quantiles)
        quantiles_array = np.stack(quantile_arrays, axis=0).transpose(1, 2, 0, 3)
        predicted_quantiles_list = quantiles_array.tolist()

    return predicted_list, actuals_list, predicted_quantiles_list, None
