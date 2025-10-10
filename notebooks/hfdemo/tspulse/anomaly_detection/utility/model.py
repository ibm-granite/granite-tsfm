# Copyright contributors to the TSFM project
#
import numpy as np
import pandas as pd

from tsfm_public.models.tspulse.modeling_tspulse import TSPulseForReconstruction
from tsfm_public.toolkit.time_series_anomaly_detection_pipeline import TimeSeriesAnomalyDetectionPipeline

from .base_model import BaseDetector


MODEL_PATH = "ibm-granite/granite-timeseries-tspulse-r1"


def attach_timestamp_column(
    df: pd.DataFrame, time_col: str = "timestamp", freq: str = "5s", start_date: str = "2002-01-01"
):
    n = df.shape[0]
    if time_col not in df:
        df[time_col] = pd.date_range(start_date, freq=freq, periods=n)
    return df


class TSAD_Pipeline(BaseDetector):
    def __init__(
        self,
        batch_size: int = 256,
        aggr_win_size: int = 96,
        num_input_channels: int = 1,
        smoothing_window: int = 8,
        prediction_mode: str = "forecast+time+fft",
        **kwargs,
    ):
        self._batch_size = batch_size
        self._headers = [f"x{i + 1}" for i in range(num_input_channels)]
        self._model = TSPulseForReconstruction.from_pretrained(
            MODEL_PATH, num_input_channels=num_input_channels, scaling="revin", mask_type="user"
        )
        prediction_mode_array = [s_.strip() for s_ in str(prediction_mode).split("+")]
        self._scorer = TimeSeriesAnomalyDetectionPipeline(
            self._model,
            timestamp_column="timestamp",
            target_columns=self._headers,
            prediction_mode=prediction_mode_array,
            aggregation_length=aggr_win_size,
            smoothing_length=smoothing_window,
            least_significant_scale=0.0,
            least_significant_score=1.0,
        )

    def zero_shot(self, x, label=None):
        self.decision_scores_ = self.decision_function(x)

    def fit(self, X, y=None):
        pass

    def decision_function(self, X):
        """
        Not used, present for API consistency by convention.
        """
        data = attach_timestamp_column(pd.DataFrame(X, columns=self._headers))
        score = self._scorer(data, batch_size=self._batch_size)
        if not isinstance(score, pd.DataFrame) or ("anomaly_score" not in score):
            raise ValueError("Error: expect anomaly_score column in the output!")

        score = score["anomaly_score"].values.ravel()
        norm_value = np.nanmax(np.asarray(score), axis=0, keepdims=True) + 1e-5
        anomaly_score = score / norm_value
        return anomaly_score
