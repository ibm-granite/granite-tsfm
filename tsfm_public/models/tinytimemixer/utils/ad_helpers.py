from collections import OrderedDict
from typing import Optional

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler as MinMaxScaler_
from torch import nn as nn
from transformers.utils.generic import ModelOutput

from tsfm_public.models.tinytimemixer.modeling_tinytimemixer import TinyTimeMixerForPrediction
from tsfm_public.toolkit.ad_helpers import ScoreListType, TSADHelperUtility


class TinyTimeMixerADUtility(TSADHelperUtility):
    """Implements TSAD Helper Utility for TSPulse model"""

    def __init__(
        self,
        model: TinyTimeMixerForPrediction,
        mode: str,
        least_significant_scale: float = 1e-2,
        least_significant_score: float = 0.2,
        **kwargs,
    ):
        """Initializer

        Args:
            model (TinyTimeMixerForPrediction): model instance
            mode (str): mode string specifies scoring logic
            least_significant_scale (float, optional): allowed model deviation from the data in the scale of data variance. Defaults to 1e-2.
            least_significant_score (float, optional): minimum anomaly score for significant detection. Defaults to 0.2.

        Raises:
            ValueError: unsupported scoring mode
        """
        if mode is None:
            mode = "forecast"
        super(TinyTimeMixerADUtility, self).__init__()
        if not self.is_valid_mode(mode):
            raise ValueError(f"Error: unsupported inference method {mode}!")
        self._model = model
        self._mode = mode
        self._least_significant_scale = least_significant_scale
        self._least_significant_score = least_significant_score

    def is_valid_mode(
        self,
        mode_str: str,
    ) -> bool:
        """Validates compatibility of the specified mode string."""
        supported_modes = ["forecast", "meandev"]
        valid_mode = 0
        for mode_type in supported_modes:
            if mode_type in mode_str:
                valid_mode += 1
        return valid_mode == 1

    def compute_score(
        self,
        payload: dict,
        expand_score: bool = False,
        **kwargs,
    ) -> ModelOutput:
        """Produces required model output for anomaly scoring

        Args:
            payload (dict): data batch
            expand_score (bool, optional): compute score for each stream for multivariate data. Defaults to False.

        Returns:
            ModelOutput: model output
        """
        mode = kwargs.get("mode", self._mode)
        use_forecast = "forecast" in mode
        use_meandev = "meandev" in mode
        anomaly_criterion = nn.MSELoss(reduce=False)

        model_forward_output = {}
        model_forward_output = self._model(**payload)

        scores = OrderedDict()
        if use_forecast:
            # forecast output
            reduction_axis = [1] if expand_score else [1, 2]
            batch_future_values = payload["future_values"]
            future_predictions = model_forward_output["prediction_outputs"]
            pointwise_score = anomaly_criterion(batch_future_values[:, 0, :], future_predictions[:, 0, :]).unsqueeze(1)
            scores["forecast"] = torch.mean(pointwise_score, dim=reduction_axis)
        if use_meandev:
            batch_future_values = payload["future_values"]
            future_predictions = model_forward_output["prediction_outputs"]
            deviation = batch_future_values - future_predictions
            if not expand_score:
                deviation = torch.mean(deviation, dim=[2])
            scores["meandev"] = deviation

        return ModelOutput(scores)

    def adjust_boundary(
        self,
        key: str,
        x: ScoreListType,
        reference: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Combines model outputs with boundary adjustment

        Args:
            key (str): key associated with model output
            x (ScoreListType): model outputs across all batches combined
            reference (Optional[np.ndarray], optional): reference data for score scale adjustment. Defaults to None.

        Returns:
            np.ndarray: combined score
        """
        if isinstance(x, (list, tuple)):
            if (len(x) > 0) and isinstance(x[0], torch.Tensor):
                x = torch.cat(x, axis=0).detach().cpu().numpy()
            else:
                x = np.concatenate(x, axis=0).astype(float)
        elif isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()

        if key == "meandev":
            n_batches = x.shape[0]
            n_obs = x.shape[1]
            total_length = n_batches + n_obs - 1
            data_dim = (total_length, x.shape[2]) if x.ndim == 3 else total_length
            counters = np.zeros(data_dim)
            predictions = np.zeros(data_dim)
            for i in range(n_batches):
                predictions[i : (i + n_obs)] += x[i]
                counters[i : (i + n_obs)] += 1
            x = (predictions / np.maximum(counters, 1)) ** 2

        start_pad_len = self._model.config.context_length
        end_pad_len = 0 if key == "meandev" else self._model.config.prediction_length - 1
        score = np.array([x[0]] * start_pad_len + list(x) + [x[-1]] * end_pad_len)
        if score.ndim == 1:
            score = score.reshape(-1, 1)

        min_score = 0.0
        if reference is not None:
            reference_data = np.asarray(reference)
            min_score = self._least_significant_scale * np.nanstd(reference_data, axis=0, keepdims=True) ** 2
            if min_score.shape[-1] != score.shape[-1]:
                min_score = np.nanmax(min_score, axis=-1)
            if key == "forecast":
                min_score = min_score / np.sqrt(2)
        score_ = score.copy()
        score_[np.where(score > min_score)] *= 1 / self._least_significant_score
        scale = 1 if np.any(score > min_score) else self._least_significant_scale
        score = MinMaxScaler_().fit_transform(score_) * scale
        return score
