from collections import OrderedDict

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler as MinMaxScaler_
from torch import nn as nn
from transformers.utils.generic import ModelOutput

from tsfm_public.models.tinytimemixer.modeling_tinytimemixer import TinyTimeMixerForPrediction
from tsfm_public.toolkit.ad_helpers import ScoreListType, TSADHelperUtility


class TinyTimeMixerADUtility(TSADHelperUtility):
    def __init__(self, model: TinyTimeMixerForPrediction, mode: str, **kwargs):
        if mode is None:
            mode = "forecast"
        super(TinyTimeMixerADUtility, self).__init__()
        if not self.is_valid_mode(mode):
            raise ValueError(f"Error: unsupported inference method {mode}!")
        self._model = model
        self._mode = mode
        self._least_significant_scale = kwargs.get('least_significant_scale', 1e-2)
        self._least_significant_score = kwargs.get('least_significant_score', 0.2)

    def is_valid_mode(
        self,
        mode_str: str,
    ) -> bool:
        supported_modes = ["forecast"]
        valid_mode = False
        for mode_type in supported_modes:
            if mode_type in mode_str:
                valid_mode = True
        return valid_mode

    def compute_score(
        self,
        payload: dict,
        **kwargs,
    ) -> ModelOutput:
        expand_score = kwargs.get("expand_score", False)
        mode = kwargs.get("mode", self._mode)
        use_forecast = "forecast" in mode
        anomaly_criterion = nn.MSELoss(reduce=False)
        batch_x = payload["past_values"]

        model_forward_output = {}
        model_forward_output = self._model(past_values=batch_x)
        reduction_axis = [1] if expand_score else [1, 2] 

        scores = OrderedDict()
        if use_forecast:
            # forecast output
            batch_future_values = payload["future_values"]
            future_predictions = model_forward_output["prediction_outputs"]
            pointwise_score = anomaly_criterion(batch_future_values[:, 0, :], future_predictions[:, 0, :]).unsqueeze(1)
            scores["forecast"] = torch.mean(pointwise_score, dim=reduction_axis)

        return ModelOutput(scores)

    def adjust_boundary(
        self,
        key: str,
        x: ScoreListType,
        **kwargs,
    ) -> np.ndarray:
        start_pad_len = self._model.config.context_length
        end_pad_len = self._model.config.prediction_length - 1
        if isinstance(x, (list, tuple)):
            if (len(x) > 0) and isinstance(x[0], torch.Tensor):
                x = torch.cat(x, axis=0).detach().cpu().numpy()
            else:
                x = np.concatenate(x, axis=0).astype(float)
        score = np.array([x[0]] * start_pad_len + list(x) + [x[-1]] * end_pad_len)
        if score.ndim == 1:
            score = score.reshape(-1, 1)
        
        min_score = 0. 
        if 'reference' in kwargs:
            reference_data = np.asarray(kwargs.get('reference'))
            min_score = self._least_significant_scale * np.nanstd(reference_data, axis=0, keepdims=True)**2
            if min_score.shape[-1] != score.shape[-1]:
                min_score = np.nanmax(min_score, axis=-1) 
            min_score = min_score / np.sqrt(2)
        score_ = score.copy()
        score_[np.where(score > min_score)] *= 1/self._least_significant_score
        scale = 1 if np.any(score > min_score) else self._least_significant_scale
        score = MinMaxScaler_().fit_transform(score_) * scale
        return score
