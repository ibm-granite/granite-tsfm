from collections import OrderedDict

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler as MinMaxScaler_
from torch import nn as nn
from transformers.utils.generic import ModelOutput

from tsfm_public.models.tinytimemixer.modeling_tinytimemixer import TinyTimeMixerForPrediction
from tsfm_public.toolkit.ad_helpers import ScoreListType, TSADHelperUtility
from tsfm_public.toolkit.conformal import PostHocProbabilisticProcessor


class TinyTimeMixerADUtility(TSADHelperUtility):
    def __init__(
        self,
        model: TinyTimeMixerForPrediction,
        posthoc_probabilistic_processor: PostHocProbabilisticProcessor,
        mode: str,
        **kwargs,
    ):
        if mode is None:
            mode = "forecast"
        super(TinyTimeMixerADUtility, self).__init__(**kwargs)
        if not self.is_valid_mode(mode):
            raise ValueError(f"Error: unsupported inference method {mode}!")
        self._model = model
        self._mode = mode
        self._posthoc_processor = posthoc_probabilistic_processor

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
        mode = kwargs.get("mode", self._mode)
        use_forecast = "forecast" in mode
        # anomaly_criterion = nn.MSELoss(reduce=False)
        # batch_x = payload["past_values"]

        model_forward_output = {}
        model_forward_output = self._model(**payload)

        scores = OrderedDict()
        if use_forecast:
            # forecast output
            batch_future_values = payload["future_values"]
            future_predictions = model_forward_output["prediction_outputs"]

            # for now assume calibration in normalized space, so no inverse is needed here

            scores["forecast"] = self.posthoc_processor.outlier_score(
                y_gt=batch_future_values, y_pred=future_predictions
            )
            # batch size x forecast_horizon x number features
            # pointwise_score = anomaly_criterion(batch_future_values[:, 0, :], future_predictions[:, 0, :]).unsqueeze(1)
            # scores["forecast"] = torch.mean(pointwise_score, dim=[1, 2])

        return ModelOutput(scores)

    def adjust_boundary(
        self,
        key: str,
        x: ScoreListType,
        **kwargs,
    ) -> np.ndarray:
        """_summary_

        Take list of scores (bs x prediction_length x num_features)

        Args:
            key (str): _description_
            x (ScoreListType): _description_

        Returns:
            np.ndarray: _description_
        """

        start_pad_len = self._model.config.context_length
        end_pad_len = self._model.config.prediction_length - 1
        if isinstance(x, (list, tuple)):
            if (len(x) > 0) and isinstance(x[0], torch.Tensor):
                x = torch.cat(x, axis=0).detach().cpu().numpy()
            else:
                x = np.concatenate(x, axis=0).astype(float)

        # call self._posthoc_processor.aggregate_method
        # time aggregation
        # dataframe length x number features
        score = np.array([x[0]] * start_pad_len + list(x) + [x[-1]] * end_pad_len)
        return MinMaxScaler_().fit_transform(score.reshape(-1, 1))
