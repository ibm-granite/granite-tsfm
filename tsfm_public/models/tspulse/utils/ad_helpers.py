from collections import OrderedDict

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler as MinMaxScaler_
from torch import nn as nn
from transformers.utils.generic import ModelOutput

from tsfm_public.models.tspulse.modeling_tspulse import TSPulseForReconstruction
from tsfm_public.toolkit.ad_helpers import ScoreListType, TSADHelperUtility

from .helpers import patchwise_stitched_reconstruction


class TSPulseADUtility(TSADHelperUtility):
    def __init__(self, model: TSPulseForReconstruction, mode: str, aggr_win_size: int, **kwargs):
        if mode is None:
            mode = "forecast+fft+time"

        super(TSPulseADUtility, self).__init__(**kwargs)
        if not self.is_valid_mode(mode):
            raise ValueError(f"Error: unsupported inference method {mode}!")
        self._model = model
        self._mode = mode
        self._aggr_win_size = aggr_win_size

    def is_valid_mode(self, mode_str: str) -> bool:
        supported_modes = ["time", "fft", "forecast"]

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
        use_fft = "fft" in mode
        use_ts = "time" in mode
        aggr_win_size = kwargs.get("aggr_win_size", self._aggr_win_size)
        anomaly_criterion = nn.MSELoss(reduce=False)

        reconstruct_start = self._model.config.context_length - aggr_win_size
        reconstruct_end = self._model.config.context_length

        batch_x = payload["past_values"]

        # Get TSPulse zeroshot output with stitched masked reconstruction
        keys_to_stitch = ["reconstruction_outputs", "reconstructed_ts_from_fft"]

        model_forward_output = {}
        if use_forecast:
            model_forward_output = self._model(past_values=batch_x)

        stitched_dict = {}
        if use_ts or use_fft:
            stitched_dict = patchwise_stitched_reconstruction(
                model=self._model,
                past_values=batch_x,
                patch_size=self._model.config.patch_length,
                keys_to_stitch=keys_to_stitch,
                keys_to_aggregate=[],
                reconstruct_start=reconstruct_start,
                reconstruct_end=reconstruct_end,
                debug=False,
            )
            if isinstance(stitched_dict, tuple):
                stitched_dict = stitched_dict[0]

        # Get desired output from TSPulse outputs
        # output shape: [batch_size, window_size, n_channels]
        scores = OrderedDict()

        if use_ts:
            # time reconstruction
            output = stitched_dict["reconstruction_outputs"]
            pointwise_score = anomaly_criterion(
                batch_x[:, reconstruct_start:reconstruct_end, :],
                output[:, reconstruct_start:reconstruct_end, :],
            )
            scores["time"] = torch.mean(pointwise_score, dim=[1, 2])

        if use_fft:
            # time reconstruction from fft
            output = stitched_dict["reconstructed_ts_from_fft"]
            pointwise_score = anomaly_criterion(
                batch_x[:, reconstruct_start:reconstruct_end, :],
                output[:, reconstruct_start:reconstruct_end, :],
            )
            scores["fft"] = torch.mean(pointwise_score, dim=[1, 2])

        if use_forecast:
            # forecast output
            batch_future_values = payload["future_values"]
            output = model_forward_output["forecast_output"]
            pointwise_score = anomaly_criterion(batch_future_values[:, 0, :], output[:, 0, :]).unsqueeze(1)
            scores["forecast"] = torch.mean(pointwise_score, dim=[1, 2])

        return ModelOutput(scores)

    def boundary_adjusted_scores(
        self,
        key: str,
        x: ScoreListType,
        **kwargs,
    ) -> np.ndarray:
        context_length = kwargs.get("context_length", self._model.config.context_length)
        aggr_win_size = kwargs.get("aggr_win_size", self._aggr_win_size)
        if isinstance(x, (list, tuple)):
            if (len(x) > 0) and isinstance(x[0], torch.Tensor):
                x = torch.cat(x, axis=0).detach().cpu().numpy()
            else:
                x = np.concatenate(x, axis=0).astype(float)
        if key == "forecast":
            start_pad_len = context_length
            end_pad_len = 0
        else:
            start_pad_len = context_length - aggr_win_size // 2
            end_pad_len = aggr_win_size // 2
        score = np.array([x[0]] * start_pad_len + list(x) + [x[-1]] * end_pad_len)
        return MinMaxScaler_().fit_transform(score.reshape(-1, 1))
