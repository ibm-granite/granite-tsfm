from collections import OrderedDict

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler as MinMaxScaler_
from torch import nn as nn
from transformers.utils.generic import ModelOutput

from tsfm_public.models.tspulse.modeling_tspulse import TSPulseForReconstruction

from .helpers import patchwise_stitched_reconstruction


def is_valid_tspulse_mode(mode_str: str) -> bool:
    supported_modes = ["time", "fft", "forecast"]

    valid_mode = False
    for mode_type in supported_modes:
        if mode_type in mode_str:
            valid_mode = True
    return valid_mode


def compute_tspulse_score(
    model: TSPulseForReconstruction,
    payload: dict,
    mode: str,
    aggr_win_size: int,
    **kwargs,
) -> ModelOutput:
    use_forecast = "forecast" in mode
    use_ts_from_fft = "fft" in mode
    use_ts = "time" in mode
    aggr_win_size = aggr_win_size
    anomaly_criterion = nn.MSELoss(reduce=False)

    reconstruct_start = model.config.context_length - aggr_win_size
    reconstruct_end = model.config.context_length

    batch_x = payload["past_values"]

    # Get TSPulse zeroshot output with stitched masked reconstruction
    keys_to_stitch = ["reconstruction_outputs", "reconstructed_ts_from_fft"]

    model_forward_output = {}
    if use_forecast:
        model_forward_output = model(past_values=batch_x)

    stitched_dict = {}
    if use_ts or use_ts_from_fft:
        stitched_dict = patchwise_stitched_reconstruction(
            model=model,
            past_values=batch_x,
            patch_size=model.config.patch_length,
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

    if use_ts_from_fft:
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


def boundary_adjusted_tspulse_scores(
    key: str,
    x: np.ndarray,
    context_length: int,
    aggr_win_size: int,
) -> np.ndarray:
    if key == "forecast":
        start_pad_len = context_length
        end_pad_len = 0
    else:
        start_pad_len = context_length - aggr_win_size // 2
        end_pad_len = aggr_win_size // 2
    score = np.array([x[0]] * start_pad_len + list(x) + [x[-1]] * end_pad_len)
    return MinMaxScaler_().fit_transform(score.reshape(-1, 1))
