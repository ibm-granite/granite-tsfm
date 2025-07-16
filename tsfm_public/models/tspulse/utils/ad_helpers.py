# Copyright contributors to the TSFM project
#
"""Helper class for anomaly detection support"""

from collections import OrderedDict
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler as MinMaxScaler_
from sklearn.preprocessing import StandardScaler as StandardScaler_
from torch import nn as nn
from transformers.utils.generic import ModelOutput

from tsfm_public.models.tspulse.modeling_tspulse import TSPulseForReconstruction
from tsfm_public.toolkit.ad_helpers import (
    AnomalyScoreMethods,
    ScoreListType,
    TSADHelperUtility,
)

from .helpers import patchwise_stitched_reconstruction, patchwise_stitched_reconstruction_with_separation


def causal_minmax(
    x,
    upper: Optional[np.ndarray | List[float]] = None,
    lower: Optional[np.ndarray | List[float]] = None,
    score2_mean: Optional[np.ndarray | List[float]] = None,
    score_mean: Optional[np.ndarray | List[float]] = None,
    score_history_length: Optional[int] = None,
    sigma_factor: float = 6.0,
    max_history_length: int = 10_000,
    eps: float = 1e-2,
    **kwargs,
) -> Tuple[np.ndarray, dict]:
    """Perform causal minmax scaling. Allow state management, to score in a streaming fashion.

    Args:
        x (_type_): data as 1D/2D numpy array
        upper (Optional[np.ndarray  |  List[float]], optional): upper bound of score known from history. Defaults to None.
        lower (Optional[np.ndarray  |  List[float]], optional): lower bound of the score known from history. Defaults to None.
        score2_mean (Optional[np.ndarray  |  List[float]], optional): states for computing running standard deviation of the scores. Defaults to None.
        score_mean (Optional[np.ndarray  |  List[float]], optional): states for computing running mean of the scores. Defaults to None.
        score_history_length (Optional[int], optional): size historical observation over which the state parameters are computer. Defaults to None.
        sigma_factor (float, optional): sigma factor for estimating upper bound for cold start use case. Defaults to 6..
        max_history_length(int, optional): maximum history length to be maintained for state. Default to 10000.
        eps (float, optional): min value cutoff for computational stability. Defaults to 1e-2.

    Raises:
        ValueError: Invalid data shape

    """
    x_ = np.asarray(x)
    expanded = False
    if x_.ndim == 1:
        x_ = x_.reshape(-1, 1)
        expanded = True
    if x_.ndim != 2:
        raise ValueError(f"Expects: 1D / 2D data got {x_.shape}!")

    curr_std = np.nanstd(x_, axis=0)
    curr_len = len(x_)

    if score2_mean is None:
        score2_mean = np.zeros(x_.shape[1])

    if score_mean is None:
        score_mean = np.zeros(x_.shape[1])

    if score_history_length is None:
        score_history_length = 0

    score_history_length = min(score_history_length, max_history_length)
    score2_mean = np.asarray(score2_mean)
    score_mean = np.asarray(score_mean)

    curr_mean_2 = np.mean(x_**2, axis=0)
    curr_mean = np.mean(x_, axis=0)
    curr_mean_2 = (curr_mean_2 * curr_len + score2_mean * score_history_length) / (curr_len + score_history_length)
    curr_mean = (curr_mean * curr_len + score_mean * score_history_length) / (curr_len + score_history_length)
    curr_len += score_history_length

    curr_std = np.maximum(np.sqrt(curr_mean_2 - np.square(curr_mean)), eps)

    dummy_head = [f"x{i}" for i in range(x_.shape[1])]
    x_expanding = pd.DataFrame(x_, columns=dummy_head).expanding()
    x_max = x_expanding.max().values
    x_min = x_expanding.min().values

    if (upper is None) or (len(upper) != x_.shape[-1]):
        upper = np.asarray([-np.inf])

    upper = np.asarray(upper)

    if (lower is None) or (len(lower) != x_.shape[-1]):
        lower = np.asarray([np.inf])
    lower = np.asarray(lower)

    upper = np.maximum(upper, curr_mean + sigma_factor * curr_std)
    # lower = np.minimum(lower, curr_mean - sigma_factor * curr_std)
    x_max = np.maximum(x_max, upper)
    x_min = np.minimum(x_min, lower)
    num = x_ - x_min
    den = np.maximum(x_max - x_min, eps)
    num[num < eps] = 0
    upper, lower = x_max[-1, ...], x_min[-1, ...]
    x_scaled = num / den
    if expanded:
        x_scaled = x_scaled.ravel()

    state = {}

    state.update(
        upper=upper.tolist(),
        lower=lower.tolist(),
        score2_mean=curr_mean_2.tolist(),
        score_mean=curr_mean.tolist(),
        score_history_length=curr_len,
        sigma_factor=sigma_factor,
        max_history_length=max_history_length,
        eps=eps,
    )
    return x_scaled, state


class TSPulseADUtility(TSADHelperUtility):
    """Implements TSAD Helper Utility for TSPulse model"""

    def __init__(
        self,
        model: TSPulseForReconstruction,
        mode: List[str],
        aggregation_length: int,
        score_exponent: float = 1.0,
        least_significant_scale: float = 1e-2,
        least_significant_score: float = 0.2,
        **kwargs,
    ):
        """Initializer

        Args:
            model (TSPulseForReconstruction): model instance.
            mode (list): mode string specifies scoring logic.
            aggregation_length (int): parameter for imputation based scoring.
            score_exponent (float, optional): parameter to sharpen the anomaly score. Defaults to 1.
            least_significant_scale (float, optional): allowed model deviation from the data in the scale of data variance. Defaults to 1e-2.
            least_significant_score (float, optional): minimum anomaly score for significant detection. Defaults to 0.2.

        Raises:
            ValueError: unsupported scoring mode
            ValueError: aggregation window not multiple of model patch_length
        """
        if mode is None:
            mode = [
                AnomalyScoreMethods.PREDICTIVE.value,
                AnomalyScoreMethods.FREQUENCY_RECONSTRUCTION.value,
                AnomalyScoreMethods.TIME_RECONSTRUCTION.value,
            ]

        super(TSPulseADUtility, self).__init__()
        if not self.is_valid_mode(mode):
            raise ValueError(f"Error: unsupported inference method {mode}!")
        if aggregation_length % model.config.patch_length != 0:
            raise ValueError(
                f"Error: aggregation window must be multiple of model patch_length {model.config.patch_length}!"
            )

        self._model = model
        self._mode = mode
        self._aggr_win_size = aggregation_length
        self._score_exponent = score_exponent
        self._least_significant_scale = least_significant_scale
        self._least_significant_score = least_significant_score

    def is_valid_mode(self, mode_str: List[str]) -> bool:
        """Validates compatibility of the specified mode string."""
        supported_modes = [
            AnomalyScoreMethods.PREDICTIVE.value,
            AnomalyScoreMethods.FREQUENCY_RECONSTRUCTION.value,
            AnomalyScoreMethods.TIME_RECONSTRUCTION.value,
        ]

        valid_mode = False
        for mode_type in supported_modes:
            if mode_type in mode_str:
                valid_mode = True
        return valid_mode

    def preprocess(self, x: DataFrame, **kwargs) -> DataFrame:
        """Performs standard normalization on the target columns before scoring.

        Args:
            x (DataFrame): input for scoring

        Returns:
            DataFrame: processed dataframe
        """
        x = super().preprocess(x, **kwargs)
        x_ = x.copy()
        target_columns = kwargs.get("target_columns", [])
        if len(target_columns) > 0:
            x_[target_columns] = StandardScaler_().fit_transform(x[target_columns].values)
        return x_

    def compute_score(
        self,
        payload: dict,
        expand_score: bool = False,
        **kwargs,
    ) -> ModelOutput:
        """Produces required model output for anomaly scoring

        Args:
            payload (dict): data batch.
            expand_score (bool): compute score for each stream for multivariate data. Defaults to False.

        Returns:
            ModelOutput: model output
        """
        mode = kwargs.get("mode", self._mode)
        use_forecast = AnomalyScoreMethods.PREDICTIVE.value in mode
        use_fft = AnomalyScoreMethods.FREQUENCY_RECONSTRUCTION.value in mode
        use_ts = AnomalyScoreMethods.TIME_RECONSTRUCTION.value in mode
        aggr_win_size = self._aggr_win_size
        anomaly_criterion = nn.MSELoss(reduction="none")

        reconstruct_start = self._model.config.context_length - aggr_win_size
        reconstruct_end = self._model.config.context_length

        batch_x = payload["past_values"]

        # Get TSPulse zeroshot output with stitched masked reconstruction
        keys_to_stitch = ["reconstruction_outputs", "reconstructed_ts_from_fft"]

        model_forward_output = {}
        if use_forecast:
            # model_forward_output = self._model(**payload)
            model_forward_output = self._model(batch_x)

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

        reduction_axis = [1] if expand_score else [1, 2]
        if use_ts:
            # time reconstruction
            output = stitched_dict["reconstruction_outputs"]
            pointwise_score = anomaly_criterion(
                batch_x[:, reconstruct_start:reconstruct_end, :],
                output[:, reconstruct_start:reconstruct_end, :],
            )
            scores[AnomalyScoreMethods.TIME_RECONSTRUCTION.value] = torch.mean(pointwise_score, dim=reduction_axis)

        if use_fft:
            # time reconstruction from fft
            output = stitched_dict["reconstructed_ts_from_fft"]
            pointwise_score = anomaly_criterion(
                batch_x[:, reconstruct_start:reconstruct_end, :],
                output[:, reconstruct_start:reconstruct_end, :],
            )
            scores[AnomalyScoreMethods.FREQUENCY_RECONSTRUCTION.value] = torch.mean(
                pointwise_score, dim=reduction_axis
            )

        if use_forecast:
            # forecast output
            batch_future_values = payload["future_values"]
            output = model_forward_output["forecast_output"]
            pointwise_score = anomaly_criterion(batch_future_values[:, 0, :], output[:, 0, :]).unsqueeze(1)
            scores[AnomalyScoreMethods.PREDICTIVE.value] = torch.mean(pointwise_score, dim=reduction_axis)

        return ModelOutput(scores)

    def compute_score_(
        self,
        payload: dict,
        **kwargs,
    ) -> ModelOutput:
        """Produces required model output for anomaly scoring

        Args:
            payload (dict): data batch.
            expand_score (bool): compute score for each stream for multivariate data. Defaults to False.

        Returns:
            ModelOutput: model output
        """
        mode = kwargs.get("mode", self._mode)
        separation = kwargs.get("separation", None)
        batch_counter = kwargs.get("batch", 1)
        state = kwargs.get("state", {})
        use_forecast = AnomalyScoreMethods.PREDICTIVE.value in mode
        use_fft = AnomalyScoreMethods.FREQUENCY_RECONSTRUCTION.value in mode
        use_ts = AnomalyScoreMethods.TIME_RECONSTRUCTION.value in mode
        aggr_win_size = self._aggr_win_size
        anomaly_criterion = nn.MSELoss(reduction="none")

        reconstruct_start = self._model.config.context_length - aggr_win_size
        reconstruct_end = self._model.config.context_length

        batch_x = payload["past_values"]

        # Get TSPulse zeroshot output with stitched masked reconstruction
        keys_to_stitch = ["reconstruction_outputs", "reconstructed_ts_from_fft"]

        model_forward_output = {}
        if use_forecast:
            # model_forward_output = self._model(**payload)
            model_forward_output = self._model(batch_x)

        stitched_dict = {}
        boundary_dict = {}
        initialized = state.get("initialized", False)

        if use_ts or use_fft:
            if (batch_counter == 0) and (not initialized):
                reconstruct_start_ = 0
                boundary_dict = patchwise_stitched_reconstruction_with_separation(
                    model=self._model,
                    past_values=batch_x[:1],
                    patch_size=self._model.config.patch_length,
                    keys_to_stitch=keys_to_stitch,
                    keys_to_aggregate=[],
                    reconstruct_start=reconstruct_start_,
                    reconstruct_end=reconstruct_end,
                    separation=separation,
                    debug=False,
                )
            stitched_dict = patchwise_stitched_reconstruction_with_separation(
                model=self._model,
                past_values=batch_x,
                patch_size=self._model.config.patch_length,
                keys_to_stitch=keys_to_stitch,
                keys_to_aggregate=[],
                reconstruct_start=reconstruct_start,
                reconstruct_end=reconstruct_end,
                separation=separation,
                debug=False,
            )
            if isinstance(stitched_dict, tuple):
                stitched_dict = stitched_dict[0]

        # Get desired output from TSPulse outputs
        # output shape: [batch_size, window_size, n_channels]
        scores = OrderedDict()

        reduction_axis = [1]
        if use_ts:
            # time reconstruction
            output = stitched_dict["reconstruction_outputs"]
            pointwise_score = anomaly_criterion(
                batch_x[:, reconstruct_start:reconstruct_end, :],
                output[:, reconstruct_start:reconstruct_end, :],
            )
            if (batch_counter == 0) and (not initialized):
                reconstruct_start_ = 0
                boundary_output = boundary_dict["reconstruction_outputs"]
                boundary_score = anomaly_criterion(
                    batch_x[:1, reconstruct_start_:reconstruct_end, :],
                    boundary_output[:, reconstruct_start_:reconstruct_end, :],
                )
                data_start = boundary_score[0].unfold(0, aggr_win_size, 1).transpose(2, 1)
                pointwise_score = torch.cat([data_start, pointwise_score[1:, :, :]], dim=0)

            scores[AnomalyScoreMethods.TIME_RECONSTRUCTION.value] = torch.mean(pointwise_score, dim=reduction_axis)

        if use_fft:
            # time reconstruction from fft
            output = stitched_dict["reconstructed_ts_from_fft"]
            pointwise_score = anomaly_criterion(
                batch_x[:, reconstruct_start:reconstruct_end, :],
                output[:, reconstruct_start:reconstruct_end, :],
            )
            if (batch_counter == 0) and (not initialized):
                reconstruct_start_ = 0
                boundary_output = boundary_dict["reconstructed_ts_from_fft"]
                boundary_score = anomaly_criterion(
                    batch_x[:1, reconstruct_start_:reconstruct_end, :],
                    boundary_output[:, reconstruct_start_:reconstruct_end, :],
                )
                data_start = boundary_score[0].unfold(0, aggr_win_size, 1).transpose(2, 1)
                pointwise_score = torch.cat([data_start, pointwise_score[1:, :, :]], dim=0)
            scores[AnomalyScoreMethods.FREQUENCY_RECONSTRUCTION.value] = torch.mean(
                pointwise_score, dim=reduction_axis
            )

        if use_forecast:
            # forecast output
            batch_future_values = payload["future_values"]
            output = model_forward_output["forecast_output"]
            pointwise_score = anomaly_criterion(batch_future_values[:, 0, :], output[:, 0, :]).unsqueeze(1)
            scores[AnomalyScoreMethods.PREDICTIVE.value] = torch.mean(pointwise_score, dim=reduction_axis)

        return ModelOutput(scores)

    def adjust_boundary(
        self,
        key: str,
        x: np.ndarray | torch.Tensor | List[np.ndarray | torch.Tensor],
        reference: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Combines model outputs with boundary adjustment.

        Args:
            key (str): key associated with model output.
            x (ScoreListType): model outputs across all batches combined.
            reference (np.ndarray, optional): reference data for score scale adjustment. Defaults to None.

        Returns:
            np.ndarray: combined score
        """
        context_length = self._model.config.context_length
        aggr_win_size = self._aggr_win_size
        score_exponent = self._score_exponent
        if isinstance(x, (list, tuple)):
            if (len(x) > 0) and isinstance(x[0], torch.Tensor):
                x = torch.cat(x, dim=0).detach().cpu().numpy()
            else:
                x = np.concatenate(x, axis=0).astype(float)
        if key == AnomalyScoreMethods.PREDICTIVE.value:
            start_pad_len = context_length
            end_pad_len = 0
        else:
            start_pad_len = context_length - aggr_win_size // 2
            end_pad_len = aggr_win_size // 2

        score = np.array([x[0]] * start_pad_len + list(x) + [x[-1]] * end_pad_len)
        if score.ndim == 1:
            score = score.reshape(-1, 1)

        min_score = 0.0
        if reference is not None:
            reference_data = np.asarray(reference)
            min_score = (
                self._least_significant_scale * np.nanstd(np.diff(reference_data, axis=0), axis=0, keepdims=True) ** 2
            )

            if min_score.shape[-1] != score.shape[-1]:
                min_score = np.nanmax(min_score, axis=-1)
            if key == AnomalyScoreMethods.PREDICTIVE.value:
                min_score = min_score * np.sqrt(2)
            else:
                min_score = min_score * (1 + self._model.config.patch_length)

        score_ = score.copy()
        # score_[np.where(score > min_score)] *= 1 / self._least_significant_score
        # scale = 1 if np.any(score > min_score) else self._least_significant_score
        score = MinMaxScaler_().fit_transform(score_**score_exponent)
        scale = np.ones_like(score_)
        scale[score_ <= min_score] = self._least_significant_score
        score = score * scale
        return score

    def adjust_boundary_(
        self,
        key: str,
        x: ScoreListType,
        reference: Optional[np.ndarray] = None,
        align: Optional[str] = None,
        pad_value: Optional[float] = None,
        state: Optional[dict] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, dict]:
        """Combines model outputs with boundary adjustment.

        Args:
            key (str): key associated with model output.
            x (ScoreListType): model outputs across all batches combined.
            reference (np.ndarray, optional): reference data for score scale adjustment. Defaults to None.
            align (str): takes value (center/left/right) suggesting how to time align the window score, default in 'center'
            pad_value (float): optional field suggesting value used for padding the end values, if not provided forward and backward fill algorithm is applied.
            state (dict): optional parameters, expects dictionary containing required values for scoring using a using state memory.

        Returns:
            np.ndarray: combined score
        """
        context_length = self._model.config.context_length
        aggr_win_size = self._aggr_win_size
        score_exponent = self._score_exponent

        if align is None:
            align = "center"

        if state is None:
            state = {"upper": None, "lower": None}

        running_std = np.asarray(state.get("running_std", 0.0))
        running_size = np.asarray(state.get("running_size", 1))

        if isinstance(x, (list, tuple)):
            if (len(x) > 0) and isinstance(x[0], torch.Tensor):
                x = torch.cat(x, dim=0).detach().cpu().numpy()
            else:
                x = np.concatenate(x, axis=0).astype(float)
        if key == AnomalyScoreMethods.PREDICTIVE.value:
            start_pad_len = context_length
            end_pad_len = 0
        else:
            if align == "center":
                start_pad_len = aggr_win_size // 2
                end_pad_len = aggr_win_size // 2
            elif align == "left":
                start_pad_len = 0
                end_pad_len = aggr_win_size
            elif align == "right":
                start_pad_len = aggr_win_size - 1
                end_pad_len = 1
            else:
                raise ValueError(f"Error: unknown alignment type {align}!")

        if pad_value is None:
            start_pad_value = x[0]
            end_pad_value = x[-1]
        else:
            start_pad_value = np.ones(x[0].shape) * pad_value
            end_pad_value = np.ones(x[-1].shape) * pad_value

        score = np.array([start_pad_value] * start_pad_len + list(x) + [end_pad_value] * end_pad_len)
        if score.ndim == 1:
            score = score.reshape(-1, 1)

        min_score = 0.0
        if reference is not None:
            reference_data = np.asarray(reference)
            curr_std = np.nanstd(np.diff(reference_data, axis=0), axis=0, keepdims=True)
            curr_size = reference_data.shape[0] - 1
            running_variance = ((curr_std**2) * curr_size + (running_std**2) * running_size) / (
                curr_size + running_size
            )
            running_size = curr_size + running_size
            running_std = np.sqrt(running_variance)
            min_score = self._least_significant_scale * running_variance

            if min_score.shape[-1] != score.shape[-1]:
                min_score = np.nanmax(min_score, axis=-1)
            if key == AnomalyScoreMethods.PREDICTIVE.value:
                min_score = min_score * np.sqrt(2)
            else:
                min_score = min_score * (1 + self._model.config.patch_length)

        score_ = score.copy()
        score, state = causal_minmax(score_**score_exponent, **state)
        scale = np.ones_like(score_)
        scale[score_ <= min_score] = self._least_significant_score
        score = score * scale
        state.update(running_std=running_std.tolist(), running_size=running_size.tolist())
        return score, state
