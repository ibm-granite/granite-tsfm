# Copyright contributors to the TSFM project
#
"""Helper class for anomaly detection support"""

from collections import OrderedDict
from typing import List, Optional

import numpy as np
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

from .helpers import patchwise_stitched_reconstruction


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

    def adjust_boundary(
        self,
        key: str,
        x: ScoreListType,
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
                x = torch.cat(x, axis=0).detach().cpu().numpy()
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
        score_[np.where(score > min_score)] *= 1 / self._least_significant_score
        scale = 1 if np.any(score > min_score) else self._least_significant_score
        score = MinMaxScaler_().fit_transform(score_**score_exponent) * scale
        return score
