# Copyright contributors to the TSFM project
#

"""Tests for QuantileEnsembleForecaster."""

import numpy as np
import pandas as pd
import torch

from tsfm_public.toolkit.forecasters import (
    DEFAULT_QUANTILE_LEVELS,
    ForecastResult,
    PatchTSTFMDataFramePipelineForecaster,
    FlowStateDataFramePipelineForecaster,
    TinyTimeMixerDataFramePipelineForecaster
)
from tsfm_public.models.ensemble.modeling_ensemble import QuantileEnsembleForecaster
from tsfm_public.toolkit.ensemble_aggregation import aggregate_linear_pool


PREDICTION_LENGTH = 24
CONTEXT_LENGTH = 90
N_QUANTILES = len(DEFAULT_QUANTILE_LEVELS)


def test_ensemble_call():
    """QuantileEnsembleForecaster.__call__ should return a valid ForecastResult."""
    # Synthetic sinusoidal data: 91 hourly points, 1 target
    n = 91
    t = np.arange(n)
    y = np.sin(2 * np.pi * t / 24)
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
            "target": y,
        }
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ensemble = QuantileEnsembleForecaster(
        members=[
            PatchTSTFMDataFramePipelineForecaster(
                model_checkpoint="ibm-research/patchtst-fm-r1",
                device=device,
            ),
            PatchTSTFMDataFramePipelineForecaster(
                model_checkpoint="ibm-granite/granite-timeseries-patchtst-fm-r1",
                device=device,
            ),
        ],
        quantile_levels=DEFAULT_QUANTILE_LEVELS,
        ensemble_function=aggregate_linear_pool,
    )

    result = ensemble(
        data,
        timestamp_column="timestamp",
        target_columns=["target"],
        prediction_length=PREDICTION_LENGTH,
        context_length=CONTEXT_LENGTH,
    )

    assert isinstance(result, ForecastResult)
    assert result.success
    # predicted_quantiles: (n_samples=2, pred_len=24, n_targets=1, n_quantiles=9)
    assert result.predicted_quantiles is not None
    arr = np.array(result.predicted_quantiles)
    assert arr.shape == (2, PREDICTION_LENGTH, 1, N_QUANTILES)
    # predicted (median): (n_samples=2, pred_len=24, n_targets=1)
    assert result.predicted is not None
    assert np.array(result.predicted).shape == (2, PREDICTION_LENGTH, 1)




def test_3model_ensemble():
    """Run QuantileEnsembleForecaster using PatchTST-FM, FlowState and TTM models"""
    # Synthetic sinusoidal data: 91 hourly points, 1 target
    n = 91
    t = np.arange(n)
    y = np.sin(2 * np.pi * t / 24)
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
            "target": y,
        }
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ensemble = QuantileEnsembleForecaster(
        members=[
            PatchTSTFMDataFramePipelineForecaster(
                model_checkpoint="ibm-granite/granite-timeseries-patchtst-fm-r1",
                device=device,
            ),
            FlowStateDataFramePipelineForecaster(
                model_checkpoint="ibm-granite/granite-timeseries-flowstate-r1",
                device = device
            ),
            TinyTimeMixerDataFramePipelineForecaster(
                model_checkpoint="ibm-granite/granite-timeseries-ttm-r3",
                device = device
            )
        ],
        quantile_levels=DEFAULT_QUANTILE_LEVELS,
        ensemble_function=aggregate_linear_pool,
    )

    result = ensemble(
        data,
        timestamp_column="timestamp",
        target_columns=["target"],
        prediction_length=PREDICTION_LENGTH,
        context_length=CONTEXT_LENGTH,
    )

    assert isinstance(result, ForecastResult)
    assert result.success
    # predicted_quantiles: (n_samples=2, pred_len=24, n_targets=1, n_quantiles=9)
    assert result.predicted_quantiles is not None
    arr = np.array(result.predicted_quantiles)
    assert arr.shape == (2, PREDICTION_LENGTH, 1, N_QUANTILES)
    # predicted (median): (n_samples=2, pred_len=24, n_targets=1)
    assert result.predicted is not None
    assert np.array(result.predicted).shape == (2, PREDICTION_LENGTH, 1)
