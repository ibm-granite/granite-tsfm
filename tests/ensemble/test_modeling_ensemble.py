# Copyright contributors to the TSFM project
#

"""Tests for QuantileEnsembleForecaster."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from tsfm_public.models.ensemble.modeling_ensemble import QuantileEnsembleForecaster
from tsfm_public.toolkit.ensemble_aggregation import aggregate_linear_pool
from tsfm_public.toolkit.forecasters import (
    DEFAULT_QUANTILE_LEVELS,
    FlowStateDataFramePipelineForecaster,
    ForecastResult,
    PatchTSTFMDataFramePipelineForecaster,
    TinyTimeMixerDataFramePipelineForecaster,
)


PREDICTION_LENGTH = 24
CONTEXT_LENGTH = 90
N_QUANTILES = len(DEFAULT_QUANTILE_LEVELS)


def test_patchtst_forecaster_default_and_load_options():
    with patch("tsfm_public.toolkit.forecasters.PatchTSTFMForPrediction.from_pretrained") as load:
        member = PatchTSTFMDataFramePipelineForecaster(device="cpu", local_files_only=True)

    load.assert_called_once_with("ibm-granite/granite-timeseries-patchtst-fm-r2", local_files_only=True)
    assert member.model is load.return_value


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
                model_checkpoint="ibm-granite/granite-timeseries-flowstate-r1", device=device
            ),
            TinyTimeMixerDataFramePipelineForecaster(
                model_checkpoint="ibm-granite/granite-timeseries-ttm-r3", device=device
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


def test_issue558_example():
    """The example from the issue now runs without dropping ensemble members b/c the TTM Forecaster is fixed."""

    ens = QuantileEnsembleForecaster.from_pretrained("ibm-granite/granite-timeseries-ensemble-r1", device="cpu")
    t = pd.date_range("2026-01-01", periods=512, freq="30min", tz="UTC")
    df = pd.DataFrame({"timestamp": t, "y": -9 + 0.4 * np.sin(np.arange(512) * 2 * np.pi / 48)})

    r = ens(df, timestamp_column="timestamp", target_columns=["y"], context_length=512, prediction_length=1)

    assert r.success
    assert r.metadata["n_models"] == 4, "None of the members get dropped."
    assert len(r.metadata["ensemble_members"]) == 4


def test_issue558_nan_forecasts():
    """QuantileEnsembleForecaster warns when dropping an ensemble member due to NA forecasts. See issue 558."""

    # Default Ensemble
    ens = QuantileEnsembleForecaster.from_pretrained("ibm-granite/granite-timeseries-ensemble-r1", device="cpu")
    # TTM forecaster that gives nan forecasts
    TTMforecaster768 = TinyTimeMixerDataFramePipelineForecaster(model_revision="768-48-dec-512-r3", device="cpu")
    # update ensemble with this forecaster
    ens.members[3] = TTMforecaster768

    # sample data
    t = pd.date_range("2026-01-01", periods=512, freq="30min", tz="UTC")
    df = pd.DataFrame({"timestamp": t, "y": -9 + 0.4 * np.sin(np.arange(512) * 2 * np.pi / 48)})

    # ensemble emits a warning
    with pytest.warns(UserWarning, match="Some members did not produce forecasts"):
        r = ens(df, timestamp_column="timestamp", target_columns=["y"], context_length=512, prediction_length=1)

    assert r.success
    assert r.metadata["n_models"] == 3, "TTM gives NA forecasts and should be dropped"
    assert len(r.metadata["ensemble_members"]) == 3
