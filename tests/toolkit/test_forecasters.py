# Copyright contributors to the TSFM project
#

"""Tests for PatchTSTFMDataFramePipelineForecaster."""

import numpy as np
import pandas as pd
import pytest
import torch

from tsfm_public.models.tinytimemixer import TinyTimeMixerForDecomposedPrediction
from tsfm_public.toolkit.forecasters import (
    DEFAULT_QUANTILE_LEVELS,
    FlowStateDataFramePipelineForecaster,
    ForecastHasNA,
    PatchTSTFMDataFramePipelineForecaster,
    TinyTimeMixerDataFramePipelineForecaster,
)


MODEL_CHECKPOINT = "ibm-research/patchtst-fm-r1"
PREDICTION_LENGTH = 24
CONTEXT_LENGTH = 90
N_QUANTILES = len(DEFAULT_QUANTILE_LEVELS)


@pytest.fixture(scope="module")
def forecaster():
    """Load model once for all tests in this module."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return PatchTSTFMDataFramePipelineForecaster(
        model_checkpoint=MODEL_CHECKPOINT,
        device=device,
    )


@pytest.fixture(scope="module")
def ts_data():
    """Synthetic sinusoidal time series — 91 hourly points, 1 target."""
    n = 91
    t = np.arange(n)
    y = np.sin(2 * np.pi * t / 24)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
            "target": y,
        }
    )


def _calc_num_cutoffs(data_length: int, context_length: int):
    """Calcs number of forecast cutoffs aka rolling windows.

    When data_length exceeds context_length, TimeSeriesForecastingPipeline
    computes rolling forecasts using historical values of size context_length
    for as many windows as possible given the data.

    In the case of TTM, the context_length attribute of the model may exceed the data_length.
    So in those cases the Max() function is needed to calc the number of cutoffs.
    """
    return max(data_length - context_length + 1, 1)


def test_call_returns_dataframe(forecaster, ts_data):
    """__call__ should return a DataFrame with prediction and quantile columns."""
    forecast = forecaster(
        data=ts_data,
        timestamp_column="timestamp",
        target_columns=["target"],
        prediction_length=PREDICTION_LENGTH,
        context_length=CONTEXT_LENGTH,
    )

    assert isinstance(forecast, pd.DataFrame)
    assert "timestamp" in forecast.columns
    assert "target" in forecast.columns
    assert "target_prediction" in forecast.columns
    # Check quantile columns are present for all default levels
    for q in DEFAULT_QUANTILE_LEVELS:
        assert f"target_prediction_q{q}" in forecast.columns

    # 91 rows with context_length=90 produces 2 samples
    assert len(forecast) == _calc_num_cutoffs(data_length=ts_data.shape[0], context_length=CONTEXT_LENGTH)


def test_forecast_in_ForecastResult(forecaster, ts_data, type_str="patchtst_fm"):
    """forecast_in_ForecastResult should return a valid ForecastResult with correct shapes."""
    result = forecaster.forecast_in_forecast_result(
        data=ts_data,
        timestamp_column="timestamp",
        target_columns=["target"],
        prediction_length=PREDICTION_LENGTH,
        context_length=CONTEXT_LENGTH,
    )

    assert result.success
    # predicted: (n_samples=2, pred_len=24, n_targets=1)
    assert result.predicted is not None

    num_cutoffs = _calc_num_cutoffs(data_length=ts_data.shape[0], context_length=CONTEXT_LENGTH)
    assert len(result.predicted) == num_cutoffs
    assert len(result.predicted[0]) == PREDICTION_LENGTH
    assert len(result.predicted[0][0]) == 1
    # actuals: same shape
    assert result.actuals is not None
    assert len(result.actuals) == num_cutoffs
    # predicted_quantiles: (n_samples=2, pred_len=24, n_targets=1, n_quantiles=9)
    assert result.predicted_quantiles is not None
    assert len(result.predicted_quantiles) == num_cutoffs
    assert len(result.predicted_quantiles[0]) == PREDICTION_LENGTH
    assert len(result.predicted_quantiles[0][0]) == 1
    assert len(result.predicted_quantiles[0][0][0]) == N_QUANTILES
    # cutoff_dates: (n_samples=2,)
    assert result.cutoff_dates is not None
    assert len(result.cutoff_dates) == num_cutoffs
    assert isinstance(result.cutoff_dates[0], str)
    # metadata
    assert result.metadata["model_type"].lower().index(type_str) > 0, "model_type should contain type_str"
    assert result.metadata["prediction_length"] == PREDICTION_LENGTH
    assert result.metadata["context_length"] == CONTEXT_LENGTH
    assert result.metadata["n_targets"] == 1


def test_forecast_for_ensemble(forecaster, ts_data):
    """forecast_for_ensemble should return an ndarray of shape (n_samples, pred_len, n_targets, n_quantiles)."""
    arr = forecaster.forecast_for_ensemble(
        data=ts_data,
        timestamp_column="timestamp",
        target_columns=["target"],
        prediction_length=PREDICTION_LENGTH,
        context_length=CONTEXT_LENGTH,
    )

    assert isinstance(arr, np.ndarray)
    num_cutoffs = _calc_num_cutoffs(data_length=ts_data.shape[0], context_length=CONTEXT_LENGTH)
    assert arr.shape == (num_cutoffs, PREDICTION_LENGTH, 1, N_QUANTILES)


"""Testing FlowState Dataframe Forecaster"""


@pytest.fixture(scope="module")
def FSforecaster():
    """Load model once for all tests in this module."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return FlowStateDataFramePipelineForecaster(device=device)


def test_fs_call_returns_dataframe(FSforecaster, ts_data):
    """__call__ on FlowStateDataFramePipelineForecaster should return a forecast DataFrame."""
    test_call_returns_dataframe(FSforecaster, ts_data)


def test_fs_forecast_in_ForecastResult(FSforecaster, ts_data):
    test_forecast_in_ForecastResult(FSforecaster, ts_data, type_str="flowstate")


def test_fs_forecast_for_ensemble(FSforecaster, ts_data):
    test_forecast_for_ensemble(FSforecaster, ts_data)


def test_fs_forecaster_pred_lens(FSforecaster, ts_data):
    forecast80_10 = FSforecaster(
        data=ts_data,
        timestamp_column="timestamp",
        target_columns=["target"],
        prediction_length=10,
        context_length=80,
    )

    num_cutoffs = _calc_num_cutoffs(data_length=ts_data.shape[0], context_length=80)

    assert forecast80_10.shape[0] == num_cutoffs, "returned data frame has expected number of rows"
    assert len(forecast80_10.target_prediction[0]) == 10, "length of prediction should match expected value"

    forecast80_30 = FSforecaster(
        data=ts_data,
        timestamp_column="timestamp",
        target_columns=["target"],
        prediction_length=30,
        context_length=80,
    )

    num_cutoffs = _calc_num_cutoffs(data_length=ts_data.shape[0], context_length=80)

    assert forecast80_30.shape[0] == num_cutoffs
    assert len(forecast80_30.target_prediction[0]) == 30


"""Testing TTM DataFrame Forecaster"""


@pytest.fixture(scope="module")
def TTMforecaster():
    """Load model once for all tests in this module."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return TinyTimeMixerDataFramePipelineForecaster(device=device, model_revision="90-30-dec-90-r3")


def test_ttm_call_returns_dataframe(TTMforecaster, ts_data):
    """__call__ on TinyTimeMixerDataFramePipelineForecaster should return a forecast DataFrame."""
    test_call_returns_dataframe(TTMforecaster, ts_data)


def test_ttm_forecast_in_ForecastResult(TTMforecaster, ts_data):
    test_forecast_in_ForecastResult(TTMforecaster, ts_data, type_str="tinytimemixer")


def test_ttm_forecast_for_ensemble(TTMforecaster, ts_data):
    test_forecast_for_ensemble(TTMforecaster, ts_data)


def test_ttm_forecaster_different_col_names(TTMforecaster, ts_data):
    df = ts_data

    # change the column names
    df.rename(columns={"timestamp": "ts", "target": "tgt"}, inplace=True)

    pred_len = 10
    result = TTMforecaster(df, timestamp_column="ts", target_columns=["tgt"], prediction_length=pred_len)

    assert isinstance(result, pd.DataFrame), "result should be a data frame"
    assert result.shape[0] == 1, "context_length not specified so full data should be used"
    assert len(result["tgt_prediction"][0]) == pred_len, "df cols name prediction_length should match expected"


def test_ttm_forecaster_loads_decomp_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    TTMforecaster = TinyTimeMixerDataFramePipelineForecaster(model_revision="512-30-dec-90-r3", device=device)
    assert isinstance(TTMforecaster.model, TinyTimeMixerForDecomposedPrediction)


def test_ttm_issue558examples():
    t = pd.date_range("2026-01-01", periods=512, freq="30min", tz="UTC")
    df = pd.DataFrame({"timestamp": t, "y": -9 + 0.4 * np.sin(np.arange(512) * 2 * np.pi / 48)})

    TTMforecaster768 = TinyTimeMixerDataFramePipelineForecaster(model_revision="768-48-dec-512-r3", device="cpu")

    with pytest.warns(UserWarning, match="NA"):
        r = TTMforecaster768(
            df, target_columns=["y"], timestamp_column="timestamp", context_length=512, prediction_length=1
        )
        assert isinstance(r, pd.DataFrame)

    TTMforecaster156 = TinyTimeMixerDataFramePipelineForecaster(model_revision="156-16-dec-52-r3", device="cpu")
    r156 = TTMforecaster156(
        df, target_columns=["y"], timestamp_column="timestamp", context_length=512, prediction_length=1
    )
    assert isinstance(r156, pd.DataFrame)

    TTMforecaster512 = TinyTimeMixerDataFramePipelineForecaster(model_revision="512-30-dec-90-r3", device="cpu")
    r512 = TTMforecaster512(
        df, target_columns=["y"], timestamp_column="timestamp", context_length=512, prediction_length=1
    )
    assert isinstance(r512, pd.DataFrame)


def test_ttm_issue558_v2():
    t = pd.date_range("2026-01-01", periods=512, freq="30min", tz="UTC")
    df = pd.DataFrame({"timestamp": t, "y": -9 + 0.4 * np.sin(np.arange(512) * 2 * np.pi / 48)})
    # why does this not work? Should be same as TTMforecaster156 above
    defaultTTMforecaster = TinyTimeMixerDataFramePipelineForecaster(device="cpu")
    rdef = defaultTTMforecaster(
        df, target_columns=["y"], timestamp_column="timestamp", context_length=512, prediction_length=1
    )
    assert isinstance(rdef, pd.DataFrame)


"""Forecaster error handling tests

Error case: nan values in forecast
Expected behavior for different methods:
    __call__:   nan values in the forecast result in a warning
    forecast_in_forecast_result: success=False, message notes NAs
    forecast_for_ensemble: raises ForecastHasNA exception

PatchTST-FM and FlowState do not seem to generate NaN in the forecast.
In case they do, their error handling behaviour should also be tested here.
"""


def test_ttm_nans():
    """Handling nan forecasts in TTMforecaster"""

    # sample data
    t = pd.date_range("2026-01-01", periods=512, freq="30min", tz="UTC")
    df = pd.DataFrame({"timestamp": t, "y": -9 + 0.4 * np.sin(np.arange(512) * 2 * np.pi / 48)})

    # Forecaster object
    forecaster = TinyTimeMixerDataFramePipelineForecaster(model_revision="768-48-dec-512-r3", device="cpu")

    # __call__ method
    with pytest.warns(expected_warning=UserWarning, match="NA"):
        call_result = forecaster(
            df, target_columns=["y"], timestamp_column="timestamp", context_length=512, prediction_length=1
        )
        assert isinstance(call_result, pd.DataFrame)

    # forecast_in_forecast_result() method
    fifr_result = forecaster.forecast_in_forecast_result(
        df, target_columns=["y"], timestamp_column="timestamp", context_length=512, prediction_length=1
    )
    assert not fifr_result.success
    assert "NA" in fifr_result.message

    # forecast_for_ensemble() method
    with pytest.raises(ForecastHasNA):
        forecaster.forecast_for_ensemble(
            df, target_columns=["y"], timestamp_column="timestamp", context_length=512, prediction_length=1
        )
