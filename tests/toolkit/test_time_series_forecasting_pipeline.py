# Copyright contributors to the TSFM project
#

"""Tests the time series preprocessor and functions"""

from typing import Any

import numpy as np
import pandas as pd
import pytest
from transformers import PatchTSTConfig, PatchTSTForPrediction

from tsfm_public import (
    PatchTSTFMForPrediction,
    TinyTimeMixerConfig,
    TinyTimeMixerForDecomposedPrediction,
    TinyTimeMixerForPrediction,
)
from tsfm_public.models.flowstate import FlowStateConfig, FlowStateForPrediction
from tsfm_public.models.patchtst_fm import PatchTSTFMConfig
from tsfm_public.toolkit.conformal import (
    PostHocProbabilisticMethod,
    PostHocProbabilisticProcessor,
)
from tsfm_public.toolkit.time_series_forecasting_pipeline import (
    TimeSeriesForecastingPipeline,
)
from tsfm_public.toolkit.time_series_preprocessor import (
    DEFAULT_FREQUENCY_MAPPING,
    TimeSeriesPreprocessor,
)
from tsfm_public.toolkit.util import select_by_index


@pytest.fixture(scope="module")
def ttm_dummy_model(conf=None):
    # model_path = "ibm-granite/granite-timeseries-ttm-v1"

    if conf is None:
        conf = TinyTimeMixerConfig()
    model = TinyTimeMixerForPrediction(conf)

    return model


@pytest.fixture(scope="module")
def ttm_probabilistic_dummy_model(conf=None):
    # model_path = "ibm-granite/granite-timeseries-ttm-v1"

    if conf is None:
        conf = TinyTimeMixerConfig(multi_quantile_head=True)
    model = TinyTimeMixerForPrediction(conf)

    return model


@pytest.fixture(scope="module")
def flowstate_dummy_model(conf=None):
    """
    Create a dummy FlowState model for testing purposes.

    Args:
        conf: Optional FlowStateConfig. If None, uses default configuration.

    Returns:
        FlowStateForPrediction: A FlowState model instance for testing.
    """
    if conf is None:
        conf = FlowStateConfig()
    model = FlowStateForPrediction(conf)

    return model


@pytest.fixture(scope="module")
def patchtst_fm_dummy_model(conf=None):
    """
    Create a dummy PatchTST-FM model for testing purposes.

    Args:
        conf: Optional PatchTSTFMConfig. If None, uses default configuration.

    Returns:
        PatchTSTFMForPrediction: A PatchTST-FM model instance for testing.
    """
    if conf is None:
        conf = PatchTSTFMConfig()
    model = PatchTSTFMForPrediction(conf)

    return model


@pytest.fixture(scope="module")
def random_sine_wave_data():
    examples = 5
    context_length = 512 + examples - 1
    num_channels = 2
    freq = "h"
    timestamps = pd.date_range(start="2024-01-01", periods=context_length, freq=freq)

    # Create synthetic data
    t = np.linspace(0, 4 * np.pi, context_length)

    data: dict[str, Any] = {"timestamp": timestamps}

    for ch in range(num_channels):
        # Combine sine waves with different frequencies
        freq1 = 5 * (1.0 + ch * 0.2)
        freq2 = 5 * (2.0 + ch * 0.3)
        phase = ch * 0.5

        series = (
            np.sin(freq1 * t + phase)
            + 0.5 * np.sin(freq2 * t + phase * 2)
            + 0.1 * np.random.randn(context_length)  # Add noise
        )

        data[f"target_{ch}"] = series

    df = pd.DataFrame(data)
    return df


# set up dummy models for patchtst-fm, flowstate, ttm
# confirm that any passthrough parameters are actually passed during the forward call

# ALL: prediction_length
# FS: prediction_type, scale_factor
# PatchTSTFM: quantile_levels
# TTM: freq_token

# define the combinations of models and parameters we want to test
testable_param_map = {
    "ttm_dummy_model": ["freq_token"],
    "flowstate_dummy_model": ["prediction_length", "prediction_type", "scale_factor"],
    "patchtst_fm_dummy_model": ["prediction_length", "quantile_levels"],
}


# this fixture couples a model with a parameter to test
@pytest.fixture(
    scope="module",
    params=((model, param) for model, params in testable_param_map.items() for param in params),
    ids=(f"{model}-{param}" for model, params in testable_param_map.items() for param in params),
)
def model_param(request):
    model, param = request.param
    yield request.getfixturevalue(model), param


def test_models_parameters_in_forecasting_pipeline(random_sine_wave_data, model_param):
    model, param = model_param

    target_columns = ["target_0"]  # , "target_1"]
    freq = "h"

    if param == "prediction_length":
        test_param = {"prediction_length": 7}
    elif param == "freq_token":
        test_param = {"freq_token": "10m"}
    elif param == "prediction_type":
        test_param = {"prediction_type": "median"}
    elif param == "scale_factor":
        test_param = {"scale_factor": 2}
    elif param == "quantile_levels":
        test_param = {"quantile_levels": [0.1, 0.5, 0.9]}

    forecast_pipeline = TimeSeriesForecastingPipeline(
        model=model,
        timestamp_column="timestamp",
        id_columns=[],  # No ID columns for single series
        target_columns=target_columns,
        freq=freq,
        context_length=model.config.context_length,
        **test_param,
    )

    forecasts = forecast_pipeline(random_sine_wave_data)
    if param == "prediction_length":
        assert len(forecasts[target_columns[0]].iloc[0]) == test_param["prediction_length"]

    if param == "quantile_levels":
        num_quantiles = len(test_param["quantile_levels"])
        assert (
            forecasts.shape[1] == 1 + 2 * len(target_columns) + len(target_columns) * num_quantiles
        ), f"Number of expetect columns does not match. Received: {forecasts.shape[1]} Expected {1 + 2 * len(target_columns) + len(target_columns) * num_quantiles}"


def test_forecasting_pipeline_defaults():
    model = PatchTSTForPrediction(PatchTSTConfig(prediction_length=3, context_length=33))

    tspipe = TimeSeriesForecastingPipeline(model)

    assert tspipe._preprocess_params["prediction_length"] == 3
    assert tspipe._preprocess_params["context_length"] == 33

    tspipe = TimeSeriesForecastingPipeline(model=model, prediction_length=6, context_length=66)

    assert tspipe._preprocess_params["prediction_length"] == 6
    assert tspipe._preprocess_params["context_length"] == 66


def test_forecasting_pipeline_forecasts(patchtst_base_model, etth_data_base):
    timestamp_column = "date"
    id_columns = []
    target_columns = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
    prediction_length = 96

    model = patchtst_base_model
    context_length = model.config.context_length

    forecast_pipeline = TimeSeriesForecastingPipeline(
        model=model,
        timestamp_column=timestamp_column,
        id_columns=id_columns,
        target_columns=target_columns,
        freq="1h",
    )

    test_end_index = 12 * 30 * 24 + 8 * 30 * 24
    test_start_index = test_end_index - context_length

    data = etth_data_base.copy()

    test_data = select_by_index(
        data,
        id_columns=id_columns,
        start_index=test_start_index,
        end_index=test_end_index,
    )

    delta = test_data[timestamp_column].iloc[-1] - test_data[timestamp_column].iloc[-2]
    df_fut = pd.DataFrame(
        {
            timestamp_column: pd.date_range(
                test_data[timestamp_column].iloc[-1] + delta,
                freq=delta,
                periods=prediction_length,
            )
        }
    )

    # base case
    forecasts = forecast_pipeline(test_data, future_time_series=df_fut)
    assert forecasts.shape == (1, 2 * len(target_columns) + 1)

    # when we provide no data for future time series, we do internal augmentation
    forecasts_no_future = forecast_pipeline(test_data)
    assert forecasts_no_future.shape == (1, 2 * len(target_columns) + 1)

    # check forecasts match
    assert forecasts_no_future.iloc[0]["OT_prediction"] == forecasts.iloc[0]["OT_prediction"]

    # test that forecasts are properly exploded
    forecast_pipeline = TimeSeriesForecastingPipeline(
        model=model,
        timestamp_column=timestamp_column,
        id_columns=id_columns,
        target_columns=target_columns,
        freq="1h",
        explode_forecasts=True,
    )

    forecasts_exploded = forecast_pipeline(test_data)
    assert forecasts_exploded.shape == (prediction_length, len(target_columns) + 1)

    forecast_pipeline = TimeSeriesForecastingPipeline(
        model=model,
        timestamp_column=timestamp_column,
        id_columns=id_columns,
        target_columns=target_columns,
        freq="1h",
        batch_size=10,
    )

    test_end_index = 12 * 30 * 24 + 8 * 30 * 24
    test_start_index = test_end_index - context_length - 9

    data = etth_data_base.copy()

    test_data = select_by_index(
        data,
        id_columns=id_columns,
        start_index=test_start_index,
        end_index=test_end_index,
    )
    forecasts = forecast_pipeline(test_data)
    assert forecast_pipeline._batch_size == 10
    assert forecasts.shape == (10, 2 * len(target_columns) + 1)


def test_ttm_native_probabilistc_forecasting_pipeline(etth_data_base):
    pfl = 10
    conf = TinyTimeMixerConfig(prediction_filter_length=pfl, multi_quantile_head=True)
    model = TinyTimeMixerForPrediction(config=conf)
    quantile_levels = conf.quantile_levels
    timestamp_column = "date"
    id_columns = []
    target_columns = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]

    context_length = model.config.context_length

    test_end_index = 12 * 30 * 24 + 8 * 30 * 24
    test_start_index = test_end_index - context_length

    forecast_pipeline = TimeSeriesForecastingPipeline(
        model=model,
        timestamp_column=timestamp_column,
        id_columns=id_columns,
        target_columns=target_columns,
        freq="1h",
        batch_size=10,
    )

    test_end_index = 12 * 30 * 24 + 8 * 30 * 24
    test_start_index = test_end_index - context_length - 9

    data = etth_data_base.copy()

    test_data = select_by_index(
        data,
        id_columns=id_columns,
        start_index=test_start_index,
        end_index=test_end_index,
    )
    forecasts = forecast_pipeline(test_data)

    print(forecasts.keys())
    assert len(forecasts[f"{target_columns[0]}"]) == pfl
    for i in quantile_levels:
        assert len(forecasts[f"{target_columns[0]}_prediction_q{i}"]) == pfl


def test_ttm_decomposed_native_probabilistc_forecasting_pipeline(etth_data_base):
    pfl = 10
    conf = TinyTimeMixerConfig(prediction_filter_length=pfl, multi_quantile_head=True)
    model = TinyTimeMixerForDecomposedPrediction(config=conf)
    quantile_levels = conf.quantile_levels
    timestamp_column = "date"
    id_columns = []
    target_columns = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]

    context_length = model.config.context_length

    test_end_index = 12 * 30 * 24 + 8 * 30 * 24
    test_start_index = test_end_index - context_length

    forecast_pipeline = TimeSeriesForecastingPipeline(
        model=model,
        timestamp_column=timestamp_column,
        id_columns=id_columns,
        target_columns=target_columns,
        freq="1h",
        batch_size=10,
    )

    test_end_index = 12 * 30 * 24 + 8 * 30 * 24
    test_start_index = test_end_index - context_length - 9

    data = etth_data_base.copy()

    test_data = select_by_index(
        data,
        id_columns=id_columns,
        start_index=test_start_index,
        end_index=test_end_index,
    )
    forecasts = forecast_pipeline(test_data)

    print(forecasts.keys())
    assert len(forecasts[f"{target_columns[0]}"]) == pfl
    for i in quantile_levels:
        assert len(forecasts[f"{target_columns[0]}_prediction_q{i}"]) == pfl


def test_forecasting_pipeline_forecasts_with_preprocessor(patchtst_base_model, etth_data_base):
    timestamp_column = "date"
    id_columns = []
    target_columns = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
    prediction_length = 96

    model = patchtst_base_model
    context_length = model.config.context_length

    data = etth_data_base.copy()
    train_end_index = 12 * 30 * 24

    test_end_index = 12 * 30 * 24 + 8 * 30 * 24
    test_start_index = test_end_index - context_length - 4

    train_data = select_by_index(
        data,
        id_columns=id_columns,
        start_index=0,
        end_index=train_end_index,
    )
    test_data = select_by_index(
        data,
        id_columns=id_columns,
        start_index=test_start_index,
        end_index=test_end_index,
    )

    tsp = TimeSeriesPreprocessor(
        timestamp_column=timestamp_column,
        id_columns=id_columns,
        target_columns=target_columns,
        context_length=context_length,
        prediction_length=prediction_length,
        freq="1h",
        scaling=True,
    )

    tsp.train(train_data)

    forecast_pipeline = TimeSeriesForecastingPipeline(
        model=model,
        timestamp_column=timestamp_column,
        id_columns=id_columns,
        target_columns=target_columns,
        freq="1h",
        feature_extractor=tsp,
        explode_forecasts=False,
        inverse_scale_outputs=True,
    )

    forecasts = forecast_pipeline(test_data)

    assert forecasts.shape == (
        test_end_index - test_start_index - context_length + 1,
        2 * len(target_columns) + 1,
    )

    # if we have inverse scaled mean should be larger
    assert forecasts["HUFL_prediction"].mean().mean() > 10


def test_frequency_token(ttm_dummy_model, etth_data):
    model = ttm_dummy_model
    train_data, test_data, params = etth_data

    timestamp_column = params["timestamp_column"]
    id_columns = params["id_columns"]
    target_columns = params["target_columns"]
    prediction_length = params["prediction_length"]
    context_length = params["context_length"]

    tsp = TimeSeriesPreprocessor(
        timestamp_column=timestamp_column,
        id_columns=id_columns,
        target_columns=target_columns,
        context_length=context_length,
        prediction_length=prediction_length,
        freq="1h",
        scaling=True,
    )

    tsp.train(train_data)

    assert model.config.resolution_prefix_tuning is False

    forecast_pipeline = TimeSeriesForecastingPipeline(
        model=model,
        timestamp_column=timestamp_column,
        id_columns=id_columns,
        target_columns=target_columns,
        freq="1h",
        feature_extractor=tsp,
        explode_forecasts=False,
        inverse_scale_outputs=True,
    )
    assert forecast_pipeline._preprocess_params["frequency_token"] is None

    model.config.resolution_prefix_tuning = True
    forecast_pipeline = TimeSeriesForecastingPipeline(
        model=model,
        timestamp_column=timestamp_column,
        id_columns=id_columns,
        target_columns=target_columns,
        freq="1h",
        feature_extractor=tsp,
        explode_forecasts=False,
        inverse_scale_outputs=True,
    )
    assert forecast_pipeline._preprocess_params["frequency_token"] == DEFAULT_FREQUENCY_MAPPING["h"]

    with pytest.raises(ValueError):
        forecast_pipeline = TimeSeriesForecastingPipeline(
            model=model,
            timestamp_column=timestamp_column,
            id_columns=id_columns,
            target_columns=target_columns,
            freq="1h",
            explode_forecasts=False,
            inverse_scale_outputs=True,
        )


def test_prediction_filter_length(etth_data):
    pfl = 10
    conf = TinyTimeMixerConfig(prediction_filter_length=pfl)
    model = TinyTimeMixerForPrediction(config=conf)
    train_data, test_data, params = etth_data

    timestamp_column = params["timestamp_column"]
    id_columns = params["id_columns"]
    target_columns = params["target_columns"]
    prediction_length = params["prediction_length"]
    context_length = params["context_length"]

    tsp = TimeSeriesPreprocessor(
        timestamp_column=timestamp_column,
        id_columns=id_columns,
        target_columns=target_columns,
        context_length=context_length,
        prediction_length=prediction_length,
        freq="1h",
        scaling=True,
    )

    tsp.train(train_data)

    assert model.config.prediction_filter_length == pfl

    forecast_pipeline = TimeSeriesForecastingPipeline(
        model=model,
        feature_extractor=tsp,
        explode_forecasts=False,
        inverse_scale_outputs=True,
        device="cpu",
    )
    forecasts = forecast_pipeline(train_data.iloc[:200])

    assert len(forecasts[f"{target_columns[0]}_prediction"].iloc[0]) == pfl


def test_probabilistic_forecasts(etth_data):
    train_data, test_data, params = etth_data
    pfl = 10
    conf = TinyTimeMixerConfig(prediction_filter_length=pfl)
    model = TinyTimeMixerForPrediction(config=conf)

    timestamp_column = params["timestamp_column"]
    id_columns = params["id_columns"]
    target_columns = params["target_columns"]
    context_length = conf.context_length
    prediction_length = conf.prediction_length

    tsp = TimeSeriesPreprocessor(
        timestamp_column=timestamp_column,
        id_columns=id_columns,
        target_columns=target_columns,
        context_length=context_length,
        prediction_length=prediction_length,
        freq="1h",
        scaling=True,
    )

    tsp.train(train_data)

    assert model.config.prediction_filter_length == pfl

    forecast_pipeline = TimeSeriesForecastingPipeline(
        model=model,
        feature_extractor=tsp,
        explode_forecasts=False,
        inverse_scale_outputs=True,
        device="cpu",
        batch_size=100,
    )

    forecasts_cal = forecast_pipeline(train_data.iloc[-200:])

    conformal = PostHocProbabilisticProcessor(
        window=100,
        quantiles=[0.1, 0.9],
        method=PostHocProbabilisticMethod.CONFORMAL.value,
    )

    # prepare calibration data
    prediction_columns = [f"{c}_prediction" for c in target_columns]
    ground_truth_columns = target_columns
    y = forecasts_cal[prediction_columns].values
    predictions_cal = np.array([np.stack(z) for z in y]).transpose(0, 2, 1)
    predictions_cal = predictions_cal[:-prediction_length, ...]
    x = forecasts_cal[ground_truth_columns].values
    ground_truth_cal = np.array([np.stack(z) for z in x]).transpose(0, 2, 1)
    ground_truth_cal = ground_truth_cal[:-prediction_length, :pfl, ...]

    conformal.train(ground_truth_cal, predictions_cal)

    # generate forecasts on test data with conformal bounds
    forecast_pipeline = TimeSeriesForecastingPipeline(
        model=model,
        feature_extractor=tsp,
        explode_forecasts=False,
        inverse_scale_outputs=True,
        device="cpu",
        probabilistic_processor=conformal,
        batch_size=100,
    )

    forecasts = forecast_pipeline(test_data)

    assert len(forecasts[f"{target_columns[0]}_prediction"].iloc[0]) == pfl
    assert len(forecasts[f"{target_columns[0]}_prediction_q{conformal.quantiles[0]}"].iloc[0]) == pfl
    assert len(forecasts[f"{target_columns[0]}_prediction_q{conformal.quantiles[1]}"].iloc[0]) == pfl

    # with explode
    forecast_pipeline = TimeSeriesForecastingPipeline(
        model=model,
        feature_extractor=tsp,
        explode_forecasts=True,
        inverse_scale_outputs=True,
        device="cpu",
        probabilistic_processor=conformal,
        batch_size=100,
    )

    forecasts = forecast_pipeline(test_data[-context_length:])

    assert len(forecasts[f"{target_columns[0]}"]) == pfl
    assert len(forecasts[f"{target_columns[0]}_q{conformal.quantiles[0]}"]) == pfl
    assert len(forecasts[f"{target_columns[0]}_q{conformal.quantiles[1]}"]) == pfl
