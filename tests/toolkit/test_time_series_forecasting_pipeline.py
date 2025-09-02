# Copyright contributors to the TSFM project
#

"""Tests the time series preprocessor and functions"""

import numpy as np
import pandas as pd
import pytest
from transformers import PatchTSTConfig, PatchTSTForPrediction

from tsfm_public import TinyTimeMixerConfig, TinyTimeMixerForPrediction
from tsfm_public.toolkit.conformal import PostHocProbabilisticMethod, PostHocProbabilisticProcessor
from tsfm_public.toolkit.time_series_forecasting_pipeline import (
    TimeSeriesForecastingPipeline,
)
from tsfm_public.toolkit.time_series_preprocessor import DEFAULT_FREQUENCY_MAPPING, TimeSeriesPreprocessor
from tsfm_public.toolkit.util import select_by_index


@pytest.fixture(scope="module")
def ttm_dummy_model(conf=None):
    # model_path = "ibm-granite/granite-timeseries-ttm-v1"

    if conf is None:
        conf = TinyTimeMixerConfig()
    model = TinyTimeMixerForPrediction(conf)

    return model


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
        model=model, feature_extractor=tsp, explode_forecasts=False, inverse_scale_outputs=True, device="cpu"
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
        window=100, quantiles=[0.1, 0.9], method=PostHocProbabilisticMethod.CONFORMAL.value
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
