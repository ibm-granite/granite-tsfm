# Copyright contributors to the TSFM project
#

"""Tests the time series preprocessor and functions"""

import pandas as pd
import pytest
from transformers import PatchTSTConfig, PatchTSTForPrediction

from tsfm_public import TinyTimeMixerConfig, TinyTimeMixerForPrediction
from tsfm_public.toolkit.time_series_forecasting_pipeline import (
    TimeSeriesForecastingPipeline,
)
from tsfm_public.toolkit.time_series_preprocessor import DEFAULT_FREQUENCY_MAPPING, TimeSeriesPreprocessor
from tsfm_public.toolkit.util import select_by_index


@pytest.fixture(scope="module")
def patchtst_model():
    model_path = "ibm/test-patchtst"
    model = PatchTSTForPrediction.from_pretrained(model_path)

    return model


@pytest.fixture(scope="module")
def ttm_model():
    # model_path = "ibm-granite/granite-timeseries-ttm-v1"

    conf = TinyTimeMixerConfig()
    model = TinyTimeMixerForPrediction(conf)

    return model


@pytest.fixture(scope="module")
def etth_data():
    timestamp_column = "date"
    id_columns = []
    target_columns = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
    prediction_length = 96

    dataset_path = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv"
    data = pd.read_csv(
        dataset_path,
        parse_dates=[timestamp_column],
    )
    train_end_index = 12 * 30 * 24

    context_length = 512  # model.config.context_length

    test_end_index = 12 * 30 * 24 + 8 * 30 * 24
    test_start_index = test_end_index - context_length - 4

    data = pd.read_csv(
        dataset_path,
        parse_dates=[timestamp_column],
    )

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

    params = {
        "timestamp_column": timestamp_column,
        "id_columns": id_columns,
        "target_columns": target_columns,
        "prediction_length": prediction_length,
        "context_length": context_length,
    }

    return train_data, test_data, params


def test_forecasting_pipeline_defaults():
    model = PatchTSTForPrediction(PatchTSTConfig(prediction_length=3, context_length=33))

    tspipe = TimeSeriesForecastingPipeline(model)

    assert tspipe._preprocess_params["prediction_length"] == 3
    assert tspipe._preprocess_params["context_length"] == 33

    tspipe = TimeSeriesForecastingPipeline(model=model, prediction_length=6, context_length=66)

    assert tspipe._preprocess_params["prediction_length"] == 6
    assert tspipe._preprocess_params["context_length"] == 66


def test_forecasting_pipeline_forecasts(patchtst_model):
    timestamp_column = "date"
    id_columns = []
    target_columns = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
    prediction_length = 96

    model = patchtst_model
    context_length = model.config.context_length

    forecast_pipeline = TimeSeriesForecastingPipeline(
        model=model,
        timestamp_column=timestamp_column,
        id_columns=id_columns,
        target_columns=target_columns,
        freq="1h",
    )

    dataset_path = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv"
    test_end_index = 12 * 30 * 24 + 8 * 30 * 24
    test_start_index = test_end_index - context_length

    data = pd.read_csv(
        dataset_path,
        parse_dates=[timestamp_column],
    )

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

    dataset_path = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv"
    test_end_index = 12 * 30 * 24 + 8 * 30 * 24
    test_start_index = test_end_index - context_length - 9

    data = pd.read_csv(
        dataset_path,
        parse_dates=[timestamp_column],
    )

    test_data = select_by_index(
        data,
        id_columns=id_columns,
        start_index=test_start_index,
        end_index=test_end_index,
    )
    forecasts = forecast_pipeline(test_data)
    assert forecast_pipeline._batch_size == 10
    assert forecasts.shape == (10, 2 * len(target_columns) + 1)


def test_forecasting_pipeline_forecasts_with_preprocessor(patchtst_model):
    timestamp_column = "date"
    id_columns = []
    target_columns = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
    prediction_length = 96

    model = patchtst_model
    context_length = model.config.context_length

    dataset_path = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv"
    data = pd.read_csv(
        dataset_path,
        parse_dates=[timestamp_column],
    )
    train_end_index = 12 * 30 * 24

    test_end_index = 12 * 30 * 24 + 8 * 30 * 24
    test_start_index = test_end_index - context_length - 4

    data = pd.read_csv(
        dataset_path,
        parse_dates=[timestamp_column],
    )

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


def test_frequency_token(ttm_model, etth_data):
    model = ttm_model
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
