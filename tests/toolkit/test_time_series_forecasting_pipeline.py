# Copyright contributors to the TSFM project
#

"""Tests the time series preprocessor and functions"""
import pandas as pd
from transformers import PatchTSTForPrediction

from tsfm_public.toolkit.time_series_forecasting_pipeline import (
    TimeSeriesForecastingPipeline,
)
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor
from tsfm_public.toolkit.util import select_by_index


def test_forecasting_pipeline_forecasts():
    timestamp_column = "date"
    id_columns = []
    target_columns = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
    prediction_length = 96

    model_path = "ibm/patchtst-etth1-forecasting"
    model = PatchTSTForPrediction.from_pretrained(model_path)
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


def test_forecasting_pipeline_forecasts_with_preprocessor():
    timestamp_column = "date"
    id_columns = []
    target_columns = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
    prediction_length = 96

    model_path = "ibm/patchtst-etth1-forecasting"
    model = PatchTSTForPrediction.from_pretrained(model_path)
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

    forecasts = forecast_pipeline(tsp.preprocess(test_data))

    assert forecasts.shape == (
        test_end_index - test_start_index - context_length + 1,
        2 * len(target_columns) + 1,
    )

    # if we have inverse scaled mean should be larger
    assert forecasts["HUFL_prediction"].mean().mean() > 10
