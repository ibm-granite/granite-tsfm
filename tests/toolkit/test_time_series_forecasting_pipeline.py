# Copyright contributors to the TSFM project
#

"""Tests the time series preprocessor and functions"""
import pandas as pd
from transformers import PatchTSTForPrediction

from tsfm_public.toolkit.time_series_forecasting_pipeline import (
    TimeSeriesForecastingPipeline,
)
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
