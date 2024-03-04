# Copyright contributors to the TSFM project
#

"""Tests the time series preprocessor and functions"""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from transformers import PatchTSTForPrediction

from tsfm_public.toolkit.time_series_forecasting_pipeline import (
    TimeSeriesForecastingPipeline,
    augment_time_series,
)
from tsfm_public.toolkit.util import select_by_index

from ..util import nreps


@pytest.fixture(scope="module")
def ts_data():
    df = pd.DataFrame(
        {
            "id": nreps(["A", "B", "C"], 50),
            "id2": nreps(["XX", "YY", "ZZ"], 50),
            "timestamp": [datetime(2021, 1, 1) + timedelta(days=i) for i in range(50)]
            * 3,
            "value1": range(150),
            "value2": np.arange(150) / 3 + 10,
        }
    )
    return df


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
    )

    dataset_path = (
        "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv"
    )
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
    assert (
        forecasts_no_future.iloc[0]["OT_prediction"]
        == forecasts.iloc[0]["OT_prediction"]
    )


def test_augment_time_series(ts_data):

    periods = 5
    a = augment_time_series(
        ts_data, timestamp_column="timestamp", grouping_columns=["id"], periods=periods
    )

    # check that length increases by periods for each id
    assert a.shape[0] == ts_data.shape[0] + 3 * periods
    assert a.shape[1] == ts_data.shape[1]

    periods = 3
    a = augment_time_series(
        ts_data,
        timestamp_column="timestamp",
        grouping_columns=["id", "id2"],
        periods=periods,
    )

    # check that length increases by periods for each id
    assert a.shape[0] == ts_data.shape[0] + 3 * periods
    assert a.shape[1] == ts_data.shape[1]

    1
