# Copyright contributors to the TSFM project
#

"""Tests basic dataset functions"""


from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from tsfm_public.toolkit.dataset import (
    ForecastDFDataset,
    PretrainDFDataset,
    ts_padding,
)
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor

from ..util import nreps


@pytest.fixture(scope="module")
def ts_data():
    df = pd.DataFrame(
        {
            "time_int": range(10),
            "id": ["A"] * 10,
            "id2": ["B"] * 10,
            "val": range(10),
            "val2": [x + 100 for x in range(10)],
        }
    )
    df["time_date"] = df["time_int"] * timedelta(days=1) + datetime(2020, 1, 1)
    return df


@pytest.fixture(scope="module")
def ts_data_with_categorical():
    return pd.DataFrame(
        {
            "id": nreps(["A", "B", "C"], 50),
            "timestamp": [datetime(2021, 1, 1) + timedelta(days=i) for i in range(50)] * 3,
            "value1": range(150),
            "value2": np.arange(150) / 3 + 10,
            "value3": np.arange(150) / 50 - 6,
            "color": nreps(["Blue", "Green", "Yellow"], 50),
            "material": nreps(["SUS316", "SUS314", "AL6061"], 50),
        }
    )


def test_ts_padding(ts_data):
    df = ts_data
    # test no padding needed
    df_padded = ts_padding(df, context_length=0)
    pd.testing.assert_frame_equal(df_padded, df)

    # test padding length
    context_length = 12
    df_padded = ts_padding(df, context_length=context_length)
    assert df_padded.shape[0] == context_length

    # test ids handled
    df_padded = ts_padding(
        df,
        id_columns=["id", "id2"],
        timestamp_column="time_int",
        context_length=context_length,
    )

    assert all(df_padded.iloc[0][["id", "id2"]] == ["A", "B"])

    # test date handled
    # integer
    assert df_padded.iloc[0]["time_int"] == df.iloc[0]["time_int"] - (context_length - df.shape[0])

    # date
    df_padded = ts_padding(
        df,
        id_columns=["id", "id2"],
        timestamp_column="time_date",
        context_length=context_length,
    )

    assert df_padded.iloc[0]["time_date"] == df.iloc[0]["time_date"] - (context_length - df.shape[0]) * timedelta(
        days=1
    )


def test_pretrain_df_dataset(ts_data):
    ds = PretrainDFDataset(
        ts_data,
        timestamp_column="time_date",
        target_columns=["val", "val2"],
        id_columns=["id", "id2"],
        context_length=12,
    )

    assert len(ds) == 1

    assert ds[0]["timestamp"] == ts_data.iloc[-1]["time_date"]


def test_forecasting_df_dataset(ts_data_with_categorical):
    prediction_length = 2
    static_categorical_columns = ["color", "material"]
    target_columns = ["value1"]
    observable_columns = ["value3"]
    conditional_columns = ["value2"]

    tsp = TimeSeriesPreprocessor(
        timestamp_column="timestamp",
        id_columns=["id"],
        target_columns=[
            "value1",
        ],
        observable_columns=["value3"],
        conditional_columns=["value2"],
        static_categorical_columns=["color", "material"],
    )

    df = tsp.train(ts_data_with_categorical).preprocess(ts_data_with_categorical)

    ds = ForecastDFDataset(
        df,
        timestamp_column="timestamp",
        id_columns=["id"],
        target_columns=target_columns,
        observable_columns=observable_columns,
        conditional_columns=conditional_columns,
        static_categorical_columns=static_categorical_columns,
        context_length=10,
        prediction_length=prediction_length,
        frequency_token=2,
    )

    # check that we produce outputs for static categorical
    assert "static_categorical_values" in ds[0]
    assert ds[0]["static_categorical_values"].shape == (len(static_categorical_columns),)

    # check that frequency token is present
    assert "freq_token" in ds[0]

    # check shape of future
    assert ds[0]["future_values"].shape == (
        prediction_length,
        len(target_columns + observable_columns + conditional_columns),
    )

    # check future values zeroed out for conditional variables
    assert np.all(ds[0]["future_values"][:, 2].numpy() == 0)


def test_forecasting_df_dataset_non_autoregressive(ts_data_with_categorical):
    prediction_length = 2
    target_columns = ["value1"]
    observable_columns = ["value3"]
    conditional_columns = ["value2"]

    ds = ForecastDFDataset(
        ts_data_with_categorical,
        timestamp_column="timestamp",
        id_columns=["id"],
        target_columns=target_columns,
        observable_columns=observable_columns,
        conditional_columns=conditional_columns,
        context_length=10,
        prediction_length=prediction_length,
        frequency_token=2,
        autoregressive_modeling=False,
    )

    # check that past values of targets are zeroed out
    assert np.all(ds[0]["past_values"][:, 0].numpy() == 0)
