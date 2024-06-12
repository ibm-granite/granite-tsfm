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


def test_forecasting_df_dataset_stride(ts_data_with_categorical):
    prediction_length = 2
    context_length = 3
    stride = 13
    target_columns = ["value1", "value2"]

    df = ts_data_with_categorical

    ds = ForecastDFDataset(
        df,
        timestamp_column="timestamp",
        id_columns=["id"],
        target_columns=target_columns,
        context_length=context_length,
        prediction_length=prediction_length,
        stride=stride,
    )

    # length check
    series_len = len(df) / len(df["id"].unique())
    assert len(ds) == ((series_len - prediction_length - context_length + 1) // stride) * len(df["id"].unique())

    # check proper windows are selected based on chosen stride
    ds_past_np = np.array([v["past_values"].numpy() for v in ds])
    ds_past_np_expected = np.array(
        [
            [[0.0, 10.0], [1.0, 10.333333], [2.0, 10.666667]],
            [[13.0, 14.333333], [14.0, 14.666667], [15.0, 15.0]],
            [[26.0, 18.666666], [27.0, 19.0], [28.0, 19.333334]],
            [[50.0, 26.666666], [51.0, 27.0], [52.0, 27.333334]],
            [[63.0, 31.0], [64.0, 31.333334], [65.0, 31.666666]],
            [[76.0, 35.333332], [77.0, 35.666668], [78.0, 36.0]],
            [[100.0, 43.333332], [101.0, 43.666668], [102.0, 44.0]],
            [[113.0, 47.666668], [114.0, 48.0], [115.0, 48.333332]],
            [[126.0, 52.0], [127.0, 52.333332], [128.0, 52.666668]],
        ]
    )

    np.testing.assert_allclose(ds_past_np, ds_past_np_expected)


def test_forecasting_observed_mask(ts_data_with_categorical):
    prediction_length = 2
    context_length = 5
    fill_value = 0.0
    target_columns = ["value2", "value3"]

    df = ts_data_with_categorical.copy()
    df.loc[10, "value3"] = np.nan

    ds = ForecastDFDataset(
        df,
        timestamp_column="timestamp",
        id_columns=["id"],
        target_columns=target_columns,
        context_length=context_length,
        prediction_length=prediction_length,
        fill_value=fill_value,
    )

    # check matching size
    assert ds[0]["past_observed_mask"].shape == ds[0]["past_values"].shape
    assert ds[0]["future_observed_mask"].shape == ds[0]["future_values"].shape

    # Check mask is correct
    np.testing.assert_allclose(ds[4]["future_observed_mask"], np.array([[True, True], [True, False]]))
    np.testing.assert_allclose(ds[6]["past_observed_mask"][-1, :], np.array([True, False]))

    # Check mask value is correct
    ds[4]["future_values"][1, 1] == fill_value

    # Check mask value is correct again
    fill_value = -100.0
    ds = ForecastDFDataset(
        df,
        timestamp_column="timestamp",
        id_columns=["id"],
        target_columns=target_columns,
        context_length=context_length,
        prediction_length=prediction_length,
        fill_value=fill_value,
    )

    ds[4]["future_values"][1, 1] == fill_value


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
