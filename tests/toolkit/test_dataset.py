# Copyright contributors to the TSFM project
#

"""Tests basic dataset functions"""

# Standard
from datetime import datetime, timedelta

# Third Party
import pandas as pd
import pytest

# Local
from tsfm_public.toolkit.dataset import PretrainDFDataset, ts_padding


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
        input_columns=["val", "val2"],
        id_columns=["id", "id2"],
        context_length=12,
    )

    assert len(ds) == 1

    assert ds[0]["timestamp"] == ts_data.iloc[-1]["time_date"]
