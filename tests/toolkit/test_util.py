"""Tests for util functions"""

import tempfile

import pandas as pd
import pytest

from tsfm_public.toolkit.util import convert_to_univariate, convert_tsfile, get_split_params, train_test_split


split_cases = [
    (0, 1, "select_by_index"),
    (0, 0.1, "select_by_relative_fraction"),
    (0.0, 0.1, "select_by_relative_fraction"),
    (0.0, 200.0, "select_by_index"),
    (0.0, 200, "select_by_index"),
    (0.5, 1, "select_by_relative_fraction"),
    (0.5, 1.0, "select_by_relative_fraction"),
    (10, 100.0, "select_by_index"),
]


@pytest.mark.parametrize("left_arg,right_arg,expected", split_cases)
def test_get_split_params(left_arg, right_arg, expected):
    """Test that get_split_params gives the right split function"""

    split_config, split_function = get_split_params({"train": [left_arg, right_arg], "valid": [0, 1], "test": [0, 1]})

    assert split_function["train"].__name__ == expected


def test_train_test_split():
    n = 100
    df = pd.DataFrame({"date": range(n), "value": range(n)})

    train, valid, test = train_test_split(df, train=0.7, test=0.2)

    assert len(train) == int(n * 0.7)
    assert len(test) == int(n * 0.2)
    valid_len_100 = n - int(n * 0.7) - int(n * 0.2)
    assert len(valid) == valid_len_100

    n = 101
    df = pd.DataFrame({"date": range(n), "value": range(n)})

    train, valid, test = train_test_split(df, train=0.7, test=0.2)

    assert len(train) == int(n * 0.7)
    assert len(test) == int(n * 0.2)
    valid_len_101 = n - int(n * 0.7) - int(n * 0.2)
    assert len(valid) == valid_len_101

    assert valid_len_100 + 1 == valid_len_101


def test_convert_to_univariate(ts_data):
    id_columns = ["id"]
    timestamp_column = "timestamp"
    target_columns = ["value1", "value2"]
    df_uni = convert_to_univariate(
        ts_data, timestamp_column=timestamp_column, id_columns=id_columns, target_columns=target_columns
    )

    assert df_uni.columns.to_list() == [
        timestamp_column,
    ] + id_columns + ["column_id", "value"]

    assert len(df_uni) == len(target_columns) * len(ts_data)

    # test single column

    target_columns = ["value1"]

    with pytest.raises(ValueError):
        df_uni = convert_to_univariate(
            ts_data, timestamp_column=timestamp_column, id_columns=id_columns, target_columns=target_columns
        )


def test_convert_tsfile():
    data = """#Test
#
#The classes are
#1. one
#2. two
#3. three
@problemName test
@timeStamps false
@missing false
@univariate true
@equalLength true
@seriesLength 5
@classLabel true 1 2 3
@data
1,2,3,4,5:1
10,20,30,40,50:2
11,12,13,14,15:3
"""
    with tempfile.NamedTemporaryFile() as t:
        t.write(data.encode("utf-8"))
        t.flush()
        df = convert_tsfile(t.name)

    assert df.shape == (15, 3)
