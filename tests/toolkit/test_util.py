"""Tests for util functions"""

import pandas as pd
import pytest

from tsfm_public.toolkit.util import get_split_params, train_test_split


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
