"""Tests for util functions"""

import pytest

from tsfm_public.toolkit.util import get_split_params


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
