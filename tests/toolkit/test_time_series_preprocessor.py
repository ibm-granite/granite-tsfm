# Copyright contributors to the TSFM project
#

"""Tests the time series preprocessor and functions"""

import numpy as np
import pandas as pd
import pytest

from tsfm_public.toolkit.time_series_preprocessor import (
    OrdinalEncoder,
    StandardScaler,
    TimeSeriesPreprocessor,
)


@pytest.fixture(scope="module")
def sample_data():
    df = pd.DataFrame(
        {
            "val": range(10),
            "val2": [x + 100 for x in range(10)],
            "cat": ["A", "B"] * 5,
            "cat2": ["CC", "DD"] * 5,
        }
    )
    return df


def test_standard_scaler(sample_data):
    scaler = StandardScaler()

    columns = ["val", "val2"]

    # check shape preserved
    result = scaler.fit_transform(sample_data[columns])
    assert result.shape == sample_data[columns].shape
    expected = (
        sample_data[columns].values - np.mean(sample_data[columns].values, axis=0)
    ) / np.std(sample_data[columns].values, axis=0)
    np.testing.assert_allclose(result, expected)

    # check serialization
    state = scaler.to_dict()

    assert StandardScaler.from_dict(state).to_dict() == state


def test_ordinal_encoder(sample_data):
    encoder = OrdinalEncoder()

    columns = ["cat", "cat2"]

    # check shape preserved
    result = encoder.fit_transform(sample_data[columns])
    assert result.shape == sample_data[columns].shape

    expected = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [0.0, 0.0],
            [1.0, 1.0],
            [0.0, 0.0],
            [1.0, 1.0],
            [0.0, 0.0],
            [1.0, 1.0],
            [0.0, 0.0],
            [1.0, 1.0],
        ]
    )
    np.testing.assert_allclose(result, expected)

    # check serialization
    state = encoder.to_dict()

    assert OrdinalEncoder.from_dict(state).to_dict() == state


def test_time_series_preprocessor_encodes(sample_data):

    tsp = TimeSeriesPreprocessor(
        input_columns=["val", "val2"], categorical_columns=["cat", "cat2"]
    )
    tsp.train(sample_data)

    print(tsp.preprocess(sample_data))
