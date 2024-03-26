# Copyright contributors to the TSFM project
#

"""Tests the time series preprocessor and functions"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from tsfm_public.toolkit.time_series_preprocessor import (
    OrdinalEncoder,
    StandardScaler,
    TimeSeriesPreprocessor,
    create_timestamps,
    extend_time_series,
)
from tsfm_public.toolkit.util import FractionLocation


def test_standard_scaler(sample_data):
    scaler = StandardScaler()

    columns = ["val", "val2"]

    # check shape preserved
    result = scaler.fit_transform(sample_data[columns])
    assert result.shape == sample_data[columns].shape
    expected = (sample_data[columns].values - np.mean(sample_data[columns].values, axis=0)) / np.std(
        sample_data[columns].values, axis=0
    )
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
    static_categorical_columns = ["cat", "cat2"]

    tsp = TimeSeriesPreprocessor(
        target_columns=["val", "val2"],
        static_categorical_columns=static_categorical_columns,
    )
    tsp.train(sample_data)

    sample_prep = tsp.preprocess(sample_data)

    for c in static_categorical_columns:
        assert sample_prep[c].dtype == float


def test_augment_time_series(ts_data):
    periods = 5
    a = extend_time_series(ts_data, timestamp_column="timestamp", grouping_columns=["id"], periods=periods)

    # check that length increases by periods for each id
    assert a.shape[0] == ts_data.shape[0] + 3 * periods
    assert a.shape[1] == ts_data.shape[1]

    periods = 3
    a = extend_time_series(
        ts_data,
        timestamp_column="timestamp",
        grouping_columns=["id", "id2"],
        periods=periods,
    )

    # check that length increases by periods for each id
    assert a.shape[0] == ts_data.shape[0] + 3 * periods
    assert a.shape[1] == ts_data.shape[1]


def test_create_timestamps():
    # start, freq, sequence, periods, expected
    test_cases = [
        (
            datetime(2020, 1, 1),
            "1d",
            [datetime(2019, 1, 2), datetime(2019, 1, 3), datetime(2019, 1, 4)],
            2,
            [datetime(2020, 1, 2), datetime(2020, 1, 3)],
        ),
        (
            datetime(2021, 1, 1),
            timedelta(days=365),
            [datetime(2017, 1, 3), datetime(2018, 1, 3), datetime(2019, 1, 3)],
            2,
            [datetime(2022, 1, 1), datetime(2023, 1, 1)],
        ),
        (
            pd.Timestamp(2020, 1, 1),
            pd.Timedelta(weeks=1),
            [pd.Timestamp(2019, 2, 1), pd.Timestamp(2019, 2, 8)],
            2,
            [datetime(2020, 1, 8), datetime(2020, 1, 15)],
        ),
        (
            100,
            3,
            [3, 6, 9],
            2,
            [103, 106],
        ),
        (
            100,
            3.5,
            [10, 13.5, 17.0],
            2,
            [103.5, 107.0],
        ),
    ]

    for start, freq, sequence, periods, expected in test_cases:
        # test based on provided freq
        ts = create_timestamps(start, freq=freq, periods=periods)
        assert ts == expected

        # test based on provided sequence
        ts = create_timestamps(start, time_sequence=sequence, periods=periods)
        assert ts == expected

    # it is an error to provide neither freq or sequence
    with pytest.raises(ValueError):
        ts = create_timestamps(start, periods=periods)


def test_get_datasets(ts_data):
    tsp = TimeSeriesPreprocessor(
        id_columns=["id"],
        target_columns=["value1", "value2"],
        prediction_length=5,
        context_length=10,
    )

    train, valid, test = tsp.get_datasets(
        ts_data,
        split_config={"train": [0, 1 / 3], "valid": [1 / 3, 2 / 3], "test": [2 / 3, 1]},
    )

    # 3 time series of length 50
    assert len(train) == 3 * (int((1 / 3) * 50) - (tsp.context_length + tsp.prediction_length) + 1)

    assert len(valid) == len(test)

    # no id columns, so treat as one big time series
    tsp = TimeSeriesPreprocessor(
        id_columns=[],
        target_columns=["value1", "value2"],
        prediction_length=5,
        context_length=10,
    )

    train, valid, test = tsp.get_datasets(
        ts_data,
        split_config={
            "train": [0, 100],
            "valid": [100, 125],
            "test": [125, 150],
        },
        fewshot_fraction=0.2,
        fewshot_location=FractionLocation.LAST.value,
    )

    # new train length should be 20% of 100, minus the usual for context length and prediction length
    fewshot_train_size = int(100 * 0.2) - (tsp.context_length + tsp.prediction_length) + 1
    assert len(train) == fewshot_train_size

    assert len(valid) == len(test)

    # no id columns, so treat as one big time series
    tsp = TimeSeriesPreprocessor(
        id_columns=[],
        target_columns=["value1", "value2"],
        prediction_length=5,
        context_length=10,
    )

    train, valid, test = tsp.get_datasets(
        ts_data,
        split_config={
            "train": [0, 100],
            "valid": [100, 125],
            "test": [125, 150],
        },
        fewshot_fraction=0.2,
        fewshot_location=FractionLocation.FIRST.value,
    )

    # new train length should be 20% of 100, minus the usual for context length and prediction length
    assert len(train) == fewshot_train_size

    assert len(valid) == len(test)
