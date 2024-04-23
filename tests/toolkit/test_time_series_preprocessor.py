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


def test_time_series_preprocessor_scales(ts_data):
    df = ts_data

    tsp = TimeSeriesPreprocessor(
        timestamp_column="timestamp",
        prediction_length=2,
        context_length=5,
        id_columns=["id", "id2"],
        target_columns=["value1", "value2"],
        scaling=True,
    )

    tsp.train(df)

    # check scaled result
    out = tsp.preprocess(df)
    assert np.allclose(out.groupby(tsp.id_columns)[tsp.target_columns].apply(lambda x: np.mean(x)), 0.0)
    assert np.allclose(out.groupby(tsp.id_columns)[tsp.target_columns].apply(lambda x: np.std(x)), 1.0)

    # check inverse scale result
    out_inv = tsp.inverse_scale_targets(out)
    assert np.all(
        out_inv.groupby(tsp.id_columns)[tsp.target_columns].apply(lambda x: np.mean(x))
        == df.groupby(tsp.id_columns)[tsp.target_columns].apply(lambda x: np.mean(x))
    )
    assert np.all(
        out_inv.groupby(tsp.id_columns)[tsp.target_columns].apply(lambda x: np.std(x))
        == df.groupby(tsp.id_columns)[tsp.target_columns].apply(lambda x: np.std(x))
    )

    # check inverse scale result, with suffix

    suffix = "_foo"
    targets_suffix = [f"{c}{suffix}" for c in tsp.target_columns]
    out.columns = [f"{c}{suffix}" if c in tsp.target_columns else c for c in out.columns]
    out_inv = tsp.inverse_scale_targets(out, suffix=suffix)
    assert np.all(
        out_inv.groupby(tsp.id_columns)[targets_suffix].apply(lambda x: np.mean(x))
        == df.groupby(tsp.id_columns)[tsp.target_columns].apply(lambda x: np.mean(x))
    )


def test_time_series_preprocessor_inv_scales_lists(ts_data):
    df = ts_data

    tsp = TimeSeriesPreprocessor(
        timestamp_column="timestamp",
        prediction_length=2,
        context_length=5,
        id_columns=["id", "id2"],
        target_columns=["value1", "value2"],
        scaling=True,
    )

    tsp.train(df)

    # check scaled result
    out = tsp.preprocess(df)

    # construct artificial result
    out["value1"] = out["value1"].apply(lambda x: np.array([x] * 3))
    out["value2"] = out["value2"].apply(lambda x: np.array([x] * 3))

    out_inv = tsp.inverse_scale_targets(out)

    assert out_inv["value1"].mean()[0] == df["value1"].mean()
    assert out_inv["value2"].mean()[0] == df["value2"].mean()


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
        (
            pd.Timestamp(2021, 12, 31),
            "QE",
            None,
            4,
            [
                pd.Timestamp(2022, 3, 31),
                pd.Timestamp(2022, 6, 30),
                pd.Timestamp(2022, 9, 30),
                pd.Timestamp(2022, 12, 31),
            ],
        ),
    ]

    for start, freq, sequence, periods, expected in test_cases:
        # test based on provided freq
        ts = create_timestamps(start, freq=freq, periods=periods)
        assert ts == expected

        # test based on provided sequence
        if sequence is not None:
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
    fewshot_train_size = (
        int((100 - tsp.context_length) * 0.2) + tsp.context_length - (tsp.context_length + tsp.prediction_length) + 1
    )
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

    # fraction splits
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
            "train": 0.7,
            "test": 0.2,
        },
    )

    assert len(train) == int(150 * 0.7) - (tsp.context_length + tsp.prediction_length) + 1

    assert len(test) == int(150 * 0.2) - tsp.prediction_length + 1

    assert len(valid) == 150 - int(150 * 0.2) - int(150 * 0.7) - tsp.prediction_length + 1


def test_train_without_targets(ts_data):
    # no targets or other columns specified
    tsp = TimeSeriesPreprocessor(id_columns=["id", "id2"], timestamp_column="timestamp")
    tsp.train(ts_data)

    assert tsp.target_columns == ["value1", "value2"]

    # some other args specified
    for arg in [
        "control_columns",
        "conditional_columns",
        "observable_columns",
        "static_categorical_columns",
    ]:
        tsp = TimeSeriesPreprocessor(
            id_columns=["id", "id2"],
            timestamp_column="timestamp",
            **{arg: ["value2"]},
        )
        tsp.train(ts_data)

        assert tsp.target_columns == ["value1"]

    # test targets honored
    tsp = TimeSeriesPreprocessor(
        id_columns=["id", "id2"],
        timestamp_column="timestamp",
        target_columns=["value2"],
    )
    tsp.train(ts_data)

    assert tsp.target_columns == ["value2"]


def test_get_datasets_without_targets(ts_data):
    ts_data = ts_data.drop(columns=["id", "id2"])
    tsp = TimeSeriesPreprocessor(
        timestamp_column="timestamp",
        prediction_length=2,
        context_length=5,
    )

    train, _, _ = tsp.get_datasets(ts_data, split_config={"train": 0.7, "test": 0.2})

    train.datasets[0].target_columns == ["value1", "value2"]


def test_id_columns_and_scaling_id_columns(ts_data_runs):
    df = ts_data_runs

    tsp = TimeSeriesPreprocessor(
        timestamp_column="timestamp",
        prediction_length=2,
        context_length=5,
        id_columns=["asset_id", "run_id"],
        scaling_id_columns=["asset_id"],
        target_columns=["value1"],
        scaling=True,
    )

    ds_train, ds_valid, ds_test = tsp.get_datasets(df, split_config={"train": 0.7, "test": 0.2})

    assert len(tsp.target_scaler_dict) == 2
    assert len(ds_train.datasets) == 4
