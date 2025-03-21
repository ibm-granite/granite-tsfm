# Copyright contributors to the TSFM project
#

"""Tests the time series preprocessor and functions"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from tsfm_public.toolkit.time_series_preprocessor import (
    DEFAULT_FREQUENCY_MAPPING,
    OrdinalEncoder,
    StandardScaler,
    TimeSeriesPreprocessor,
    create_timestamps,
    extend_time_series,
    get_datasets,
)
from tsfm_public.toolkit.util import FractionLocation

from ..util import nreps


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
            [0.0, 2.0],
            [1.0, 3.0],
            [0.0, 4.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 2.0],
            [0.0, 3.0],
            [1.0, 4.0],
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

    # two static categoricals were defined
    assert tsp.categorical_vocab_size_list == [2, 5]

    categorical_columns = ["cat", "cat2"]
    tsp = TimeSeriesPreprocessor(
        target_columns=["val", "val2"],
        categorical_columns=categorical_columns,
        control_columns=categorical_columns,
    )
    tsp.train(sample_data)

    sample_prep = tsp.preprocess(sample_data)

    for c in categorical_columns:
        assert sample_prep[c].dtype == float

    # no static categorical columns defined
    assert tsp.categorical_vocab_size_list is None


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
    assert np.allclose(out.groupby(tsp.id_columns)[tsp.target_columns].apply(lambda x: np.std(x, axis=0)), 1.0)

    # check inverse scale result
    out_inv = tsp.inverse_scale_targets(out)
    assert np.all(
        out_inv.groupby(tsp.id_columns)[tsp.target_columns].apply(lambda x: np.mean(x))
        == df.groupby(tsp.id_columns)[tsp.target_columns].apply(lambda x: np.mean(x))
    )
    assert np.all(
        out_inv.groupby(tsp.id_columns)[tsp.target_columns].apply(lambda x: np.std(x, axis=0))
        == df.groupby(tsp.id_columns)[tsp.target_columns].apply(lambda x: np.std(x, axis=0))
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


def test_extend_time_series(ts_data):
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

    # test different lengths

    ts_data_2 = pd.DataFrame(
        {
            "id": list(nreps(["A", "B"], 50)) + ["C"] * 20,
            "timestamp": [datetime(2021, 1, 1) + timedelta(days=i) for i in range(50)] * 2
            + [datetime(2021, 1, 1) + timedelta(days=i) for i in range(20)],
            "value1": range(120),
        }
    )

    a = extend_time_series(ts_data_2, timestamp_column="timestamp", grouping_columns=["id"], total_periods=60)

    assert len(a) == 180


def test_create_timestamps():
    base_last_timestamp = datetime(2020, 1, 1)
    base_timedelta = timedelta(days=1)
    base_timedelta_strs = ["1d", "1 days 00:00:00", "24h"]
    num_periods = 4
    base_expected = [base_last_timestamp + base_timedelta * i for i in range(1, num_periods + 1)]

    date_types = [datetime, pd.Timestamp, np.datetime64]
    timedelta_types = [timedelta, pd.Timedelta, np.timedelta64]

    test_cases = []
    for date_type in date_types:
        for timedelta_type in timedelta_types:
            test_cases.append(
                {
                    "last_timestamp": base_last_timestamp
                    if isinstance(base_last_timestamp, date_type)
                    else date_type(base_last_timestamp),
                    "freq": base_timedelta
                    if isinstance(base_timedelta, timedelta_type)
                    else timedelta_type(base_timedelta),
                    "periods": 4,
                    "expected": [d if isinstance(d, date_type) else date_type(d) for d in base_expected],
                }
            )

    for date_type in date_types:
        for freq in base_timedelta_strs:
            test_cases.append(
                {
                    "last_timestamp": base_last_timestamp
                    if isinstance(base_last_timestamp, date_type)
                    else date_type(base_last_timestamp),
                    "freq": freq,
                    "periods": 4,
                    "expected": [d if isinstance(d, date_type) else date_type(d) for d in base_expected],
                }
            )

    test_cases.extend(
        [
            {"last_timestamp": 100, "freq": 3, "periods": 2, "expected": [103, 106]},
            {"last_timestamp": 100, "freq": 3.5, "periods": 2, "expected": [103.5, 107.0]},
            {"last_timestamp": 100, "freq": "3.5", "periods": 2, "expected": [103.5, 107.0]},
            {"last_timestamp": np.float32(100), "freq": "3.5", "periods": 2, "expected": [103.5, 107.0]},
        ]
    )

    for test_record in test_cases:
        expected = test_record.pop("expected")
        out = create_timestamps(**test_record)
        assert out == expected


def test_create_timestamps_with_sequence():
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

    train, valid, test = get_datasets(
        tsp,
        ts_data,
        split_config={"train": [0, 1 / 3], "valid": [1 / 3, 2 / 3], "test": [2 / 3, 1]},
    )

    # 3 time series of length 50
    assert len(train) == 3 * (int((1 / 3) * 50) - (tsp.context_length + tsp.prediction_length) + 1)

    assert len(valid) == len(test)

    full_lengths = [len(train), len(valid), len(test)]

    stride = 3
    num_ids = len(ts_data["id"].unique())
    # test stride
    train, valid, test = get_datasets(
        tsp, ts_data, split_config={"train": [0, 1 / 3], "valid": [1 / 3, 2 / 3], "test": [2 / 3, 1]}, stride=stride
    )

    strided_lengths = [len(train), len(valid), len(test)]

    # x is full length under stride 1
    # x // 3 is full length for each ID, need to subtract one and then compute strided length per ID
    assert [(((x // num_ids) - 1) // stride + 1) * num_ids for x in full_lengths] == strided_lengths

    # no id columns, so treat as one big time series
    tsp = TimeSeriesPreprocessor(
        id_columns=[],
        target_columns=["value1", "value2"],
        prediction_length=5,
        context_length=10,
    )

    train, valid, test = get_datasets(
        tsp,
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

    train, valid, test = get_datasets(
        tsp,
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

    train, valid, test = get_datasets(
        tsp,
        ts_data,
        split_config={
            "train": 0.7,
            "test": 0.2,
        },
    )

    assert len(train) == int(150 * 0.7) - (tsp.context_length + tsp.prediction_length) + 1

    assert len(test) == int(150 * 0.2) - tsp.prediction_length + 1

    assert len(valid) == 150 - int(150 * 0.2) - int(150 * 0.7) - tsp.prediction_length + 1

    full_train_size = len(train)

    train, valid, test = get_datasets(
        tsp,
        ts_data,
        split_config={
            "train": 0.7,
            "test": 0.2,
        },
        fewshot_fraction=0.2,
        fewshot_location=FractionLocation.UNIFORM.value,
        seed=42,
    )

    assert len(train) == int(full_train_size * 0.2)


def test_get_datasets_padding(ts_data):
    tsp = TimeSeriesPreprocessor(
        id_columns=["id"],
        target_columns=["value1", "value2"],
        prediction_length=5,
        context_length=13,
    )

    train, valid, test = get_datasets(
        tsp,
        ts_data,
        split_config={"train": [0, 1 / 3], "valid": [1 / 3, 2 / 3], "test": [2 / 3, 1]},
    )

    assert len(train) == 3

    with pytest.raises(RuntimeError):
        train, valid, test = get_datasets(
            tsp,
            ts_data,
            split_config={"train": [0, 1 / 3], "valid": [1 / 3, 2 / 3], "test": [2 / 3, 1]},
            enable_padding=False,
        )


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

    train, _, _ = get_datasets(tsp, ts_data, split_config={"train": 0.7, "test": 0.2})

    assert train.datasets[0].target_columns == ["value1", "value2"]


def test_get_datasets_univariate(ts_data):
    tsp = TimeSeriesPreprocessor(
        timestamp_column="timestamp",
        id_columns=["id", "id2"],
        target_columns=["value1", "value2"],
        prediction_length=2,
        context_length=5,
    )

    # for baseline
    train_base, _, _ = get_datasets(tsp, ts_data, split_config={"train": 0.7, "test": 0.2})

    train, _, _ = get_datasets(tsp, ts_data, split_config={"train": 0.7, "test": 0.2}, as_univariate=True)

    assert train[0]["id"][-1] == "value1"
    assert len(train) == 2 * len(train_base)


def test_get_datasets_with_frequency_token(ts_data):
    ts_data = ts_data.drop(columns=["id", "id2"])
    tsp = TimeSeriesPreprocessor(timestamp_column="timestamp", prediction_length=2, context_length=5, freq="d")

    train, _, _ = get_datasets(tsp, ts_data, split_config={"train": 0.7, "test": 0.2}, use_frequency_token=True)

    assert train[0]["freq_token"] == DEFAULT_FREQUENCY_MAPPING["d"]


def test_masking_specification(ts_data):
    ts_data = ts_data.drop(columns=["id", "id2"])
    tsp = TimeSeriesPreprocessor(timestamp_column="timestamp", prediction_length=2, context_length=5, freq="d")

    train, _, _ = get_datasets(
        tsp,
        ts_data,
        split_config={"train": 0.7, "test": 0.2},
        use_frequency_token=True,
        fill_value=-1000,
        masking_specification=[("value1", -2)],
    )

    assert np.all(train[0]["past_values"].numpy()[-2:, 0] == -1000)


def test_get_frequency_token():
    tsp = TimeSeriesPreprocessor(timestamp_column="date")

    assert tsp.get_frequency_token("1h") == DEFAULT_FREQUENCY_MAPPING["h"]
    assert tsp.get_frequency_token("h") == DEFAULT_FREQUENCY_MAPPING["h"]
    assert tsp.get_frequency_token("0 days 01:00:00") == DEFAULT_FREQUENCY_MAPPING["h"]
    assert tsp.get_frequency_token("H") == DEFAULT_FREQUENCY_MAPPING["h"]
    assert tsp.get_frequency_token("1H") == DEFAULT_FREQUENCY_MAPPING["h"]


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

    ds_train, _, _ = get_datasets(tsp, df, split_config={"train": 0.7, "test": 0.2})

    assert len(tsp.target_scaler_dict) == 2
    assert len(ds_train.datasets) == 4

    tsp = TimeSeriesPreprocessor(
        timestamp_column="timestamp",
        prediction_length=2,
        context_length=5,
        id_columns=["asset_id", "run_id"],
        scaling_id_columns=[],
        target_columns=["value1"],
        scaling=True,
    )

    ds_train, _, _ = get_datasets(tsp, df, split_config={"train": 0.7, "test": 0.2})

    assert len(tsp.target_scaler_dict) == 1
    assert len(ds_train.datasets) == 4

    tsp = TimeSeriesPreprocessor(
        timestamp_column="timestamp",
        prediction_length=2,
        context_length=5,
        id_columns=["asset_id", "run_id"],
        scaling_id_columns=None,
        target_columns=["value1"],
        scaling=True,
    )

    ds_train, _, _ = get_datasets(tsp, df, split_config={"train": 0.7, "test": 0.2})

    assert len(tsp.target_scaler_dict) == 4
    assert len(ds_train.datasets) == 4


def test_get_datasets_with_categoricical(ts_data):
    ts_data = ts_data.copy()

    ts_data["varying_cat"] = ts_data.apply(lambda x: x["id"] + str(int(x["value1"] % 5)), axis=1)

    tsp = TimeSeriesPreprocessor(
        timestamp_column="timestamp",
        id_columns=["id", "id2"],
        target_columns=["value1", "value2"],
        prediction_length=2,
        context_length=5,
        categorical_columns=["varying_cat"],
        conditional_columns=["varying_cat"],
        scaling=False,
    )

    # for baseline
    train, _, _ = get_datasets(tsp, ts_data, split_config={"train": 0.7, "test": 0.2})
    expected = np.array([2.0, 3.0, 4.0, 0.0, 1.0])
    np.testing.assert_allclose(train[2]["past_values"][:, 2].numpy(), expected)

    tsp = TimeSeriesPreprocessor(
        timestamp_column="timestamp",
        id_columns=["id", "id2"],
        target_columns=["value1", "value2"],
        prediction_length=2,
        context_length=5,
        categorical_columns=["varying_cat"],
        conditional_columns=["varying_cat"],
        scaling=True,
    )

    # for baseline
    train, _, _ = get_datasets(tsp, ts_data, split_config={"train": 0.7, "test": 0.2})
    expected = np.array([0.0000, 0.7071, 1.4142, -1.4142, -0.7071])
    np.testing.assert_allclose(train[2]["past_values"][:, 2].numpy(), expected, rtol=1e-4)
