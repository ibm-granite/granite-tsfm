# Copyright contributors to the TSFM project
#

"""Tests the time series classification preprocessor and functions"""

import numpy as np
import pytest

from tsfm_public.toolkit.time_series_classification_preprocessor import (
    TimeSeriesClassificationPreprocessor,
)


def test_label_encodes(ts_data_nested):
    df = ts_data_nested.copy()

    tsp = TimeSeriesClassificationPreprocessor(
        timestamp_column="time_date",
        input_columns=["val", "val2"],
        label_column="label",
        id_columns=[
            "id",
        ],
    )

    tsp.train(df)
    assert tsp.label_encoder is not None
    assert len(tsp.label_encoder.classes_) == len(df.label.unique())

    df_prep = tsp.preprocess(df)
    assert df_prep.label.dtype == "int64"


def test_scaling(ts_data_nested):
    df = ts_data_nested.copy()

    tsp = TimeSeriesClassificationPreprocessor(
        timestamp_column="time_date",
        input_columns=["val", "val2"],
        label_column="label",
        id_columns=[
            "id",
        ],
        scaling=True,
    )

    tsp.train(df)

    df_prep = tsp.preprocess(df)

    # to do add condition
    np.testing.assert_almost_equal(df_prep.val.apply(np.mean).mean(), 0)


def test_static_categorical(ts_data_nested):
    df = ts_data_nested.copy()
    df["cat"] = df.label.copy()

    tsp = TimeSeriesClassificationPreprocessor(
        timestamp_column="time_date",
        input_columns=["val", "val2"],
        label_column="label",
        static_categorical_columns=["cat"],
        id_columns=[
            "id",
        ],
        scaling=True,
    )

    with pytest.raises(Exception):
        tsp.categorical_vocab_size_list

    tsp.train(df)

    assert tsp.categorical_vocab_size_list == [4]


def test_helpers():
    tsp = TimeSeriesClassificationPreprocessor(
        timestamp_column="time_date",
        input_columns=["val", "val2"],
        label_column="label",
        id_columns=[
            "id",
        ],
        scaling_id_columns=["id"],
    )

    assert tsp.num_input_channels == 2
    assert tsp.exogenous_channel_indices == []
    assert tsp.categorical_vocab_size_list is None
