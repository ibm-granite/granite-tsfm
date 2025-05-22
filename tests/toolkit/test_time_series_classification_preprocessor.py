# Copyright contributors to the TSFM project
#

"""Tests the time series classification preprocessor and functions"""

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
