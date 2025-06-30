# Copyright contributors to the TSFM project
#

"""Tests the time series classification pipeline"""

import pytest

from tsfm_public.models.tspulse import TSPulseConfig, TSPulseForClassification
from tsfm_public.toolkit.time_series_classification_pipeline import TimeSeriesClassificationPipeline
from tsfm_public.toolkit.time_series_classification_preprocessor import TimeSeriesClassificationPreprocessor


@pytest.fixture(scope="module")
def tspulse_model():
    conf = TSPulseConfig(
        context_length=512,
        d_model=24,
        decoder_d_model=24,
        num_patches=128,
        patch_stride=8,
        fft_time_add_forecasting_pt_loss=False,
        num_input_channels=2,  # ts_data_nested has 2 input_columns
        patch_length=8,
        mask_block_length=2,
        mode="common_channel",
        gated_attn=True,
        use_positional_encoding=False,
        self_attn=False,
        head_aggregation="max_pool",
        fuse_fft=True,
        use_learnable_mask_token=True,
        prediction_length=4,
        channel_mix_init="identity",
        loss="cross_entropy",
        num_targets=4,  # set to 4 classes as in the ts_data_nested
    )
    model = TSPulseForClassification(conf)
    return model


def test_time_series_classification_pipeline_defaults(tspulse_model, ts_data_nested):
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

    pipe = TimeSeriesClassificationPipeline(tspulse_model, feature_extractor=tsp, device="cpu")

    out_df = pipe(df)

    assert df.shape[0] == out_df.shape[0]
    assert all(t in out_df.columns for t in ["val", "val2"])
    assert "label_prediction" in out_df.columns


def test_time_series_classification_pipeline_predictions(tspulse_model, ts_data_nested):
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

    pipe = TimeSeriesClassificationPipeline(tspulse_model, feature_extractor=tsp, device="cpu")

    out_df = pipe(df)

    assert out_df["label_prediction"].isin(out_df["label"]).all()


def test_time_series_classification_pipeline_inputs_integrity(tspulse_model, ts_data_nested):
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

    pipe = TimeSeriesClassificationPipeline(tspulse_model, feature_extractor=tsp, device="cpu")

    out_df = pipe(df)

    assert ((out_df["val"] == df["val"]) & (out_df["val2"] == df["val2"])).all()
