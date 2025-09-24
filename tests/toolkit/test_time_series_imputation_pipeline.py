# Copyright contributors to the TSFM project
#

"""Tests the time series imputation preprocessor and functions"""

import numpy as np
import pytest

from tsfm_public import TSPulseConfig, TSPulseForReconstruction
from tsfm_public.toolkit.time_series_imputation_pipeline import TimeSeriesImputationPipeline
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor


@pytest.fixture(scope="module")
def tspulse_model(etth_data):
    _, _, params = etth_data
    conf = TSPulseConfig(
        context_length=512,
        d_model=24,
        decoder_d_model=24,
        num_patches=128,
        patch_stride=8,
        fft_time_add_forecasting_pt_loss=False,
        num_input_channels=len(params["target_columns"]),
        mask_type="user",
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
    )
    model = TSPulseForReconstruction(conf)
    return model


@pytest.fixture(scope="module")
def etth_missing_data(etth_data):
    train_data, test_data, params = etth_data
    test_data = test_data.copy()

    # quick and dirty random missing
    n = 20
    rng = np.random.default_rng(seed=42)
    inds = rng.integers(test_data.index[0], test_data.index[-1], n)
    sizes = rng.integers(1, 9, n)
    cols = rng.integers(0, len(params["target_columns"]), n)

    for i, s, c in zip(inds, sizes, cols):
        test_data.loc[i : i + s, params["target_columns"][c]] = np.nan

    return train_data, test_data, params


def test_time_series_imputation_pipeline_defaults(tspulse_model, etth_missing_data):
    train_data, test_data, params = etth_missing_data
    test_data = test_data.copy()

    tsp = TimeSeriesPreprocessor(target_columns=params["target_columns"], scaling=True)
    tsp.train(train_data)

    pipe = TimeSeriesImputationPipeline(tspulse_model, feature_extractor=tsp, device="cpu")

    test_imputed = pipe(test_data)

    assert test_imputed.shape[0] == test_data.shape[0]
    assert all(t in test_imputed.columns for t in params["target_columns"])
    assert all(f"{t}_imputed" in test_imputed.columns for t in params["target_columns"])


def test_imputation_pipeline_outputs_for_nan(tspulse_model, etth_missing_data):
    train_data, test_data, params = etth_missing_data
    test_data = test_data.copy()

    tsp = TimeSeriesPreprocessor(target_columns=params["target_columns"], scaling=True)
    tsp.train(train_data)

    pipe = TimeSeriesImputationPipeline(tspulse_model, feature_extractor=tsp, device="cpu")

    test_imputed = pipe(test_data)

    imputed_columns = []
    for c in params["target_columns"]:
        imputed_columns.append(f"{c}_imputed")

    assert not test_imputed[imputed_columns].isnull().values.any()


def test_imputation_pipeline_outputs_for_original_values(tspulse_model, etth_missing_data):
    train_data, test_data, params = etth_missing_data
    test_data = test_data.copy()

    tsp = TimeSeriesPreprocessor(target_columns=params["target_columns"], scaling=True)
    tsp.train(train_data)

    pipe = TimeSeriesImputationPipeline(tspulse_model, feature_extractor=tsp, device="cpu")

    test_imputed = pipe(test_data)

    for col in params["target_columns"]:
        imp_col = f"{col}_imputed"

        non_missing_loc = test_imputed[col].notna()
        assert (test_imputed.loc[non_missing_loc, col] == test_imputed.loc[non_missing_loc, imp_col]).all()


def test_idempotency_on_fully_observed_data(tspulse_model, etth_data):
    train_data, test_data, params = etth_data
    test_data = test_data.copy()

    assert not test_data.isna().any().any()  # no missing values

    tsp = TimeSeriesPreprocessor(target_columns=params["target_columns"], scaling=True)
    tsp.train(train_data)

    pipe = TimeSeriesImputationPipeline(tspulse_model, feature_extractor=tsp, device="cpu")

    test_imputed = pipe(test_data)

    for col in params["target_columns"]:
        imputed_col = f"{col}_imputed"
        assert imputed_col in test_imputed.columns, f"Missing imputed column: {imputed_col}"

        original_vals = test_imputed[col].values
        imputed_vals = test_imputed[imputed_col].values

        assert np.allclose(
            original_vals, imputed_vals
        ), f"Imputed column '{imputed_col}' differs from original column '{col}'"
