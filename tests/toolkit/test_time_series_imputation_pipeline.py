# Copyright contributors to the TSFM project
#

"""Tests the time series preprocessor and functions"""

import numpy as np

from tsfm_public import TSPulseForReconstruction
from tsfm_public.toolkit.time_series_imputation_pipeline import TimeSeriesImputationPipeline
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor


def test_time_series_imputation_pipeline_defaults(etth_data):
    train_data, test_data, params = etth_data
    test_data = test_data.copy()

    tsp = TimeSeriesPreprocessor(target_columns=params["target_columns"], scaling=True)
    tsp.train(train_data)

    model_path = "add your path here"

    # quick and dirty random missing
    n = 20
    rng = np.random.default_rng(seed=42)
    inds = rng.integers(test_data.index[0], test_data.index[-1], n)
    sizes = rng.integers(1, 9, n)
    cols = rng.integers(0, len(params["target_columns"]), n)

    for i, s, c in zip(inds, sizes, cols):
        test_data.loc[i : i + s, params["target_columns"][c]] = np.nan

    model = TSPulseForReconstruction.from_pretrained(
        model_path,
        fft_time_add_forecasting_pt_loss=False,
        num_input_channels=len(params["target_columns"]),
        mask_type="user",
    )

    pipe = TimeSeriesImputationPipeline(model, feature_extractor=tsp, device="cpu")

    test_imputed = pipe(test_data)

    assert test_imputed.shape[0] == test_data.shape[0]
    assert all(t in test_imputed.columns for t in params["target_columns"])
    assert all(f"{t}_imputed" in test_imputed.columns for t in params["target_columns"])
