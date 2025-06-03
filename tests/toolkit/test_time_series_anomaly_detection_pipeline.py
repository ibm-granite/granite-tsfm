# Copyright contributors to the TSFM project
#

"""Tests the time series preprocessor and functions"""

import numpy as np
import pandas as pd
import pytest

from tsfm_public.models.tinytimemixer import TinyTimeMixerConfig, TinyTimeMixerForPrediction
from tsfm_public.models.tspulse import TSPulseConfig, TSPulseForReconstruction
from tsfm_public.toolkit.time_series_anomaly_detection_pipeline import (
    AnomalyPredictionModes,
    TimeSeriesAnomalyDetectionPipeline,
)


@pytest.fixture(scope="module")
def example_dataset():
    n_variables: int = 2
    target_variables = [f"X{i + 1}" for i in range(n_variables)]
    data = np.array(
        [np.convolve(np.random.normal(0, 1, 1000), np.ones(15) / 15, "same") for _ in range(n_variables)]
    ).T
    timestamp = pd.date_range("2021-01-01", periods=len(data), freq=pd.Timedelta(5, "minute"))
    df = pd.DataFrame(data, columns=target_variables)
    df["timestamp"] = timestamp
    return target_variables, df


def test_tsad_tspulse_pipeline_defaults(example_dataset):
    target_variables, dataset = example_dataset
    params = {}
    params.update(
        context_length=512,
        patch_length=8,
        mask_block_length=2,
        num_input_channels=len(target_variables),
        patch_stride=8,
        mode="common_channel",
        gated_attn=True,
        use_positional_encoding=False,
        self_attn=False,
        head_aggregation="max_pool",
        fuse_fft=True,
        use_learnable_mask_token=True,
        prediction_length=4,
        fft_time_add_forecasting_pt_loss=True,
        channel_mix_init="identity",
    )

    model = TSPulseForReconstruction(TSPulseConfig(**params))

    tspipe = TimeSeriesAnomalyDetectionPipeline(
        model,
        prediction_mode=AnomalyPredictionModes.PREDICTIVE_WITH_IMPUTATION.value,
        timestamp_column="timestamp",
        target_columns=target_variables,
        aggr_win_size=32,
    )

    assert tspipe._preprocess_params["prediction_length"] == 1
    assert tspipe._preprocess_params["context_length"] == model.config.context_length

    result = tspipe(dataset)
    assert result.shape[0] == dataset.shape[0]
    assert "anomaly_score" in result

    result = tspipe(dataset, expand_score=True)
    assert result.shape[0] == dataset.shape[0]
    for tgt in target_variables:
        assert f"{tgt}_anomaly_score" in result


def test_tsad_tinytimemixture_pipeline_defaults(example_dataset):
    target_variables, dataset = example_dataset
    model = TinyTimeMixerForPrediction(
        TinyTimeMixerConfig(context_length=120, prediction_length=60, num_input_channels=len(target_variables))
    )

    tspipe = TimeSeriesAnomalyDetectionPipeline(
        model,
        prediction_mode=AnomalyPredictionModes.PREDICTIVE.value,
        timestamp_column="timestamp",
        target_columns=["X1", "X2"],
    )
    assert tspipe._preprocess_params["prediction_length"] == model.config.prediction_length
    assert tspipe._preprocess_params["context_length"] == model.config.context_length
    result = tspipe(dataset)
    assert result.shape[0] == dataset.shape[0]
    assert "anomaly_score" in result

    result = tspipe(dataset, expand_score=True)
    assert result.shape[0] == dataset.shape[0]
    for tgt in target_variables:
        assert f"{tgt}_anomaly_score" in result
