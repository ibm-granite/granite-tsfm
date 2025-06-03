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
        patch_length=2,
        mask_block_length=2,
        num_input_channels=len(target_variables),
        patch_stride=2,
        expansion_factor=2,
        num_layers=2,
        dropout=0.1,
        mode="common_channel",
        gated_attn=True,
        norm_mlp="LayerNorm",
        head_dropout=0.1,
        scaling="std",
        use_positional_encoding=False,
        self_attn=False,
        self_attn_heads=1,
        num_parallel_samples=4,
        decoder_num_layers=2,
        decoder_mode="mix_channel",
        d_model_layerwise_scale=[1, 0.75],
        num_patches_layerwise_scale=[1, 0.75],  # , 0.5, 0.5],
        decoder_num_patches_layerwise_scale=[0.75, 1],  #  0.75, 1],
        decoder_d_model_layerwise_scale=[0.75, 1],  # 0.75, 1],
        num_channels_layerwise_scale=[1, 1],  # 1, 1],  # [1, 0.75, 0.5, 0.5],
        decoder_num_channels_layerwise_scale=[
            1,
            1,
        ],  # 1, 1],  # [0.5, 0.5, 0.75, 1],
        d_model=8,
        decoder_d_model=8,
        num_targets=5,
        head_aggregation="max_pool",
        fuse_fft=True,
        patch_register_tokens=2,
        channel_register_tokens=None,
        fft_mask_ratio=0.2,
        fft_mask_strategy="random",
        use_learnable_mask_token=True,
        prediction_length=4,
        fft_time_add_forecasting_pt_loss=True,
        channel_mix_init="identity",
        reconstruction_loss_weight=1,
        masked_reconstruction_loss_weight=1,
        register_mixer_layers=1,
        head_gated_attention_activation="softmax",
        gated_attention_activation="softmax",
        head_attention=False,
        head_reduce_channels=None,
        fft_applied_on="scaled_ts",
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
