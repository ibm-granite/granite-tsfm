# Copyright contributors to the TSFM project
#

"""Tests the time series preprocessor and functions"""

import numpy as np
import pandas as pd
import pytest

from tsfm_public.models.tinytimemixer import TinyTimeMixerConfig, TinyTimeMixerForPrediction
from tsfm_public.models.tspulse import TSPulseConfig, TSPulseForReconstruction
from tsfm_public.toolkit.ad_helpers import AnomalyScoreMethods
from tsfm_public.toolkit.conformal import (
    NonconformityScores,
    PostHocProbabilisticMethod,
    PostHocProbabilisticProcessor,
)
from tsfm_public.toolkit.time_series_anomaly_detection_pipeline import (
    TimeSeriesAnomalyDetectionPipeline,
)
from tsfm_public.toolkit.time_series_forecasting_pipeline import TimeSeriesForecastingPipeline


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


@pytest.mark.parametrize(
    "method",
    [
        [AnomalyScoreMethods.TIME_RECONSTRUCTION.value],
        [AnomalyScoreMethods.FREQUENCY_RECONSTRUCTION.value],
        [AnomalyScoreMethods.PREDICTIVE.value],
        [
            AnomalyScoreMethods.TIME_RECONSTRUCTION.value,
            AnomalyScoreMethods.FREQUENCY_RECONSTRUCTION.value,
            AnomalyScoreMethods.PREDICTIVE.value,
        ],
    ],
)
def test_tsad_tspulse_pipeline_defaults(example_dataset, method):
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
        prediction_mode=method,
        timestamp_column="timestamp",
        target_columns=target_variables,
        aggregation_length=32,
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

    result = tspipe(dataset, report_mode=True)
    assert result.shape[0] == dataset.shape[0]
    assert "selected_mode" in result

    result = tspipe(dataset, expand_score=True, report_mode=True)
    assert result.shape[0] == dataset.shape[0]
    for tgt in target_variables:
        assert f"{tgt}_selected_mode" in result


@pytest.mark.parametrize(
    "method",
    [
        [AnomalyScoreMethods.PREDICTIVE.value],
        [AnomalyScoreMethods.MEAN_DEVIATION.value],
        [AnomalyScoreMethods.MEAN_DEVIATION.value, AnomalyScoreMethods.PREDICTIVE.value],
    ],
)
def test_tsad_tinytimemixer_pipeline_defaults(example_dataset, method):
    target_variables, dataset = example_dataset
    model = TinyTimeMixerForPrediction(
        TinyTimeMixerConfig(context_length=120, prediction_length=60, num_input_channels=len(target_variables))
    )

    tspipe = TimeSeriesAnomalyDetectionPipeline(
        model,
        prediction_mode=method,
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


@pytest.mark.parametrize(
    "method",
    [
        [AnomalyScoreMethods.PROBABILISTIC.value],
        [AnomalyScoreMethods.PROBABILISTIC.value, AnomalyScoreMethods.MEAN_DEVIATION.value],
    ],
)
def test_tsad_tinytimemixer_pipeline_probabilistic(example_dataset, method):
    target_variables, dataset = example_dataset
    model = TinyTimeMixerForPrediction(
        TinyTimeMixerConfig(context_length=120, prediction_length=60, num_input_channels=len(target_variables))
    )

    prob_proc = PostHocProbabilisticProcessor(
        window_size=200,
        quantiles=[0.25, 0.75],
        nonconformity_score=NonconformityScores.ABSOLUTE_ERROR.value,
        method=PostHocProbabilisticMethod.CONFORMAL.value,
        smoothing_length=None,
    )

    fpipe = TimeSeriesForecastingPipeline(
        model, timestamp_column="timestamp", id_columns=[], target_columns=target_variables, device="cpu"
    )
    forecasts = fpipe(dataset)

    prediction_length = 60
    prediction_columns = [f"{c}_prediction" for c in target_variables]
    ground_truth_columns = target_variables
    y = forecasts[prediction_columns].values
    predictions = np.array([np.stack(z) for z in y]).transpose(0, 2, 1)
    predictions = predictions[:-prediction_length, ...]
    x = forecasts[ground_truth_columns].values
    ground_truth = np.array([np.stack(z) for z in x]).transpose(0, 2, 1)
    ground_truth = ground_truth[:-prediction_length, ...]

    prob_proc.train(y_cal_gt=ground_truth, y_cal_pred=predictions)

    tspipe = TimeSeriesAnomalyDetectionPipeline(
        model,
        prediction_mode=method,
        probabilistic_processor=prob_proc,
        timestamp_column="timestamp",
        target_columns=["X1", "X2"],
        device="cpu",
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

    result = tspipe(
        dataset,
        prediction_mode=f"{AnomalyScoreMethods.PROBABILISTIC.value}+{AnomalyScoreMethods.MEAN_DEVIATION.value}",
        expand_score=True,
    )
    assert result.shape[0] == dataset.shape[0]
    for tgt in target_variables:
        assert f"{tgt}_anomaly_score" in result
