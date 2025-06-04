# Copyright contributors to the TSFM project
#

"""Tests conformal processor capabilities"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from tsfm_public.toolkit.conformal import (
    NonconformityScores,
    PostHocGaussian,
    PostHocProbabilisticMethod,
    PostHocProbabilisticProcessor,
    WeightedConformalForecasterWrapper,
)


@pytest.mark.parametrize(
    "method", [PostHocProbabilisticMethod.CONFORMAL.value, PostHocProbabilisticMethod.GAUSSIAN.value]
)
def test_conformal_save_pretrained(method):
    # initial test to check that we save the ProbabbilisticProcessor as intended
    p = PostHocProbabilisticProcessor(method=method)

    with tempfile.TemporaryDirectory() as d:
        p.save_pretrained(d)
        p_new = PostHocProbabilisticProcessor.from_pretrained(d)
        assert Path(d).joinpath(PostHocProbabilisticProcessor.PROCESSOR_NAME).exists()

        assert p_new.method == method

        # to do: add checks that p and p_new are equivalent


def test_posthoc_probabilistic_processor():
    np.random.seed(42)

    # Parameters
    window_size = 10
    quantiles = [0.1, 0.5, 0.9]

    # Data
    n_samples = 30
    forecast_horizon = 5
    n_features = 3

    y_pred = np.random.randn(n_samples, forecast_horizon, n_features)
    y_gt = y_pred + np.random.normal(0, 0.5, size=y_pred.shape)

    y_cal_pred = y_pred[0:20]
    y_cal_gt = y_gt[0:20]
    y_test_pred = y_pred[-10:]
    # y_test_gt = y_gt[-10:]

    for method in [PostHocProbabilisticMethod.GAUSSIAN.value, PostHocProbabilisticMethod.CONFORMAL.value]:
        if method == PostHocProbabilisticMethod.CONFORMAL.value:
            nonconformity_score_list = [NonconformityScores.ABSOLUTE_ERROR.value, NonconformityScores.ERROR.value]
        else:
            nonconformity_score_list = [NonconformityScores.ABSOLUTE_ERROR.value]
        for nonconformity_score in nonconformity_score_list:
            # print(method,nonconformity_score )
            p = PostHocProbabilisticProcessor(
                window_size=window_size, quantiles=quantiles, nonconformity_score=nonconformity_score, method=method
            )
            p.train(y_cal_gt=y_cal_gt, y_cal_pred=y_cal_pred)
            y_test_prob_pred = p.predict(y_test_pred)

            ### ASSERTIONS ###

            ## Attributes checked correctly
            assert (
                p.method == method
            ), " method attribute wasnt assigned properly for method {} nonconformity score {}".format(
                method, nonconformity_score
            )
            assert (
                p.window_size == window_size
            ), " window_size attribute wasnt assigned properly for method {} nonconformity score {}".format(
                method, nonconformity_score
            )
            assert (
                p.quantiles == quantiles
            ), " quantiles attribute wasnt assigned properly for method {} nonconformity score {}".format(
                method, nonconformity_score
            )
            assert (
                p.nonconformity_score == nonconformity_score
            ), " nonconformity_score attribute wasnt assigned properly for method {} nonconformity score {}".format(
                method, nonconformity_score
            )

            assert (
                p.model.window_size == window_size
            ), " window_size attribute of processors model wasnt assigned properly for method {} nonconformity score {}".format(
                method, nonconformity_score
            )

            ## Check based on method
            if method == PostHocProbabilisticMethod.GAUSSIAN.value:
                assert isinstance(
                    p.model, PostHocGaussian
                ), "model is not an instance of PostHocGaussian for method {} nonconformity score {}".format(
                    method, nonconformity_score
                )

            if method == PostHocProbabilisticMethod.CONFORMAL.value:
                assert isinstance(
                    p.model, WeightedConformalForecasterWrapper
                ), "model is not an instance of WeightedConformalForecasterWrapper for method {} nonconformity score {}".format(
                    method, nonconformity_score
                )

            ## Prediction Output check
            assert isinstance(
                y_test_prob_pred, np.ndarray
            ), "Unexpected output type from predict(), it should be np array for method {} nonconformity score {}".format(
                method, nonconformity_score
            )
            assert y_test_prob_pred.shape == (
                y_test_pred.shape[0],
                forecast_horizon,
                n_features,
                len(quantiles),
            ), "Unexpected output shape from predict() for method {} nonconformity score {}".format(
                method, nonconformity_score
            )

            # Monotonicity check across quantiles
            assert np.all(
                y_test_prob_pred[..., 0] <= y_test_prob_pred[..., 1]
            ), "Quantile 0.1 is not <= 0.5 for method {} nonconformity score {}".format(method, nonconformity_score)
            assert np.all(
                y_test_prob_pred[..., 1] <= y_test_prob_pred[..., 2]
            ), "Quantile 0.5 is not <= 0.9 for method {} nonconformity score {}".format(method, nonconformity_score)


def test_posthoc_probabilistic_processor_online_update():
    """
    1. Generate Synthetic Data
    """
    np.random.seed(42)

    # Parameters
    window_size = 20
    quantiles = [0.1, 0.5, 0.9]

    # Data
    n_samples = 100
    n_cal = window_size
    forecast_horizon = 20
    n_features = 3
    sigma1 = 0.5
    sigma2 = 1.0
    transition_width = 10
    y_pred = np.random.randn(n_samples, forecast_horizon, n_features)

    # noise based on sigmoid transition from sigma1 to sigma2
    x = np.arange(n_samples)
    transition_center = n_samples // 2
    sigmoid = 1 / (1 + np.exp(-(x - transition_center) / (transition_width / 10)))
    sigma_values = sigma1 + (sigma2 - sigma1) * sigmoid
    sigma_expanded = sigma_values[:, None, None]  # shape: (n_samples, 1, 1)
    noise = np.random.normal(0, sigma_expanded, size=y_pred.shape)

    # ground truth
    y_gt = y_pred + noise

    y_cal_pred = y_pred[0:n_cal]
    y_cal_gt = y_gt[0:n_cal]

    y_test_pred = y_pred[n_cal:]
    y_test_gt = y_gt[n_cal:]

    """
    2. Run Methods with updates
    """
    nonconformity_score_list = [NonconformityScores.ABSOLUTE_ERROR.value]
    method_list = [PostHocProbabilisticMethod.CONFORMAL.value, PostHocProbabilisticMethod.GAUSSIAN.value]
    for method in method_list:
        for nonconformity_score in nonconformity_score_list:
            p = PostHocProbabilisticProcessor(
                window_size=window_size, quantiles=quantiles, nonconformity_score=nonconformity_score, method=method
            )

            ### 2. Fit
            p.train(y_cal_gt=y_cal_gt, y_cal_pred=y_cal_pred)

            ### 3. Predict Test (OFFLINE)
            y_test_prob_pred = p.predict(y_test_pred)

            ### 4. Predict Test with update (ONLINE)
            y_test_prob_pred_online = []
            for i in range(y_test_pred.shape[0]):
                # 4.1 Predict
                y_test_prob_pred_online.append(p.predict(y_test_pred[i : i + 1, ...]))
                # 4.2 Update (assume we observed the sample)
                p.update(y_gt=y_test_gt[i : i + 1, ...], y_pred=y_test_pred[i : i + 1, ...])

            y_test_prob_pred_online = np.stack(y_test_prob_pred_online)
            y_test_prob_pred_online = y_test_prob_pred_online.reshape(
                -1,
                y_test_prob_pred_online.shape[-3],
                y_test_prob_pred_online.shape[-2],
                y_test_prob_pred_online.shape[-1],
            )
            """
            Evaluation
            """
            coverage_online = []
            coverage_offline = []
            for q in range(len(quantiles)):
                covered_online = y_test_gt <= y_test_prob_pred_online[..., q]
                covered_offline = y_test_gt <= y_test_prob_pred[..., q]
                coverage_online.append(covered_online.mean(axis=(1, 2)))
                coverage_offline.append(covered_offline.mean(axis=(1, 2)))
            coverage_online = np.array(coverage_online).transpose()
            coverage_offline = np.array(coverage_offline).transpose()

            # coverage error
            error_online = np.abs(coverage_online - quantiles)
            error_offline = np.abs(coverage_offline - quantiles)

            mean_error_online = error_online.mean()
            mean_error_offline = error_offline.mean()

            # print(f"Mean Coverage Error (Online):  {mean_error_online:.4f}")
            # print(f"Mean Coverage Error (Offline): {mean_error_offline:.4f}")

            ### ASSERTIONS ###

            ## Prediction Output check
            assert isinstance(
                y_test_prob_pred, np.ndarray
            ), "Unexpected output type from predict(), it should be np array for method {} nonconformity score {}".format(
                method, nonconformity_score
            )
            assert y_test_prob_pred.shape == (
                y_test_pred.shape[0],
                forecast_horizon,
                n_features,
                len(quantiles),
            ), "Unexpected output shape from predict() for method {} nonconformity score {}".format(
                method, nonconformity_score
            )

            assert isinstance(
                y_test_prob_pred_online, np.ndarray
            ), "Unexpected output type from predict(), it should be np array for method {} nonconformity score {}".format(
                method, nonconformity_score
            )
            assert y_test_prob_pred_online.shape == (
                y_test_pred.shape[0],
                forecast_horizon,
                n_features,
                len(quantiles),
            ), "Unexpected output shape from predict() for method {} nonconformity score {}".format(
                method, nonconformity_score
            )

            ## Expected Behaviour of an online approach
            assert (
                mean_error_online <= mean_error_offline
            ), "Mean coverage error of online method should smaller that offline for method {} nonconformity score {}".format(
                method, nonconformity_score
            )


def test_posthoc_probabilistic_processor_outlier_score():
    """
    1. Generate Synthetic Data
    """
    np.random.seed(42)
    n = 250
    alarm_rate = 0.01
    window_size_min = int(np.ceil(1 / alarm_rate))
    size = int((n - window_size_min) * alarm_rate)

    t = np.linspace(0, 1, n)
    trend = 3 * t + 2 * t**2 - t**3
    baseline = 0.5 * np.sin(2 * np.pi * 5 * t)
    sigma = 0.2
    noise = np.random.normal(0, sigma, size=n)

    # Generate Signal
    signal = trend + baseline + noise

    # Inject only point anomalies
    anomaly_indices = np.random.choice(np.arange(window_size_min, n), size=size, replace=False)
    anomaly_magnitudes = 1.5
    anomaly_signs = np.random.choice([-1, 1], size=size)
    signal[anomaly_indices] += anomaly_magnitudes * anomaly_signs

    # Labels
    labels = np.zeros(n)
    labels[anomaly_indices] = 1

    # Naive Forecaster
    horizon = 1
    n_forecast = len(signal) - horizon

    # Create naive predictions
    y_pred_naive = np.tile(signal[:n_forecast], (horizon, 1)).T  # (n_forecast, horizon)
    y_pred_naive = y_pred_naive[..., np.newaxis]  # added feature dimension

    # Ground truth for evaluation
    y_true = np.array([signal[t + 1 : t + 1 + horizon] for t in range(n_forecast)])  # (n_forecast, horizon)
    y_true = y_true[..., np.newaxis]  # added feature dimension

    # Cal/Test Splits
    window_size = window_size_min
    y_cal_gt = y_true[0:window_size]
    y_cal_pred = y_pred_naive[0:window_size]
    y_test_pred = y_pred_naive[window_size:]
    y_test_gt = y_true[window_size:]

    quantiles = [alarm_rate / 2, 1 - alarm_rate / 2]

    """
    2. Run Method outlier score
    """
    nonconformity_score_list = [NonconformityScores.ABSOLUTE_ERROR.value]
    method_list = [PostHocProbabilisticMethod.CONFORMAL.value]
    for method in method_list:
        for nonconformity_score in nonconformity_score_list:
            p = PostHocProbabilisticProcessor(
                window_size=window_size, quantiles=quantiles, nonconformity_score=nonconformity_score, method=method
            )

            ### 2. Fit
            p.train(y_cal_gt=y_cal_gt, y_cal_pred=y_cal_pred)

            ### 3. Outlier
            output_outlier = p.outlier_score(
                y_pred=y_test_pred, y_gt=y_test_gt, significance=alarm_rate, aggregation=0
            )

            labels_test = labels[-output_outlier.shape[0] :]
            labels_prediction_test = output_outlier[..., 1].flatten()
            p_value_scores_test = output_outlier[..., 0].flatten()

            ### ASSERTIONS ###

            ## Prediction Output check
            assert isinstance(
                output_outlier, np.ndarray
            ), "Unexpected output type from outlier_score(), it should be np array for method {} nonconformity score {}".format(
                method, nonconformity_score
            )
            assert output_outlier.shape == (
                y_test_pred.shape[0],
                y_test_pred.shape[2],
                2,
            ), "Unexpected output shape from predict() for method {} nonconformity score {}".format(
                method, nonconformity_score
            )

            ### Expected Behaviour of an outlier approach ###

            # highest p-value for predicted outliers is below the alarm rate
            assert (
                np.max(p_value_scores_test[labels_prediction_test == 1]) <= alarm_rate
            ), "Max p value among predicted outliers exceeds alarm rate for method {} nonconformity score {}".format(
                method, nonconformity_score
            )

            # false positive rate is below the alarm rate
            assert (
                np.mean(labels_prediction_test[labels_test == 0]) <= alarm_rate
            ), "False positive rate exceeds alarm rate for method {} nonconformity score {}".format(
                method, nonconformity_score
            )

            # true positive rate is at least (1 - alarm_rate)
            assert (
                np.mean(labels_prediction_test[labels_test == 1]) >= 1 - alarm_rate
            ), "True positive rate is lower than expectedfor method {} nonconformity score {}".format(
                method, nonconformity_score
            )


# if __name__ == "__main__":
#     test_posthoc_probabilistic_processor_outlier_score()
