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


# if __name__ == '__main__':
# test_posthoc_probabilistic_processor()
# test_posthoc_probabilistic_processor_online_update()
