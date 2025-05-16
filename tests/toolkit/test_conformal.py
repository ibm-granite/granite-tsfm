# Copyright contributors to the TSFM project
#

"""Tests conformal processor capabilities"""

import tempfile
from pathlib import Path
import numpy as np

from tsfm_public.toolkit.conformal import PostHocProbabilisticProcessor,PostHocGaussian,WeightedConformalForecasterWrapper


def test_conformal_save_pretrained():
    # initial test to check that we save the ProbabbilisticProcessor as intended
    p = PostHocProbabilisticProcessor()

    with tempfile.TemporaryDirectory() as d:
        p.save_pretrained(d)
        p_new = PostHocProbabilisticProcessor.from_pretrained(d)
        assert Path(d).joinpath(PostHocProbabilisticProcessor.PROCESSOR_NAME).exists()

        # to do: add checks that p and p_new are equivalent

def test_posthoc_probabilistic_processor():
    np.random.seed(42)

    # Parameters
    window_size = 10
    quantiles = [0.1, 0.5, 0.9]
    nonconformity_score = 'absolute_error'

    # Data
    n_samples = 30
    forecast_horizon = 5
    n_features = 3

    y_pred = np.random.randn(n_samples, forecast_horizon, n_features)
    y_gt = y_pred + np.random.normal(0, 0.5, size=y_pred.shape)

    y_cal_pred = y_pred[0:20]
    y_cal_gt = y_gt[0:20]
    y_test_pred = y_pred[-10:]
    y_test_gt = y_gt[-10:]

    for method in ['gaussian','conformal']:
        print(method)
        p = PostHocProbabilisticProcessor(
            window_size=window_size,
            quantiles=quantiles,
            nonconformity_score=nonconformity_score,
            method=method
            )
        p.train(y_cal_gt=y_cal_gt,y_cal_pred=y_cal_pred)
        y_test_prob_pred = p.predict(y_test_pred)
        # print(y_test_prob_pred)


        ### ASSERTIONS ###

        ## Attributes checked correctly
        assert p.method == method, ' method attribute wasnt assigned properly'
        assert p.window_size == window_size, ' window_size attribute wasnt assigned properly'
        assert p.quantiles == quantiles, ' quantiles attribute wasnt assigned properly'
        assert p.nonconformity_score == nonconformity_score, ' nonconformity_score attribute wasnt assigned properly'

        assert p.model.window_size == window_size, ' window_size attribute of processors model wasnt assigned properly'

        ## Check based on method
        if method == 'gaussian':
            assert isinstance(p.model, PostHocGaussian), "model is not an instance of PostHocGaussian"

        if method == 'conformal':
            assert isinstance(p.model, WeightedConformalForecasterWrapper), "model is not an instance of WeightedConformalForecasterWrapper"

        ## Prediction Output check
        assert isinstance(y_test_prob_pred, np.ndarray), "Unexpected output type from predict(), it should be np array"
        assert y_test_prob_pred.shape == (
            y_test_pred.shape[0], forecast_horizon, n_features, len(quantiles)
        ), "Unexpected output shape from predict()"

        # Monotonicity check across quantiles
        assert np.all(y_test_prob_pred[..., 0] <= y_test_prob_pred[..., 1]), "Quantile 0.1 is not <= 0.5"
        assert np.all(y_test_prob_pred[..., 1] <= y_test_prob_pred[..., 2]), "Quantile 0.5 is not <= 0.9"



# if __name__ == '__main__':
#     test_posthoc_probabilistic_processor()
