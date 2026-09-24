"""Tests of the shared result contract for ensemble aggregation."""

import unittest
from types import SimpleNamespace

import numpy as np

from tsfm_public.models.ensemble.modeling_ensemble import QuantileEnsembleForecaster
from tsfm_public.toolkit.ensemble_aggregation import (
    aggregate_iqr_weighted,
    aggregate_linear_pool,
    aggregate_vincent,
)
from tsfm_public.toolkit.forecasters import ForecastResult


class TestEnsembleForecastResult(unittest.TestCase):
    def setUp(self):
        self.levels = [0.1, 0.5, 0.9]
        self.predictions = np.array([0.0, 1.0, 2.0]).reshape(1, 1, 1, 3, 1)
        self.predictions = np.repeat(self.predictions, 2, axis=-1)
        self.aggregators = (aggregate_linear_pool, aggregate_vincent, aggregate_iqr_weighted)

    def test_shared_result_and_forecast_values(self):
        for aggregate in self.aggregators:
            with self.subTest(method=aggregate.__name__):
                result = aggregate(self.predictions, self.levels)
                self.assertIsInstance(result, ForecastResult)
                self.assertTrue(result.success)
                np.testing.assert_allclose(result.predicted, [[[1.0]]])
                np.testing.assert_allclose(result.predicted_quantiles, [[[[0.0, 1.0, 2.0]]]])
                self.assertEqual(result.metadata["method"], aggregate.__name__)
                self.assertEqual(result.metadata["quantile_levels"], self.levels)
                self.assertEqual(result.metadata["n_models"], 2)
                self.assertIsNone(result.actuals)
                self.assertIsNone(result.cutoff_dates)

    def test_validation_failures_use_shared_result(self):
        for aggregate in self.aggregators:
            with self.subTest(method=aggregate.__name__):
                result = aggregate(self.predictions, [])
                self.assertIsInstance(result, ForecastResult)
                self.assertFalse(result.success)
                self.assertIsNone(result.predicted)

    def test_method_is_preserved_on_aggregation_specific_failures(self):
        for aggregate in (aggregate_vincent, aggregate_iqr_weighted):
            with self.subTest(method=aggregate.__name__):
                result = aggregate(self.predictions, [0.5])
                self.assertFalse(result.success)
                self.assertEqual(result.metadata["method"], aggregate.__name__)

    def test_ensemble_returns_shared_result(self):
        member = SimpleNamespace(forecast_for_ensemble=lambda data, **kwargs: self.predictions[..., 0])
        ensemble = QuantileEnsembleForecaster([member], self.levels)
        result = ensemble(None)
        self.assertIsInstance(result, ForecastResult)
        self.assertTrue(result.success)
        self.assertEqual(result.metadata["method"], "aggregate_linear_pool")
        np.testing.assert_allclose(result.predicted, [[[1.0]]])

    def test_ensemble_reports_when_all_members_fail(self):
        def fail(data, **kwargs):
            raise ValueError("member failure")

        member = SimpleNamespace(forecast_for_ensemble=fail)
        ensemble = QuantileEnsembleForecaster([member], self.levels)

        with self.assertRaisesRegex(
            RuntimeError, "All ensemble members failed to produce forecasts"
        ):
            ensemble(None)


if __name__ == "__main__":
    unittest.main()
