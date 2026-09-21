# Copyright contributors to the TSFM project
#
"""Offline checks for ensemble recipe loading and construction."""

import tempfile
import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from tsfm_public.models.ensemble.configuration_ensemble import ProbabilisticEnsembleConfig
from tsfm_public.models.ensemble import modeling_ensemble as modeling
from tsfm_public.toolkit.ensemble_aggregation import aggregate_iqr_weighted, aggregate_linear_pool


class TestEnsembleConfig(unittest.TestCase):
    def test_save_and_reload_modified_recipe(self):
        config = ProbabilisticEnsembleConfig(
            aggregation_method="iqr_weighted",
            quantile_levels=[0.1, 0.5, 0.9],
            weights=[0.4, 0.3, 0.2, 0.1],
        )
        config.iqr_weighted_options["temperature"] = 0.8
        with tempfile.TemporaryDirectory() as directory:
            config.save_pretrained(directory)
            loaded = ProbabilisticEnsembleConfig.from_pretrained(directory)
        for field in ("members", "quantile_levels", "aggregation_method", "iqr_weighted_options", "weights"):
            self.assertEqual(getattr(loaded, field), getattr(config, field))

    def test_defaults_are_independent(self):
        first, second = ProbabilisticEnsembleConfig(), ProbabilisticEnsembleConfig()
        first.members[0]["model_checkpoint"] = "custom/model"
        first.iqr_weighted_options["temperature"] = 0.8
        self.assertNotEqual(first.members, second.members)
        self.assertEqual(second.iqr_weighted_options["temperature"], 0.5)

    def test_invalid_recipes_fail(self):
        cases = [
            {"aggregation_method": "unknown"},
            {"members": []},
            {"members": [{"forecaster_type": "unknown", "model_checkpoint": "model"}]},
            {"members": [{"forecaster_type": "patchtst"}]},
            {"members": [{"forecaster_type": "patchtst", "model_checkpoint": "model", "typo": True}]},
            {"members": [{"forecaster_type": "ttm", "model_checkpoint": "model", "model_revision": "fixed"}]},
            {"quantile_levels": [0.1, 0.9]},
            {"quantile_levels": [0.5, 0.1, 0.9]},
            {"iqr_weighted_options": {"temperature": 0}},
            {"iqr_weighted_options": {"max_weights": 0.4}},
            {"aggregation_method": "iqr_weighted", "iqr_weighted_options": {"max_weight": 0.1}},
            {"weights": [1.0]},
            {"weights": [0.1, 0.1, 0.1, 0.1]},
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ProbabilisticEnsembleConfig(**kwargs)

    def test_member_changes_validate_cap_only_when_iqr_is_selected(self):
        config = ProbabilisticEnsembleConfig()
        config.members = config.members[:2]
        config.validate()
        config.aggregation_method = "iqr_weighted"
        with self.assertRaisesRegex(ValueError, "1 / n_members"):
            config.validate()
        config.iqr_weighted_options["max_weight"] = 0.6
        config.validate()

    def test_factory_revalidates_mutated_config_before_loading(self):
        config = ProbabilisticEnsembleConfig()
        config.aggregation_method = "unknown"
        with patch.object(modeling, "PatchTSTFMDataFramePipelineForecaster") as loader:
            with self.assertRaises(ValueError):
                modeling.QuantileEnsembleForecaster.from_config(config)
            loader.assert_not_called()

    def test_factory_matches_explicit_construction(self):
        for method in ("linear_pool", "iqr_weighted"):
            with self.subTest(method=method):
                config = ProbabilisticEnsembleConfig(
                    aggregation_method=method,
                    quantile_levels=[0.1, 0.5, 0.9],
                    weights=[0.4, 0.3, 0.2, 0.1],
                )
                def member(offset):
                    def forecast(data, quantile_levels):
                        return (offset + np.asarray(quantile_levels)).reshape(1, 1, 1, -1)
                    return SimpleNamespace(forecast_for_ensemble=forecast)
                members = [member(i) for i in range(4)]
                with patch.object(
                    modeling, "PatchTSTFMDataFramePipelineForecaster", side_effect=members[:2]
                ) as patchtst, patch.object(
                    modeling, "FlowStateDataFramePipelineForecaster", return_value=members[2]
                ) as flowstate, patch.object(
                    modeling, "TinyTimeMixerDataFramePipelineForecaster", return_value=members[3]
                ) as ttm:
                    configured = modeling.QuantileEnsembleForecaster.from_config(config, device="cpu")
                self.assertEqual(configured.members, members)
                self.assertEqual(patchtst.call_args_list[0].kwargs["model_checkpoint"], config.members[0]["model_checkpoint"])
                self.assertEqual(patchtst.call_args_list[1].kwargs["model_checkpoint"], config.members[1]["model_checkpoint"])
                flowstate.assert_called_once_with(
                    device="cpu", model_checkpoint=config.members[2]["model_checkpoint"], model_revision="r1.1"
                )
                ttm.assert_called_once_with(device="cpu", model_checkpoint=config.members[3]["model_checkpoint"])
                function = aggregate_linear_pool if method == "linear_pool" else partial(
                    aggregate_iqr_weighted, **config.iqr_weighted_options
                )
                explicit = modeling.QuantileEnsembleForecaster(
                    members, config.quantile_levels, function, weights=np.asarray(config.weights)
                )
                result, expected = configured(None), explicit(None)
                self.assertTrue(result.success)
                np.testing.assert_allclose(result.predicted_quantiles, expected.predicted_quantiles)
                np.testing.assert_allclose(result.predicted, expected.predicted)
                self.assertEqual(result.metadata, expected.metadata)


if __name__ == "__main__":
    unittest.main()
