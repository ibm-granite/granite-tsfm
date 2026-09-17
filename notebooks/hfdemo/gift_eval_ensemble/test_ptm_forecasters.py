"""Basic tests of PreTrainedModel forecasters

Run from this directory like this:
$ uv run pytest test_ptm_forecasters.py

"""

from types import SimpleNamespace

import numpy as np
import torch

from ptm_forecasters import (
    FlowstateGiftModelForecaster,
    PatchTSTFMGiftModelForecaster,
    RecordingForecaster,
    TinyTimeMixerPreTrainedGiftModelForecaster,
    _TTMContextScaler,
    _single_member_result,
    build_gift_ensemble,
)
from tsfm_public.models.ensemble.modeling_ensemble import QuantileEnsembleTimeSeriesForecast

N_QUANTILES = 9  # [0.1, 0.2, ..., 0.9]


class FakeTTM:
    def __init__(self):
        self.config = SimpleNamespace(context_length=4, resolution_prefix_tuning=True)
        self.inputs = None

    def eval(self):
        return self

    def __call__(self, **kwargs):
        self.inputs = kwargs
        quantiles = torch.ones((1, N_QUANTILES, 2, 1))
        return SimpleNamespace(quantile_outputs=quantiles)


def get_sample_data(n_samples=100, period=24, n_targets=1):
    """Generate synthetic sinusoidal time-series data."""
    t = np.arange(n_samples)
    data = []
    for i in range(n_targets):
        phase_shift = i * np.pi / 4
        amplitude = 1.0 + i * 0.2
        baseline = 20.0 + i * 10.0
        y = baseline + amplitude * np.sin(2 * np.pi * t / period + phase_shift)
        data.append(y.tolist())
    return data


def get_sample_data_with_nans(n_samples=100, period=24, n_targets=1, nan_indices=(5, 10, 50)):
    """Generate synthetic sinusoidal time-series data with NaN values at specified indices."""
    data = get_sample_data(n_samples=n_samples, period=period, n_targets=n_targets)
    data_with_nans = []
    for series in data:
        arr = np.array(series, dtype=float)
        arr[list(nan_indices)] = np.nan
        data_with_nans.append(arr.tolist())
    return data_with_nans

# ---------------------------------------------------------------------------
# PatchTSTFMGiftModelForecaster
# ---------------------------------------------------------------------------

def test_patchtstfm_call():
    for model_version in ["patchtst-fm-r1", "granite-patchtst-fm-r1", "granite-patchtst-fm-r2"]:
        # use_fill_nan=False (default): clean data, no NaN filling applied
        forecaster = PatchTSTFMGiftModelForecaster(device="cpu", model_version=model_version, use_fill_nan=False)
        data = get_sample_data(n_samples=100, period=24, n_targets=1)
        fcast = forecaster(data, prediction_length=[20])

        assert fcast is not None
        assert len(fcast) == 1, "one entry per target"
        assert len(fcast[0]["median"]) == 20, "prediction length is correct"
        assert "quantile_levels" in fcast[0]
        assert len(fcast[0]["quantile_levels"]) == N_QUANTILES

        # use_fill_nan=True: data contains NaNs, forecaster should fill them and still produce valid output
        forecaster_fill = PatchTSTFMGiftModelForecaster(device="cpu", model_version=model_version, use_fill_nan=True)
        data_with_nans = get_sample_data_with_nans(n_samples=100, period=24, n_targets=1)
        fcast_fill = forecaster_fill(data_with_nans, prediction_length=[20])

        assert fcast_fill is not None
        assert len(fcast_fill) == 1, "one entry per target (use_fill_nan=True)"
        assert len(fcast_fill[0]["median"]) == 20, "prediction length is correct (use_fill_nan=True)"
        assert "quantile_levels" in fcast_fill[0]
        assert len(fcast_fill[0]["quantile_levels"]) == N_QUANTILES
        assert not np.any(np.isnan(fcast_fill[0]["median"])), "median output must not contain NaNs"


def test_patchtstfm_forecast_for_ensemble():
    for model_version in ["patchtst-fm-r1","granite-patchtst-fm-r1","granite-patchtst-fm-r2"]:
        forecaster = PatchTSTFMGiftModelForecaster(device="cpu",model_version = model_version)
        n_series = 3
        pred_len = 12
        data = get_sample_data(n_samples=100, period=24, n_targets=n_series)
        prediction_length = [pred_len] * n_series

        arr = forecaster.forecast_for_ensemble(data, prediction_length=prediction_length)

        assert isinstance(arr, np.ndarray)
        assert arr.shape == (n_series, pred_len, 1, N_QUANTILES), (
            f"Expected ({n_series}, {pred_len}, 1, {N_QUANTILES}), got {arr.shape}"
        )


# ---------------------------------------------------------------------------
# TinyTimeMixerPreTrainedGiftModelForecaster
# ---------------------------------------------------------------------------

def test_ttm_leaderboard_preprocessing():
    scaler = _TTMContextScaler(
        [
            {"item_id": "a", "target": np.array([1.0, 2.0, 3.0])},
            {"item_id": "a", "target": np.array([2.0, 4.0, 6.0])},
        ]
    )
    forecaster = TinyTimeMixerPreTrainedGiftModelForecaster.__new__(
        TinyTimeMixerPreTrainedGiftModelForecaster
    )
    forecaster.quantile_levels = [index / 10 for index in range(1, 10)]
    forecaster.ix_median = 4
    forecaster.model = FakeTTM()
    forecaster.max_context_length = 4
    forecaster.device = "cpu"
    forecaster.freq = "D"
    forecaster.scaler = scaler

    # A later rolling window must not alter normalization at this origin.
    scaler.mean["a"][:] = 10000
    scaler.std["a"][:] = 1000
    result = forecaster([[2.0, 4.0, 6.0]], [2], series_ids=["a"])

    np.testing.assert_array_equal(
        forecaster.model.inputs["past_observed_mask"].numpy().reshape(-1),
        [False, True, True, True],
    )
    np.testing.assert_allclose(
        forecaster.model.inputs["past_values"].numpy().reshape(-1),
        [0.0, -1.2247449, 0.0, 1.2247449],
    )
    assert forecaster.model.inputs["freq_token"].item() == 8
    np.testing.assert_allclose(result[0]["median"], [4.0 + np.std([2.0, 4.0, 6.0])] * 2)


def test_recording_forecaster_only_forwards_series_ids_when_supported():
    class FakeForecaster:
        def forecast_for_ensemble(self, data, prediction_length):
            return np.zeros((1, prediction_length[0], 1, N_QUANTILES))

    wrapped = RecordingForecaster("fake", FakeForecaster())
    result = wrapped.forecast_for_ensemble(
        [[1.0, 2.0]], prediction_length=[2], series_ids=["a"]
    )

    assert result.shape == (1, 2, 1, N_QUANTILES)


def test_single_member_result_preserves_quantiles():
    quantiles = np.arange(18, dtype=float).reshape(1, 2, 1, N_QUANTILES)
    result = _single_member_result(
        quantiles[..., np.newaxis],
        [index / 10 for index in range(1, 10)],
    )

    np.testing.assert_array_equal(result.predicted_quantiles, quantiles)
    np.testing.assert_array_equal(result.predicted, quantiles[..., 4])

def test_ttm_call():
    data = get_sample_data(n_samples=100, period=24, n_targets=1)
    pred_len = 20
    context_length = 100
    forecaster = TinyTimeMixerPreTrainedGiftModelForecaster(
        model_version="ttm-r3-pt", device="cpu", use_get_gift_model=True,context_length=context_length, prediction_length = pred_len
    )
    fcast = forecaster(data, prediction_length=[pred_len])

    assert fcast is not None
    assert len(fcast) == 1, "one entry per target"
    assert len(fcast[0]["median"]) == 20, "prediction length is correct"
    assert "quantile_levels" in fcast[0]
    assert len(fcast[0]["quantile_levels"]) == N_QUANTILES


def test_ttm_forecast_for_ensemble():
    
    n_series = 3
    pred_len = 12
    data = get_sample_data(n_samples=100, period=24, n_targets=n_series)
    prediction_length = [pred_len] * n_series
    context_length = 100

    forecaster = TinyTimeMixerPreTrainedGiftModelForecaster(model_version="ttm-r3-pt", device="cpu", use_get_gift_model=True,context_length=context_length, prediction_length = pred_len)
    arr = forecaster.forecast_for_ensemble(data, prediction_length=prediction_length)

    assert isinstance(arr, np.ndarray)
    assert arr.shape == (n_series, pred_len, 1, N_QUANTILES), (
        f"Expected ({n_series}, {pred_len}, 1, {N_QUANTILES}), got {arr.shape}"
    )


# ---------------------------------------------------------------------------
# FlowstateGiftModelForecaster
# ---------------------------------------------------------------------------

def test_flowstate_call():
    for model_version in ["flowstate-r1.1","granite-flowstate-r1.1"]:
        forecaster = FlowstateGiftModelForecaster(
            model_version=model_version, device="cpu"
        )
        data = get_sample_data(n_samples=100, period=24, n_targets=1)
        fcast = forecaster(data, prediction_length=[20])

        assert fcast is not None
        assert len(fcast) == 1, "one entry per target"
        assert len(fcast[0]["median"]) == 20, "prediction length is correct"
        assert "quantile_levels" in fcast[0]
        assert len(fcast[0]["quantile_levels"]) == N_QUANTILES


def test_flowstate_call_with_freq():
    """Verify that passing freq_str at call time (override) works."""
    for model_version in ["flowstate-r1.1","granite-flowstate-r1.1"]:
        forecaster = FlowstateGiftModelForecaster(
            model_version=model_version, device="cpu", freq="H"
        )
        data = get_sample_data(n_samples=100, period=24, n_targets=1)
        # Override with a different freq at call time
        fcast = forecaster(data, prediction_length=[20], freq_str="D")

        assert fcast is not None
        assert len(fcast[0]["median"]) == 20


def test_flowstate_forecast_for_ensemble():
    for model_version in ["flowstate-r1.1","granite-flowstate-r1.1"]:
        forecaster = FlowstateGiftModelForecaster(
            model_version=model_version, device="cpu"
        )
        n_series = 3
        pred_len = 12
        data = get_sample_data(n_samples=100, period=24, n_targets=n_series)
        prediction_length = [pred_len] * n_series

        arr = forecaster.forecast_for_ensemble(data, prediction_length=prediction_length)

        assert isinstance(arr, np.ndarray)
        assert arr.shape == (n_series, pred_len, 1, N_QUANTILES), (
            f"Expected ({n_series}, {pred_len}, 1, {N_QUANTILES}), got {arr.shape}"
        )


# ---------------------------------------------------------------------------
# build_gift_ensemble
# ---------------------------------------------------------------------------

_ENSEMBLE_CANDIDATE_MODELS = [
    "granite-patchtst-fm-r2",
    "granite-flowstate-r1.1",
]


def test_build_gift_ensemble_returns_correct_type():
    """build_gift_ensemble returns a QuantileEnsembleTimeSeriesForecast with the
    expected number of members and quantile levels."""
    ensemble = build_gift_ensemble(
        candidate_models=_ENSEMBLE_CANDIDATE_MODELS,
        device="cpu",
        freq="H",
    )

    assert isinstance(ensemble, QuantileEnsembleTimeSeriesForecast)
    assert len(ensemble.members) == len(_ENSEMBLE_CANDIDATE_MODELS), (
        f"Expected {len(_ENSEMBLE_CANDIDATE_MODELS)} members, got {len(ensemble.members)}"
    )
    assert ensemble.quantile_levels == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def test_build_gift_ensemble_patchtst_use_fill_nan():
    """PatchTST member respects patchtst_use_fill_nan for both True and False."""
    for use_fill_nan in [False, True]:
        ensemble = build_gift_ensemble(
            candidate_models=_ENSEMBLE_CANDIDATE_MODELS,
            device="cpu",
            freq="H",
            patchtst_use_fill_nan=use_fill_nan,
        )

        patchtst_member = next(
            m for m in ensemble.members if "patchtst" in m.model_name
        )
        assert patchtst_member.forecaster.use_fill_nan is use_fill_nan, (
            f"Expected use_fill_nan={use_fill_nan}, got {patchtst_member.forecaster.use_fill_nan}"
        )


def test_build_gift_ensemble_call():
    """End-to-end: built ensemble can be called and returns valid quantile predictions."""
    n_series = 2
    pred_len = 12
    data = get_sample_data(n_samples=100, period=24, n_targets=n_series)
    prediction_length = [pred_len] * n_series

    ensemble = build_gift_ensemble(
        candidate_models=_ENSEMBLE_CANDIDATE_MODELS,
        device="cpu",
        freq="H",
        patchtst_use_fill_nan=True,
    )

    result = ensemble(data, prediction_length=prediction_length)
    assert result is not None
    assert np.array(result.predicted).shape == (n_series, pred_len, 1), (
        f"Expected predicted shape ({n_series}, {pred_len}, 1), got {np.array(result.predicted).shape}"
    )
    assert np.array(result.predicted_quantiles).shape == (n_series, pred_len, 1, N_QUANTILES), (
        f"Expected quantiles shape ({n_series}, {pred_len}, 1, {N_QUANTILES}), got {np.array(result.predicted_quantiles).shape}"
    )
    assert not np.any(np.isnan(np.array(result.predicted))), "predicted (median) must not contain NaNs"


# GIFT-Eval window generation tests (no model downloads).
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from gift_eval_windows import get_gift_ensemble_predictions_df, get_test_window_lengths


class FakeTestData:
    def __init__(self, inputs, labels):
        self.input = inputs
        self.label = labels

    def __len__(self):
        return len(self.input)


class FakeDataset:
    def __init__(self, inputs, labels):
        self.test_data = FakeTestData(inputs, labels)


def entry(target, start, frequency="D", item_id=None):
    value = {
        "target": np.asarray(target, dtype=float),
        "start": pd.Period(start, freq=frequency),
        "freq": frequency,
    }
    if item_id is not None:
        value["item_id"] = item_id
    return value


class RunGiftEvalTest(unittest.TestCase):
    def test_get_test_window_lengths(self):
        dataset = FakeDataset(
            inputs=[entry([1, 2, 3], "2020-01-01"), entry([4, 5], "2020-01-02")],
            labels=[entry([6, 7], "2020-01-04"), entry([8, 9, 10], "2020-01-04")],
        )

        self.assertEqual(get_test_window_lengths(dataset), (2, 3))

    def test_get_gift_ensemble_predictions_df_uses_generated_windows(self):
        inputs = [
            entry([1, np.nan, 3], "2020-01-01", item_id="a"),
            entry([4, 5], "2020-01-02", item_id="b"),
        ]
        labels = [entry([6, 7], "2020-01-04"), entry([8, 9], "2020-01-04")]
        dataset = FakeDataset(inputs=inputs, labels=labels)
        calls = []

        def model_pipeline(data, prediction_length, series_ids):
            calls.append((data, prediction_length, series_ids))
            horizon = prediction_length[0]
            quantiles = np.arange(horizon * 9, dtype=float).reshape(1, horizon, 1, 9)
            return SimpleNamespace(
                metadata={"quantile_levels": [index / 10 for index in range(1, 10)]},
                predicted_quantiles=quantiles,
            )

        frame = get_gift_ensemble_predictions_df(dataset, model_pipeline)

        self.assertEqual(len(frame), 2)
        self.assertEqual(calls[0][1], [2])
        self.assertEqual(calls[0][2], ["a"])
        np.testing.assert_allclose(calls[0][0][0], inputs[0]["target"], equal_nan=True)
        self.assertEqual(frame["future_start"].tolist(), [label["start"] for label in labels])
        self.assertEqual(frame["frequency"].tolist(), ["D", "D"])
        self.assertEqual(
            list(frame.iloc[0]["final_pred"]),
            [f"quantile_{index}" for index in range(9)],
        )

    def test_get_gift_ensemble_predictions_df_includes_members(self):
        inputs = [entry([1, 2, 3], "2020-01-01")]
        labels = [entry([4, 5], "2020-01-04")]
        dataset = FakeDataset(inputs=inputs, labels=labels)

        class FakeMember:
            model_name = "model-a"
            last_forecast = None

        class FakePipeline:
            def __init__(self):
                self.members = [FakeMember()]

            def __call__(self, data, prediction_length):
                horizon = prediction_length[0]
                quantiles = np.arange(horizon * 9, dtype=float).reshape(1, horizon, 1, 9)
                self.members[0].last_forecast = quantiles + 1
                return SimpleNamespace(
                    metadata={"quantile_levels": [index / 10 for index in range(1, 10)]},
                    predicted_quantiles=quantiles,
                )

        ensemble_frame, member_frames = get_gift_ensemble_predictions_df(
            dataset,
            FakePipeline(),
            include_member_forecasts=True,
        )

        self.assertEqual(len(ensemble_frame), 1)
        self.assertEqual(set(member_frames), {"model-a"})
        self.assertEqual(len(member_frames["model-a"]), 1)




class EvaluationInterfaceTest(unittest.TestCase):
    """Check the shared notebook/CLI entry point without model downloads."""

    def test_device_selection_and_cli(self):
        from unittest.mock import patch
        import run_gift_eval as runner
        with patch.object(runner.torch.cuda, "is_available", return_value=False), \
             patch.object(runner.torch.backends.mps, "is_available", return_value=False):
            self.assertEqual(runner.resolve_device(), "cpu")
            self.assertEqual(runner.resolve_device("cpu"), "cpu")
            with self.assertRaisesRegex(ValueError, "CUDA is unavailable"):
                runner.resolve_device("cuda")
        with patch.object(runner.torch.cuda, "is_available", return_value=True):
            self.assertEqual(runner.resolve_device(), "cuda")
            self.assertEqual(runner.resolve_device("cpu"), "cpu")
        self.assertEqual(runner.parse_args(["--device", "cpu"]).device, "cpu")
        with self.assertRaises(ValueError):
            runner.resolve_device("invalid")

    def test_callable_results_members_and_resume(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import MagicMock, patch
        import run_gift_eval as runner
        dataset = SimpleNamespace(target_dim=1, freq="M", test_data=[object()])
        metrics = pd.DataFrame([{"dataset": "us_births/M/short", "MSE[mean]": 1.0}])
        fake_adapters = SimpleNamespace(build_gift_ensemble=MagicMock())
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict(runner.CONFIGURATIONS, {"test-single-member": {
                 "model_names": ("granite-patchtst-fm-r2",),
                 "ensemble": "probability_space_aggregation"}}), \
             patch.dict("sys.modules", {"ptm_forecasters": fake_adapters}), \
             patch.object(runner, "Dataset", return_value=dataset), \
             patch.object(runner, "get_gift_ensemble_predictions_df",
                          return_value=(pd.DataFrame(), {"fake-member": pd.DataFrame([{}])})), \
             patch.object(runner, "eval_gift_dataset", return_value=metrics):
            settings = dict(model_name_config="test-single-member", datasets=["us_births/M"],
                            out_dir=directory, device="cpu", save_member_results=True,
                            patchtst_use_fill_nan=True)
            result = runner.run_evaluation(**settings)
            self.assertEqual(result, Path(directory) / "test-single-member/all_results.csv")
            self.assertEqual(len(pd.read_csv(result)), 1)
            member = result.parent / "members/fake-member/all_results.csv"
            self.assertTrue(member.is_file())
            kwargs = fake_adapters.build_gift_ensemble.call_args.kwargs
            self.assertEqual(kwargs["device"], "cpu")
            self.assertTrue(kwargs["patchtst_use_fill_nan"])
            fake_adapters.build_gift_ensemble.reset_mock()
            self.assertEqual(runner.run_evaluation(**settings, skip_processed=True), result)
            fake_adapters.build_gift_ensemble.assert_not_called()


class PatchTSTSourceTest(unittest.TestCase):
    def test_all_versions_use_package_classes_and_keep_input_modes(self):
        from unittest.mock import MagicMock, patch
        from ptm_forecasters import PatchTSTFMGiftModelForecaster
        for version in PatchTSTFMGiftModelForecaster._PATCHTST_MODELS:
            model = MagicMock()
            model.config.context_length = 8192
            with patch("ptm_forecasters.PatchTSTFMConfig") as config_cls, \
                 patch("ptm_forecasters.PatchTSTFMForPrediction") as model_cls, \
                 patch.dict("os.environ", {"HF_TOKEN": "test-token"}):
                model_cls.from_pretrained.return_value.to.return_value = model
                adapter = PatchTSTFMGiftModelForecaster(model_version=version, device="cpu")
                checkpoint = adapter._PATCHTST_MODELS[version]["model_checkpoint"]
                config_cls.from_pretrained.assert_called_once_with(checkpoint, token="test-token")
                model_cls.from_pretrained.assert_called_once_with(
                    checkpoint, config=config_cls.from_pretrained.return_value, token="test-token")
                model_cls.from_pretrained.return_value.to.assert_called_once_with("cpu")
                self.assertIs(adapter.model, model)
                self.assertEqual(adapter.uses_variable_length_input,
                                 version == "granite-patchtst-fm-r2")
                model.eval.assert_called_once()
