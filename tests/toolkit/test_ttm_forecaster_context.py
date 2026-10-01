"""Regression checks for TTM input context and forecast cutoff alignment."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from tsfm_public.toolkit.forecasters import TinyTimeMixerDataFramePipelineForecaster


class TestTTMForecasterContext(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01", periods=4096, freq="h"),
                "target": np.arange(4096, dtype=float),
            }
        )
        model = SimpleNamespace(config=SimpleNamespace(context_length=2048, prediction_length=96))
        with patch("tsfm_public.toolkit.forecasters.TinyTimeMixerForPrediction.from_pretrained", return_value=model):
            self.member = TinyTimeMixerDataFramePipelineForecaster(model_revision="main", device="cpu")

    def test_explicit_main_revision_is_loaded_once_and_preserved(self):
        model = self.member.model
        with (
            patch(
                "tsfm_public.toolkit.forecasters.TinyTimeMixerForPrediction.from_pretrained", return_value=model
            ) as load,
            patch("tsfm_public.toolkit.forecasters.get_model") as select,
            patch("tsfm_public.toolkit.forecasters.TimeSeriesPreprocessor"),
            patch("tsfm_public.toolkit.forecasters.TimeSeriesForecastingPipeline"),
        ):
            member = TinyTimeMixerDataFramePipelineForecaster(model_revision="main", device="cpu")
            for context_length in [512, 1024]:
                member(self.data.tail(context_length), "timestamp", ["target"], 96, context_length=context_length)
            load.assert_called_once_with(member.model_checkpoint, revision="main")
            select.assert_not_called()
            self.assertIs(member.model, model)

    def test_automatic_selection_runs_for_each_requested_context(self):
        with (
            patch(
                "tsfm_public.toolkit.forecasters.TinyTimeMixerForPrediction.from_pretrained",
                return_value=self.member.model,
            ) as load,
            patch("tsfm_public.toolkit.forecasters.get_model", side_effect=["512-96", "1024-96"]) as select,
            patch("tsfm_public.toolkit.forecasters.TimeSeriesPreprocessor"),
            patch("tsfm_public.toolkit.forecasters.TimeSeriesForecastingPipeline"),
        ):
            member = TinyTimeMixerDataFramePipelineForecaster(device="cpu")
            load.assert_not_called()
            for context_length in [512, 1024]:
                member(self.data.tail(context_length), "timestamp", ["target"], 96, context_length=context_length)
            self.assertEqual([call.kwargs["context_length"] for call in select.call_args_list], [512, 1024])
            self.assertEqual([call.kwargs["revision"] for call in load.call_args_list], ["512-96", "1024-96"])

    def test_long_single_window_keeps_native_context_and_cutoff(self):
        with (
            patch("tsfm_public.toolkit.forecasters.TimeSeriesPreprocessor"),
            patch("tsfm_public.toolkit.forecasters.TimeSeriesForecastingPipeline") as pipeline,
        ):
            self.member(self.data, "timestamp", ["target"], 96, context_length=4096)
            self.assertEqual(pipeline.call_args.kwargs["context_length"], 2048)
            passed_data = pipeline.return_value.call_args.args[0]
            pd.testing.assert_frame_equal(passed_data, self.data.tail(2048))
            self.assertEqual(passed_data.timestamp.iloc[-1], self.data.timestamp.iloc[-1])
            self.assertEqual(len(self.data), 4096)

    def test_shorter_context_is_preserved(self):
        data = self.data.tail(512)
        with (
            patch("tsfm_public.toolkit.forecasters.TimeSeriesPreprocessor"),
            patch("tsfm_public.toolkit.forecasters.TimeSeriesForecastingPipeline") as pipeline,
        ):
            self.member(data, "timestamp", ["target"], 96, context_length=512)
            self.assertEqual(pipeline.call_args.kwargs["context_length"], 512)
            pd.testing.assert_frame_equal(pipeline.return_value.call_args.args[0], data)

    def test_grouped_and_rolling_inputs_are_not_silently_trimmed(self):
        for data, ids in [(self.data.assign(series="a"), ["series"]), (self.data.iloc[:-1], [])]:
            with self.subTest(id_columns=ids, rows=len(data)):
                with self.assertRaisesRegex(ValueError, "single series with one context window"):
                    self.member(
                        data,
                        "timestamp",
                        ["target"],
                        96,
                        context_length=4096,
                        id_columns=ids,
                    )


if __name__ == "__main__":
    unittest.main()
