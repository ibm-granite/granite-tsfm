# Copyright contributors to the TSFM project
#
"""Test service support utility"""

import tempfile

import pytest
from transformers import (
    PatchTSTConfig,
    PatchTSTForPrediction,
    TimeSeriesTransformerConfig,
    TimeSeriesTransformerForPrediction,
)

from tsfm_public.toolkit.service_util import save_deployment_package
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor


def test_save_deployment_package():
    c = PatchTSTConfig(prediction_length=77)
    m = PatchTSTForPrediction(c)
    tsp = TimeSeriesPreprocessor(random=42)

    with tempfile.TemporaryDirectory() as d:
        save_deployment_package(d, m, ts_processor=tsp)
        m_new = PatchTSTForPrediction.from_pretrained(d)
        assert m_new.config.prediction_length == m.config.prediction_length
        tsp_new = TimeSeriesPreprocessor.from_pretrained(d)
        assert tsp_new.random == tsp.random

    c = TimeSeriesTransformerConfig(prediction_length=11)
    m = TimeSeriesTransformerForPrediction(c)

    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(Exception):
            save_deployment_package(d, m)
