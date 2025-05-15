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


def test_save_deployment_package():
    c = PatchTSTConfig(prediction_length=77)
    m = PatchTSTForPrediction(c)

    with tempfile.TemporaryDirectory() as d:
        save_deployment_package(d, m)
        m_new = PatchTSTForPrediction.from_pretrained(d)
        assert m_new.config.prediction_length == m.config.prediction_length

    c = TimeSeriesTransformerConfig(prediction_length=11)
    m = TimeSeriesTransformerForPrediction(c)

    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(Exception):
            save_deployment_package(d, m)
