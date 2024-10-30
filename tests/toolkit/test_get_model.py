# Copyright contributors to the TSFM project
#

"""Tests GetTTM"""

import tempfile

from tsfm_public.toolkit.get_model import GetTTM


def test_get_ttm():
    mp = "ibm-granite/granite-timeseries-ttm-r2"
    cl = 512
    fl = 10
    model = GetTTM.from_pretrained(model_path=mp, context_length=cl, forecast_length=fl, dropout=0.4)
    assert model.config.prediction_length == 96
    assert model.config.context_length == cl
    assert model.config.d_model == 192

    tmp_dir = tempfile.mkdtemp()
    model.save_pretrained(tmp_dir)
    model = GetTTM.from_pretrained(tmp_dir)
    assert model.config.d_model == 192

    mp = "ibm-granite/granite-timeseries-ttm-r2"
    cl = 1536
    fl = 200
    model = GetTTM.from_pretrained(model_path=mp, context_length=cl, forecast_length=fl)
    assert model.config.prediction_length == 336
    assert model.config.context_length == cl
    assert model.config.d_model == 384

    mp = "ibm-granite/granite-timeseries-ttm-r1"
    cl = 1024
    fl = 56
    model = GetTTM.from_pretrained(model_path=mp, context_length=cl, forecast_length=fl, dropout=0.3)
    assert model.config.prediction_length == 96
    assert model.config.context_length == cl
    assert model.config.d_model == 192

    mp = "ibm/TTM"
    cl = 512
    fl = 90
    model = GetTTM.from_pretrained(model_path=mp, context_length=cl, forecast_length=fl)
    assert model.config.prediction_length == 96
    assert model.config.context_length == cl
    assert model.config.d_model == 192
