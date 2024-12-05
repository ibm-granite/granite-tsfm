# Copyright contributors to the TSFM project
#

"""Tests get_model"""

import tempfile

from tsfm_public.toolkit.get_model import get_model


def test_get_model():
    mp = "ibm-granite/granite-timeseries-ttm-r2"
    cl = 512
    fl = 10
    model = get_model(model_path=mp, context_length=cl, prediction_length=fl, dropout=0.4, decoder_num_layers=1)
    assert model.config.prediction_length == 96
    assert model.config.context_length == cl
    assert model.config.d_model == 192

    tmp_dir = tempfile.mkdtemp()
    model.save_pretrained(tmp_dir)
    model = get_model(tmp_dir)
    assert model.config.d_model == 192

    mp = "ibm-granite/granite-timeseries-ttm-r2"
    cl = 1536
    fl = 200
    model = get_model(model_path=mp, context_length=cl, prediction_length=fl, decoder_adaptive_patching_levels=2)
    assert model.config.prediction_length == 336
    assert model.config.context_length == cl
    assert model.config.d_model == 384

    mp = "ibm-granite/granite-timeseries-ttm-r1"
    cl = 1024
    fl = 56
    model = get_model(model_path=mp, context_length=cl, prediction_length=fl, head_dropout=0.3)
    assert model.config.prediction_length == 96
    assert model.config.context_length == cl
    assert model.config.d_model == 192

    mp = "ibm/TTM"
    cl = 512
    fl = 90
    model = get_model(model_path=mp, context_length=cl, prediction_length=fl)
    assert model.config.prediction_length == 96
    assert model.config.context_length == cl
    assert model.config.d_model == 192

    mp = "ibm-granite/granite-timeseries-ttm-r1"
    for cl in [512, 1024]:
        for fl in [96]:
            model = get_model(model_path=mp, context_length=cl, prediction_length=fl)
            assert model.config.prediction_length == fl
            assert model.config.context_length == cl

    mp = "ibm-granite/granite-timeseries-ttm-r2"
    for cl in [512, 1024, 1536]:
        for fl in [96, 192, 336, 720]:
            model = get_model(model_path=mp, context_length=cl, prediction_length=fl)
            assert model.config.prediction_length == fl
            assert model.config.context_length == cl

    mp = "ibm/ttm-research-r2"
    for cl in [512, 1024, 1536]:
        for fl in [96, 192, 336, 720]:
            model = get_model(model_path=mp, context_length=cl, prediction_length=fl)
            assert model.config.prediction_length == fl
            assert model.config.context_length == cl

    mp = "ibm/ttm-research-r2"
    for cl in range(1, 2000, 500):
        for fl in range(1, 900, 90):
            model = get_model(model_path=mp, context_length=cl, prediction_length=fl)
            if model.config.prediction_filter_length is not None:
                assert model.config.prediction_filter_length == fl

    mp = "ibm-granite/granite-timeseries-ttm-r2"
    for cl in range(1, 2000, 500):
        for fl in range(1, 900, 90):
            model = get_model(model_path=mp, context_length=cl, prediction_length=fl)
            if model.config.prediction_filter_length is not None:
                assert model.config.prediction_filter_length == fl

    mp = "ibm-granite/granite-timeseries-ttm-r1"
    for cl in range(512, 2000, 500):
        for fl in range(1, 720, 90):
            model = get_model(model_path=mp, context_length=cl, prediction_length=fl)
            if model.config.prediction_filter_length is not None:
                assert model.config.prediction_filter_length == fl
