# Copyright contributors to the TSFM project
#

"""Tests get_model"""

import tempfile

import numpy as np
import pytest

from tsfm_public.toolkit.get_model import get_model


def prefer_longer_context():
    mp = "ibm-granite/granite-timeseries-ttm-r2"
    # Case 1
    cl = 100
    fl = 20
    model = get_model(model_path=mp, context_length=cl, prediction_length=fl, prefer_longer_context=True)
    assert model.config.prediction_length == 30
    assert model.config.prediction_filter_length == 20
    assert model.config.context_length == 90
    # Case 2
    cl = 100
    fl = 20
    model = get_model(model_path=mp, context_length=cl, prediction_length=fl, prefer_longer_context=False)
    assert model.config.prediction_length == 30
    assert model.config.prediction_filter_length == 20
    assert model.config.context_length == 90
    # Case 3
    cl = 100
    fl = 10
    model = get_model(model_path=mp, context_length=cl, prediction_length=fl, prefer_longer_context=False)
    assert model.config.prediction_length == 16
    assert model.config.prediction_filter_length == 10
    assert model.config.context_length == 52
    # Case 4
    cl = 100
    fl = 10
    model = get_model(model_path=mp, context_length=cl, prediction_length=fl, prefer_longer_context=True)
    assert model.config.prediction_length == 30
    assert model.config.prediction_filter_length == 10
    assert model.config.context_length == 90


def freq_tuning():
    mp = "ibm-granite/granite-timeseries-ttm-r2"
    # # ----------- test freq_prefix_tuning -----------
    # Case 1
    cl = 512
    fl = 96
    model = get_model(
        model_path=mp,
        context_length=cl,
        prediction_length=fl,
        prefer_l1_loss=False,
        freq_prefix_tuning=False,
        resolution="h",
    )
    assert model.config.prediction_length == fl
    assert model.config.context_length == cl
    assert not model.config.resolution_prefix_tuning
    assert model.config.loss == "mse"

    # Case 2
    cl = 512
    fl = 96
    model = get_model(
        model_path=mp,
        context_length=cl,
        prediction_length=fl,
        prefer_l1_loss=True,
        freq_prefix_tuning=False,
        resolution="H",
    )
    assert model.config.prediction_length == fl
    assert model.config.context_length == cl
    assert model.config.resolution_prefix_tuning  # won't match since not available
    assert model.config.loss == "mae"  # won't match since not available
    # Case 3
    cl = 512
    fl = 96
    model = get_model(
        model_path=mp,
        context_length=cl,
        prediction_length=fl,
        prefer_l1_loss=False,
        freq_prefix_tuning=True,
        resolution="3min",
    )
    assert model.config.prediction_length == fl
    assert model.config.context_length == cl
    assert not model.config.resolution_prefix_tuning
    assert model.config.loss == "mse"
    # Case 4
    cl = 512
    fl = 96
    model = get_model(
        model_path=mp,
        context_length=cl,
        prediction_length=fl,
        prefer_l1_loss=True,
        freq_prefix_tuning=True,
        resolution="H",
    )
    assert model.config.prediction_length == fl
    assert model.config.context_length == cl
    assert model.config.resolution_prefix_tuning
    assert model.config.loss == "mae"


def granite_r2_basic_models():
    mp = "ibm-granite/granite-timeseries-ttm-r2"
    # 12 high resolution models
    for cl in [512, 1024, 1536]:
        for fl in [96, 192, 336, 720]:
            model = get_model(model_path=mp, context_length=cl, prediction_length=fl)
            assert model.config.prediction_length == fl
            assert model.config.context_length == cl
            assert not model.config.resolution_prefix_tuning
            assert model.config.loss == "mse"

    # ---- Low resolution models [52, 90, 180, 360, 512] ----
    # 52 - l2 loss
    cl = 52
    fl = 16
    model = get_model(model_path=mp, context_length=cl, prediction_length=fl)
    assert model.config.prediction_length == fl
    assert model.config.context_length == cl
    assert model.config.resolution_prefix_tuning
    assert model.config.loss == "mse"
    # 52 - l1 loss
    cl = 52
    fl = 16
    model = get_model(model_path=mp, context_length=cl, prediction_length=fl, prefer_l1_loss=True)
    assert model.config.prediction_length == fl
    assert model.config.context_length == cl
    assert model.config.resolution_prefix_tuning
    assert model.config.loss == "mae"

    # 90 - l2 loss
    cl = 90
    fl = 30
    model = get_model(model_path=mp, context_length=cl, prediction_length=fl)
    assert model.config.prediction_length == fl
    assert model.config.context_length == cl
    assert model.config.resolution_prefix_tuning
    assert model.config.loss == "mse"
    # 90 - l1 loss
    cl = 90
    fl = 30
    model = get_model(model_path=mp, context_length=cl, prediction_length=fl, prefer_l1_loss=True)
    assert model.config.prediction_length == fl
    assert model.config.context_length == cl
    assert model.config.resolution_prefix_tuning
    assert model.config.loss == "mae"

    # 180 - l1 loss
    cl = 180
    fl = 60
    model = get_model(model_path=mp, context_length=cl, prediction_length=fl, prefer_l1_loss=True)
    assert model.config.prediction_length == fl
    assert model.config.context_length == cl
    assert model.config.resolution_prefix_tuning
    assert model.config.loss == "mae"

    # 360 - l1 loss
    cl = 360
    fl = 60
    model = get_model(model_path=mp, context_length=cl, prediction_length=fl, prefer_l1_loss=True)
    assert model.config.prediction_length == fl
    assert model.config.context_length == cl
    assert model.config.resolution_prefix_tuning
    assert model.config.loss == "mae"

    # 512/48 - l2 loss
    cl = 512
    fl = 48
    model = get_model(model_path=mp, context_length=cl, prediction_length=fl, prefer_l1_loss=False)
    assert model.config.prediction_length == fl
    assert model.config.context_length == cl
    assert model.config.resolution_prefix_tuning
    assert model.config.loss == "mse"

    # 512/48 - l1 loss
    cl = 512
    fl = 48
    model = get_model(model_path=mp, context_length=cl, prediction_length=fl, prefer_l1_loss=True)
    assert model.config.prediction_length == fl
    assert model.config.context_length == cl
    assert model.config.resolution_prefix_tuning
    assert model.config.loss == "mae"

    # 512/96 - l2 loss
    cl = 512
    fl = 96
    model = get_model(
        model_path=mp,
        context_length=cl,
        prediction_length=fl,
        prefer_l1_loss=False,
        freq_prefix_tuning=True,
        resolution="d",
    )
    assert model.config.prediction_length == fl
    assert model.config.context_length == cl
    assert model.config.resolution_prefix_tuning
    assert model.config.loss == "mse"

    # 512/96 - l1 loss
    cl = 512
    fl = 96
    model = get_model(
        model_path=mp,
        context_length=cl,
        prediction_length=fl,
        prefer_l1_loss=True,
        freq_prefix_tuning=True,
        resolution="10min",
    )
    assert model.config.prediction_length == fl
    assert model.config.context_length == cl
    assert model.config.resolution_prefix_tuning
    assert model.config.loss == "mae"


def granite_r2_other_models():
    mp = "ibm-granite/granite-timeseries-ttm-r2"
    for cl in range(1, 2000, 500):
        for fl in range(1, 900, 90):
            model = get_model(model_path=mp, context_length=cl, prediction_length=fl)
            if model.config.prediction_filter_length is not None:
                assert model.config.prediction_filter_length == fl

    mp = "ibm-granite/granite-timeseries-ttm-r2"
    cl = 512
    fl = 10
    model = get_model(model_path=mp, context_length=cl, prediction_length=fl, dropout=0.4, decoder_num_layers=1)
    assert model.config.prediction_length == 48
    assert model.config.context_length == cl
    assert model.config.dropout == 0.4

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


def test_granite_r2_models():
    granite_r2_basic_models()
    granite_r2_other_models()
    freq_tuning()
    prefer_longer_context()


def test_granite_r1_models():
    # Check basic cases
    mp = "ibm-granite/granite-timeseries-ttm-r1"
    for cl in [512, 1024]:
        for fl in [96]:
            model = get_model(model_path=mp, context_length=cl, prediction_length=fl)
            assert model.config.prediction_length == fl
            assert model.config.context_length == cl

    mp = "ibm-granite/granite-timeseries-ttm-r1"
    for cl in range(512, 2000, 500):
        for fl in range(1, 720, 90):
            model = get_model(model_path=mp, context_length=cl, prediction_length=fl)
            if model.config.prediction_filter_length is not None:
                assert model.config.prediction_filter_length == fl

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

    mp = "ibm/TTM"
    cl = 300
    fl = 45
    model = get_model(model_path=mp, context_length=cl, prediction_length=fl, prefer_longer_context=False)
    assert model.config.prediction_length == 96
    assert model.config.context_length == 512
    assert model.config.d_model == 192


def test_research_r2_models():
    mp = "ibm-research/ttm-research-r2"
    for cl in [512, 1024, 1536]:
        for fl in [96, 192, 336, 720]:
            model = get_model(model_path=mp, context_length=cl, prediction_length=fl)
            assert model.config.prediction_length == fl
            assert model.config.context_length == cl

    for cl in range(1, 2000, 500):
        for fl in range(1, 900, 90):
            model = get_model(model_path=mp, context_length=cl, prediction_length=fl)
            if model.config.prediction_filter_length is not None:
                assert model.config.prediction_filter_length == fl


def test_random_models():
    mp = "ibm-granite/granite-timeseries-ttm-r2"

    for cl in np.linspace(2, 20_000, 10):
        for fl in [cl // 8, cl // 4, cl // 2]:
            fl = int(fl)
            cl = int(cl)
            if fl < 1:
                continue
            model = get_model(model_path=mp, context_length=cl, prediction_length=fl)
            assert model is not None


# Define test cases with multiple combinations
# (cl, fl, res, ft, l1, longer, force_return)
test_cases_granite_r2 = [
    # Basic exact matches
    (512, 96, None, False, False, True, True, "512-96-r2"),
    (1024, 192, None, False, False, True, True, "1024-192-r2"),
    (1536, 720, None, False, False, True, True, "1536-720-r2"),
    # Forecast length filtering
    (512, 720, None, False, False, True, True, "512-720-r2"),
    (1024, 1000, None, False, False, True, False, "TTM(small)"),
    (1024, 1000, None, False, False, True, True, "1024-720-r2"),
    (512, 192, None, False, False, True, True, "512-192-r2"),
    (180, 60, None, False, False, True, True, "180-60-ft-l1-r2.1"),
    (200, 50, None, False, False, True, True, "180-60-ft-l1-r2.1"),
    # Context length filtering
    (2000, 336, None, False, False, True, True, "1536-336-r2"),
    (1200, 96, None, False, False, True, True, "1024-96-r2"),
    (600, 336, None, False, False, True, True, "512-336-r2"),
    # Context length ordering (Prefer Longer Context)
    (1200, 96, None, False, False, True, True, "1024-96-r2"),
    (1200, 96, None, False, False, False, True, "512-96-r2"),
    (80, 20, "10min", False, False, True, True, "52-16-ft-r2.1"),
    (80, 20, "10min", False, False, True, False, "TTM(small)"),
    (80, 40, "10min", False, False, True, True, "52-16-ft-r2.1"),
    (200, 80, "10min", False, False, False, True, "180-60-ft-l1-r2.1"),
    (200, 80, "10min", False, False, True, True, "180-60-ft-l1-r2.1"),
    (400, 80, "10min", False, False, False, True, "180-60-ft-l1-r2.1"),
    (400, 80, "10min", False, False, True, True, "360-60-ft-l1-r2.1"),
    # FT and L1 Preferences (Only Apply for CL ≤ 512)
    (512, 96, None, False, True, True, True, "512-96-ft-l1-r2.1"),
    (1024, 96, None, False, True, True, True, "1024-96-r2"),  # L1 ignored
    (512, 96, None, True, False, True, True, "512-96-ft-r2.1"),
    (1024, 96, None, True, False, True, True, "1024-96-r2"),  # FT ignored
    (512, 96, None, True, True, True, True, "512-96-ft-l1-r2.1"),
    # Resolution-based filtering
    (512, 96, "d", False, False, True, True, "512-96-ft-r2.1"),
    (512, 96, "d", False, True, True, True, "512-96-ft-l1-r2.1"),
    (300, 20, "W", True, True, True, True, "180-60-ft-l1-r2.1"),
    (60, 12, "W", True, True, True, True, "52-16-ft-l1-r2.1"),
    (60, 12, "W", True, False, True, True, "52-16-ft-r2.1"),
    (36, 12, "W", True, True, False, True, "52-16-ft-l1-r2.1"),
    (36, 12, "W", False, False, False, True, "52-16-ft-r2.1"),  # since force_return=True
    (36, 12, "M", False, False, True, False, "TTM(small)"),  # since force_return=False
    (512, 96, "oov", False, False, True, True, "512-96-r2"),
    (512, 96, "5min", False, False, True, True, "512-96-r2"),
    (512, 96, "random", False, False, True, True, "512-96-r2"),  # Invalid resolution, but ft=False
    (512, 96, "random", True, False, True, True, "TTM(small)"),  # Invalid resolution, but ft=True
    (20, 6, "W", True, True, False, True, "52-16-ft-l1-r2.1"),
    (200, 24, "W", True, False, True, True, "180-60-ft-l1-r2.1"),
    (200, 24, "W", True, False, False, True, "90-30-ft-r2.1"),
    (200, 24, "H", True, False, False, True, "90-30-ft-r2.1"),
    (20, 6, "A", True, True, True, True, "TTM(small)"),
    # Edge Cases (No Match Scenarios + some match scenarios)
    (10, 5, None, False, False, True, False, "TTM(small)"),  # force_return=False
    (10, 5, None, True, False, True, False, "TTM(small)"),  # force_return=False, but ft=True
    (10, 5, None, False, False, True, True, "1536-96-r2"),
    (10, 5, None, False, False, False, True, "52-16-ft-r2.1"),
    (10, 5, None, True, False, False, True, "52-16-ft-r2.1"),
    (100, 50, None, True, False, True, False, "TTM(small)"),
    (100, 50, None, True, False, True, True, "90-30-ft-r2.1"),
    (100, 50, None, True, True, True, True, "90-30-ft-l1-r2.1"),
    (600, 50, None, False, False, True, True, "512-96-r2"),
    (600, 40, None, False, False, True, True, "512-48-ft-r2.1"),  # ft ignored
    (1536, 1000, None, False, False, True, False, "TTM(small)"),
    (1536, 336, "W", False, False, True, False, "TTM(small)"),
    (1536, 96, "d", True, True, True, False, "512-96-ft-l1-r2.1"),
    (13, 3, "d", True, True, True, False, "TTM(small)"),
    # Complex Cases with Multiple Constraints
    (512, 96, "oov", True, True, True, True, "512-96-ft-l1-r2.1"),
    (512, 192, "d", True, False, False, True, "512-96-ft-r2.1"),
    (512, 192, "d", True, False, False, False, "TTM(small)"),
    (1024, 96, None, True, True, False, True, "512-96-ft-l1-r2.1"),
    (1536, 336, "d", False, False, True, True, "512-96-ft-r2.1"),
    (1536, 336, "d", False, True, True, True, "512-96-ft-l1-r2.1"),
    # Sorting correctness
    (1200, 500, None, False, False, True, True, "1024-720-r2"),
    (1200, 500, None, False, False, False, True, "512-720-r2"),
    (1200, 500, None, False, True, False, True, "512-96-ft-l1-r2.1"),
    (1200, 500, None, False, True, False, False, "TTM(small)"),
]


@pytest.mark.parametrize("cl, fl, res, ft, l1, longer, force_return, expected", test_cases_granite_r2)
def test_all_cases_granite_r2(cl, fl, res, ft, l1, longer, force_return, expected):
    model, model_key = get_model(
        "ibm-granite/granite-timeseries-ttm-r2",
        context_length=cl,
        prediction_length=fl,
        resolution=res,
        freq_prefix_tuning=ft,
        prefer_l1_loss=l1,
        prefer_longer_context=longer,
        force_return=force_return,
        return_model_key=True,
    )
    assert model_key == expected


test_cases_granite_r1 = [
    # Basic exact matches
    (512, 96, None, False, False, True, True, "512-96-r1"),
    (1024, 96, None, False, False, True, True, "1024-96-r1"),
    # filter prediction length
    (512, 50, None, False, False, False, True, "512-96-r1"),
    # invalid context length
    (50, 50, None, False, False, False, True, "512-96-r1"),
    (50, 50, None, False, False, False, False, "TTM(small)"),
    # prefer longer context
    (1200, 50, None, False, False, True, True, "1024-96-r1"),
    (1200, 50, None, False, False, False, True, "512-96-r1"),
]


@pytest.mark.parametrize("cl, fl, res, ft, l1, longer, force_return, expected", test_cases_granite_r1)
def test_all_cases_granite_r1(cl, fl, res, ft, l1, longer, force_return, expected):
    model, model_key = get_model(
        "ibm-granite/granite-timeseries-ttm-r1",
        context_length=cl,
        prediction_length=fl,
        resolution=res,
        freq_prefix_tuning=ft,
        prefer_l1_loss=l1,
        prefer_longer_context=longer,
        force_return=force_return,
        return_model_key=True,
    )
    assert model_key == expected


test_cases_research_r2 = [
    # Basic exact matches
    (512, 96, None, False, False, True, True, "512-96-ft-r2"),
    (1024, 96, None, False, False, True, True, "1024-96-ft-r2"),
    # filter prediction length
    (512, 50, None, False, False, False, True, "512-96-ft-r2"),
    # invalid context length
    (50, 50, None, False, False, False, True, "512-96-ft-r2"),
    (50, 50, None, False, False, False, False, "TTM(small)"),
    # prefer longer context
    (1200, 50, None, False, False, True, True, "1024-96-ft-r2"),
    (1200, 50, None, False, False, False, True, "512-96-ft-r2"),
]


@pytest.mark.parametrize("cl, fl, res, ft, l1, longer, force_return, expected", test_cases_research_r2)
def test_all_cases_research_r2(cl, fl, res, ft, l1, longer, force_return, expected):
    model, model_key = get_model(
        "ibm-research/ttm-research-r2",
        context_length=cl,
        prediction_length=fl,
        resolution=res,
        freq_prefix_tuning=ft,
        prefer_l1_loss=l1,
        prefer_longer_context=longer,
        force_return=force_return,
        return_model_key=True,
    )
    assert model_key == expected
