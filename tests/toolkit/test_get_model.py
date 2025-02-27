# Copyright contributors to the TSFM project
#

"""Tests get_model"""

import numpy as np
import pytest

from tsfm_public.toolkit.get_model import get_model


def test_granite_r1_models():
    # Check basic cases
    mp = "ibm-granite/granite-timeseries-ttm-r1"
    for cl in [512, 1024]:
        for fl in [96]:
            model_key = get_model(model_path=mp, context_length=cl, prediction_length=fl, return_model_key=True)
            assert int(model_key.split("-")[1]) == fl
            assert int(model_key.split("-")[0]) == cl

    mp = "ibm-granite/granite-timeseries-ttm-r1"
    cl = 1024
    fl = 56
    model_key = get_model(
        model_path=mp,
        context_length=cl,
        prediction_length=fl,
        head_dropout=0.3,
        return_model_key=True,
    )

    mp = "ibm/TTM"
    cl = 512
    fl = 90
    model_key = get_model(model_path=mp, context_length=cl, prediction_length=fl, return_model_key=True)
    assert int(model_key.split("-")[0]) == cl
    assert int(model_key.split("-")[1]) == 96

    mp = "ibm/TTM"
    cl = 300
    fl = 45
    model_key = get_model(
        model_path=mp,
        context_length=cl,
        prediction_length=fl,
        prefer_longer_context=False,
        return_model_key=True,
        force_return="zeropad",
    )
    assert int(model_key.split("-")[0]) == 512
    assert int(model_key.split("-")[1]) == 96


def test_research_r2_models():
    mp = "ibm-research/ttm-research-r2"
    for cl in [512, 1024, 1536]:
        for fl in [96, 192, 336, 720]:
            model_key = get_model(model_path=mp, context_length=cl, prediction_length=fl, return_model_key=True)
            assert int(model_key.split("-")[1]) == fl
            assert int(model_key.split("-")[0]) == cl


def test_granite_r2_basic_models():
    mp = "ibm-granite/granite-timeseries-ttm-r2"
    # 12 high freq models
    for cl in [512, 1024, 1536]:
        for fl in [96, 192, 336, 720]:
            model_key = get_model(model_path=mp, context_length=cl, prediction_length=fl, return_model_key=True)
            assert int(model_key.split("-")[1]) == fl
            assert int(model_key.split("-")[0]) == cl

    # ---- Low freq models [52, 90, 180, 360, 512] ----
    # 52 - l2 loss
    cl = 52
    fl = 16
    model_key = get_model(model_path=mp, context_length=cl, prediction_length=fl, return_model_key=True)
    assert int(model_key.split("-")[1]) == fl
    assert int(model_key.split("-")[0]) == cl

    # 52 - l1 loss
    cl = 52
    fl = 16
    model_key = get_model(
        model_path=mp, context_length=cl, prediction_length=fl, prefer_l1_loss=True, return_model_key=True
    )
    assert int(model_key.split("-")[1]) == fl
    assert int(model_key.split("-")[0]) == cl
    assert "l1" in model_key

    # 90 - l2 loss
    cl = 90
    fl = 30
    model_key = get_model(model_path=mp, context_length=cl, prediction_length=fl, return_model_key=True)
    assert int(model_key.split("-")[1]) == fl
    assert int(model_key.split("-")[0]) == cl

    # 90 - l1 loss
    cl = 90
    fl = 30
    model_key = get_model(
        model_path=mp, context_length=cl, prediction_length=fl, prefer_l1_loss=True, return_model_key=True
    )
    assert int(model_key.split("-")[1]) == fl
    assert int(model_key.split("-")[0]) == cl
    assert "l1" in model_key

    # 180 - l1 loss
    cl = 180
    fl = 60
    model_key = get_model(
        model_path=mp, context_length=cl, prediction_length=fl, prefer_l1_loss=True, return_model_key=True
    )
    assert int(model_key.split("-")[1]) == fl
    assert int(model_key.split("-")[0]) == cl
    assert "l1" in model_key

    # 360 - l1 loss
    cl = 360
    fl = 60
    model_key = get_model(
        model_path=mp, context_length=cl, prediction_length=fl, prefer_l1_loss=True, return_model_key=True
    )
    assert int(model_key.split("-")[1]) == fl
    assert int(model_key.split("-")[0]) == cl
    assert "l1" in model_key

    # 512/48 - l2 loss
    cl = 512
    fl = 48
    model_key = get_model(
        model_path=mp, context_length=cl, prediction_length=fl, prefer_l1_loss=False, return_model_key=True
    )
    assert int(model_key.split("-")[1]) == fl
    assert int(model_key.split("-")[0]) == cl

    # 512/48 - l1 loss
    cl = 512
    fl = 48
    model_key = get_model(
        model_path=mp, context_length=cl, prediction_length=fl, prefer_l1_loss=True, return_model_key=True
    )
    assert int(model_key.split("-")[1]) == fl
    assert int(model_key.split("-")[0]) == cl
    assert "l1" in model_key

    # 512/96 - l2 loss
    cl = 512
    fl = 96
    model_key = get_model(
        model_path=mp,
        context_length=cl,
        prediction_length=fl,
        prefer_l1_loss=False,
        freq_prefix_tuning=True,
        freq="d",
        return_model_key=True,
    )
    assert int(model_key.split("-")[1]) == fl
    assert int(model_key.split("-")[0]) == cl

    # 512/96 - l1 loss
    cl = 512
    fl = 96
    model_key = get_model(
        model_path=mp,
        context_length=cl,
        prediction_length=fl,
        prefer_l1_loss=True,
        freq_prefix_tuning=True,
        freq="10min",
        return_model_key=True,
    )
    assert int(model_key.split("-")[1]) == fl
    assert int(model_key.split("-")[0]) == cl
    assert "l1" in model_key


def test_granite_r2_other_models():
    mp = "ibm-granite/granite-timeseries-ttm-r2"
    cl = 512
    fl = 10
    model_key = get_model(
        model_path=mp,
        context_length=cl,
        prediction_length=fl,
        dropout=0.4,
        decoder_num_layers=1,
        return_model_key=True,
    )
    assert int(model_key.split("-")[1]) == 48
    assert int(model_key.split("-")[0]) == cl

    mp = "ibm-granite/granite-timeseries-ttm-r2"
    cl = 1536
    fl = 200
    model_key = get_model(
        model_path=mp,
        context_length=cl,
        prediction_length=fl,
        decoder_adaptive_patching_levels=2,
        return_model_key=True,
    )
    assert int(model_key.split("-")[1]) == 336
    assert int(model_key.split("-")[0]) == cl


def test_prefer_longer_context():
    mp = "ibm-granite/granite-timeseries-ttm-r2"
    # Case 1
    cl = 100
    fl = 20
    model_key = get_model(
        model_path=mp, context_length=cl, prediction_length=fl, prefer_longer_context=True, return_model_key=True
    )
    assert int(model_key.split("-")[1]) == 30
    assert int(model_key.split("-")[0]) == 90

    # Case 2
    cl = 100
    fl = 20
    model_key = get_model(
        model_path=mp, context_length=cl, prediction_length=fl, prefer_longer_context=False, return_model_key=True
    )
    assert int(model_key.split("-")[1]) == 30
    assert int(model_key.split("-")[0]) == 90

    # Case 3
    cl = 100
    fl = 10
    model_key = get_model(
        model_path=mp, context_length=cl, prediction_length=fl, prefer_longer_context=False, return_model_key=True
    )
    assert int(model_key.split("-")[1]) == 16
    assert int(model_key.split("-")[0]) == 52

    # Case 4
    cl = 100
    fl = 10
    model_key = get_model(
        model_path=mp, context_length=cl, prediction_length=fl, prefer_longer_context=True, return_model_key=True
    )
    assert int(model_key.split("-")[1]) == 30
    assert int(model_key.split("-")[0]) == 90


def test_freq_tuning():
    mp = "ibm-granite/granite-timeseries-ttm-r2"
    # # ----------- test freq_prefix_tuning -----------
    # Case 1
    cl = 512
    fl = 96
    model_key = get_model(
        model_path=mp,
        context_length=cl,
        prediction_length=fl,
        prefer_l1_loss=False,
        freq_prefix_tuning=False,
        freq="h",
        return_model_key=True,
    )
    assert int(model_key.split("-")[1]) == fl
    assert int(model_key.split("-")[0]) == cl

    # Case 2
    cl = 512
    fl = 96
    model_key = get_model(
        model_path=mp,
        context_length=cl,
        prediction_length=fl,
        prefer_l1_loss=True,
        freq_prefix_tuning=False,
        freq="H",
        return_model_key=True,
    )
    assert int(model_key.split("-")[1]) == fl
    assert int(model_key.split("-")[0]) == cl
    assert "l1" in model_key

    # Case 3
    cl = 512
    fl = 96
    model_key = get_model(
        model_path=mp,
        context_length=cl,
        prediction_length=fl,
        prefer_l1_loss=False,
        freq_prefix_tuning=True,
        freq="3min",
        return_model_key=True,
        force_return="random_init_large",
    )
    assert "large" in model_key

    # Case 4
    cl = 512
    fl = 96
    model_key = get_model(
        model_path=mp,
        context_length=cl,
        prediction_length=fl,
        prefer_l1_loss=True,
        freq_prefix_tuning=True,
        freq="H",
        return_model_key=True,
    )
    assert int(model_key.split("-")[1]) == fl
    assert int(model_key.split("-")[0]) == cl
    assert "l1" in model_key


def test_random_models():
    mp = "ibm-granite/granite-timeseries-ttm-r2"

    for cl in np.linspace(2, 20_000, 10):
        for fl in [cl // 8, cl // 4, cl // 2]:
            fl = int(fl)
            cl = int(cl)
            if fl < 1:
                continue
            if cl < 52:
                force_return = "zeropad"
            if fl > 720:
                force_return = "rolling"
            model_key = get_model(
                model_path=mp,
                context_length=cl,
                prediction_length=fl,
                return_model_key=True,
                force_return=force_return,
            )
            assert model_key is not None


# Define test cases with multiple combinations
# (cl, fl, res, ft, l1, longer, force_return)
test_cases_granite_r2 = [
    # Basic exact matches
    (512, 96, None, False, False, True, None, "512-96-r2"),
    (1024, 192, None, False, False, True, None, "1024-192-r2"),
    (1536, 720, None, False, False, True, None, "1536-720-r2"),
    # Forecast length filtering
    (512, 720, None, False, False, True, "rolling", "512-720-r2"),
    (1024, 1000, None, False, False, True, "rolling", "1024-720-r2"),
    (1024, 1000, None, False, False, True, "rolling", "1024-720-r2"),
    (512, 192, None, False, False, True, "rolling", "512-192-r2"),
    (180, 60, None, False, False, True, "rolling", "180-60-ft-l1-r2.1"),
    (200, 50, None, False, False, True, "rolling", "180-60-ft-l1-r2.1"),
    # Context length filtering
    (2000, 336, None, False, False, True, "zeropad", "1536-336-r2"),
    (1200, 96, None, False, False, True, "zeropad", "1024-96-r2"),
    (600, 336, None, False, False, True, "zeropad", "512-336-r2"),
    # Context length ordering (Prefer Longer Context)
    (1200, 96, None, False, False, True, None, "1024-96-r2"),
    (1200, 96, None, False, False, False, None, "512-96-r2"),
    (80, 20, "10min", False, False, True, "rolling", "52-16-ft-r2.1"),
    (80, 200, "10min", False, False, True, "random_init_small", "TTM(small)"),
    (80, 40, "10min", False, False, True, "rolling", "52-16-ft-r2.1"),
    (200, 80, "10min", False, False, False, "rolling", "180-60-ft-l1-r2.1"),
    (200, 80, "10min", False, False, True, "rolling", "180-60-ft-l1-r2.1"),
    (400, 80, "10min", False, False, False, "rolling", "180-60-ft-l1-r2.1"),
    (400, 80, "10min", False, False, True, "rolling", "360-60-ft-l1-r2.1"),
    # FT and L1 Preferences (Only Apply for CL ≤ 512)
    (512, 96, None, False, True, True, None, "512-96-ft-l1-r2.1"),
    (1024, 96, None, False, True, True, None, "1024-96-r2"),  # L1 ignored
    (512, 96, None, True, False, True, None, "512-96-ft-r2.1"),
    (1024, 96, None, True, False, True, None, "1024-96-r2"),  # FT ignored
    (512, 96, None, True, True, True, None, "512-96-ft-l1-r2.1"),
    # Resolution-based filtering
    (512, 96, "d", False, False, True, None, "512-96-ft-r2.1"),
    (512, 96, "d", False, True, True, None, "512-96-ft-l1-r2.1"),
    (300, 20, "W", True, True, True, "zeropad", "180-60-ft-l1-r2.1"),
    (60, 12, "W", True, True, True, "zeropad", "52-16-ft-l1-r2.1"),
    (60, 12, "W", True, False, True, "zerop", "52-16-ft-r2.1"),
    (36, 12, "W", True, True, False, "zeropad", "52-16-ft-l1-r2.1"),
    (36, 12, "W", False, False, False, "zeropad", "52-16-ft-r2.1"),
    (36, 12, "M", False, False, True, "random_init_small", "TTM(small)"),
    (512, 96, "oov", False, False, True, None, "512-96-r2"),
    (512, 96, "5min", False, False, True, None, "512-96-r2"),
    (512, 96, "random", False, False, True, None, "512-96-r2"),  # Invalid freq, but ft=False
    (512, 96, "random", True, False, True, "random_init_medium", "TTM(medium)"),  # Invalid freq, but ft=True
    (20, 6, "W", True, True, False, "zeropad", "52-16-ft-l1-r2.1"),
    (200, 24, "W", True, False, True, "zeropad", "180-60-ft-l1-r2.1"),
    (200, 24, "W", True, False, False, "zeropad", "90-30-ft-r2.1"),
    (200, 24, "H", True, False, False, "zeropad", "90-30-ft-r2.1"),
    (20, 6, "A", True, True, True, "random_init_small", "TTM(small)"),
    # Edge Cases (No Match Scenarios + some match scenarios)
    (10, 5, None, False, False, True, "random_init_small", "TTM(small)"),
    (10, 5, None, True, False, True, "random_init_small", "TTM(small)"),
    (10, 5, None, False, False, True, "zeropad", "1536-96-r2"),
    (10, 5, None, False, False, False, "zeropad", "52-16-ft-r2.1"),
    (10, 5, None, True, False, False, "zeropad", "52-16-ft-r2.1"),
    (100, 50, None, True, False, True, "random_init_small", "TTM(small)"),
    (100, 50, None, True, False, True, "random_init_medium", "TTM(medium)"),
    (100, 50, None, True, True, True, "rolling", "90-30-ft-l1-r2.1"),
    (600, 50, None, False, False, True, None, "512-96-r2"),
    (600, 40, None, False, False, True, None, "512-48-ft-r2.1"),
    (1536, 1000, None, False, False, True, "random_init_large", "TTM(large)"),
    (1536, 336, "W", False, False, True, "random_init_small", "TTM(small)"),
    (1536, 96, "d", True, True, True, None, "512-96-ft-l1-r2.1"),
    (13, 3, "d", True, True, True, "random_init_small", "TTM(small)"),
    # Complex Cases with Multiple Constraints
    (512, 96, "oov", True, True, True, None, "512-96-ft-l1-r2.1"),
    (512, 192, "d", True, False, False, "rolling", "512-96-ft-r2.1"),
    (512, 192, "d", True, False, False, "random_init_small", "TTM(small)"),
    (1024, 96, None, True, True, False, None, "512-96-ft-l1-r2.1"),
    (1536, 336, "d", False, False, True, "rolling", "512-96-ft-r2.1"),
    (1536, 336, "d", False, True, True, "rolling", "512-96-ft-l1-r2.1"),
    # Sorting correctness
    (1200, 500, None, False, False, True, None, "1024-720-r2"),
    (1200, 500, None, False, False, False, None, "512-720-r2"),
    (1200, 500, None, False, True, False, "rolling", "512-96-ft-l1-r2.1"),
    (1200, 500, None, False, True, False, "random_init_small", "TTM(small)"),
]


@pytest.mark.parametrize("cl, fl, res, ft, l1, longer, force_return, expected", test_cases_granite_r2)
def test_all_cases_granite_r2(cl, fl, res, ft, l1, longer, force_return, expected):
    model_key = get_model(
        "ibm-granite/granite-timeseries-ttm-r2",
        context_length=cl,
        prediction_length=fl,
        freq=res,
        freq_prefix_tuning=ft,
        prefer_l1_loss=l1,
        prefer_longer_context=longer,
        force_return=force_return,
        return_model_key=True,
    )
    assert model_key == expected


# (cl, fl, res, ft, l1, longer, force_return)
test_cases_granite_r1 = [
    # Basic exact matches
    (512, 96, None, False, False, True, None, "512-96-r1"),
    (1024, 96, None, False, False, True, None, "1024-96-r1"),
    # filter prediction length
    (512, 50, None, False, False, False, None, "512-96-r1"),
    # invalid context length
    (50, 50, None, False, False, False, "zeropad", "512-96-r1"),
    (50, 50, None, False, False, False, "random_init_small", "TTM(small)"),
    # prefer longer context
    (1200, 50, None, False, False, True, None, "1024-96-r1"),
    (1200, 50, None, False, False, False, None, "512-96-r1"),
]


@pytest.mark.parametrize("cl, fl, res, ft, l1, longer, force_return, expected", test_cases_granite_r1)
def test_all_cases_granite_r1(cl, fl, res, ft, l1, longer, force_return, expected):
    model_key = get_model(
        "ibm-granite/granite-timeseries-ttm-r1",
        context_length=cl,
        prediction_length=fl,
        freq=res,
        freq_prefix_tuning=ft,
        prefer_l1_loss=l1,
        prefer_longer_context=longer,
        force_return=force_return,
        return_model_key=True,
    )
    assert model_key == expected


test_cases_research_r2 = [
    # Basic exact matches
    (512, 96, None, False, False, True, None, "512-96-ft-r2"),
    (1024, 96, None, False, False, True, None, "1024-96-ft-r2"),
    # filter prediction length
    (512, 50, None, False, False, False, None, "512-96-ft-r2"),
    # invalid context length
    (50, 50, None, False, False, False, "zeropad", "512-96-ft-r2"),
    (50, 50, None, False, False, False, "random_init_small", "TTM(small)"),
    # prefer longer context
    (1200, 50, None, False, False, True, None, "1024-96-ft-r2"),
    (1200, 50, None, False, False, False, None, "512-96-ft-r2"),
]


@pytest.mark.parametrize("cl, fl, res, ft, l1, longer, force_return, expected", test_cases_research_r2)
def test_all_cases_research_r2(cl, fl, res, ft, l1, longer, force_return, expected):
    model_key = get_model(
        "ibm-research/ttm-research-r2",
        context_length=cl,
        prediction_length=fl,
        freq=res,
        freq_prefix_tuning=ft,
        prefer_l1_loss=l1,
        prefer_longer_context=longer,
        force_return=force_return,
        return_model_key=True,
    )
    assert model_key == expected
