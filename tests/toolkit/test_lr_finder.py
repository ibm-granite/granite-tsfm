# Copyright contributors to the TSFM project
#

"""Tests learning rate finder utility"""

import tempfile
from pathlib import Path

import numpy as np
from transformers import set_seed

from tsfm_public.toolkit.dataset import ForecastDFDataset
from tsfm_public.toolkit.lr_finder import LRFinder, optimal_lr_finder
from tsfm_public.toolkit.util import select_by_index


def test_lr_finder(ttm_base_model, etth_data_base):
    set_seed(42)
    model = ttm_base_model

    context_length = 512
    prediction_length = 96
    timestamp_column = "date"
    id_columns = []
    target_columns = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]

    data = etth_data_base.copy()

    train_end_index = 12 * 30 * 24 + 8 * 30 * 24
    train_start_index = train_end_index - context_length - prediction_length

    train_data = select_by_index(
        data,
        id_columns=id_columns,
        start_index=train_start_index,
        end_index=train_end_index,
    )

    train_dataset = ForecastDFDataset(
        train_data,
        timestamp_column=timestamp_column,
        target_columns=target_columns,
        prediction_length=prediction_length,
        context_length=context_length,
    )

    learning_rate, finetune_forecast_model = optimal_lr_finder(
        model, train_dataset, device="cpu", batch_size=32, num_iter=10
    )

    np.testing.assert_allclose(learning_rate, 0.00035938136638046257)

    learning_rate, finetune_forecast_model = optimal_lr_finder(
        model, train_dataset, device="mps", batch_size=32, num_iter=10
    )


def test_lr_finder_save_load(ttm_base_model):
    set_seed(42)
    model = ttm_base_model

    lr_finder = LRFinder(
        model,
        opt=None,
        device="cpu",
        start_lr=0.01,
        end_lr=0.001,
        num_iter=2,
        step_mode="exp",
        suggestion="valley",
        enable_prefix_tuning=False,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        fname = lr_finder.save("test_model", tmp_dir)
        lr_finder.load(fname)

        assert isinstance(lr_finder.model, type(model))
