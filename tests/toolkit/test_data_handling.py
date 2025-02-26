# Copyright contributors to the TSFM project
#

"""Tests data handling functions"""

import numpy as np

from tsfm_public import load_dataset


def test_load_dataset():
    dset_train, dset_valid, dset_test = load_dataset(
        dataset_name="etth1",
        context_length=512,
        forecast_length=96,
        fewshot_fraction=1.0,
        dataset_path="https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
    )

    np.testing.assert_allclose([len(x) for x in [dset_train, dset_valid, dset_test]], [8033, 2785, 2785])
