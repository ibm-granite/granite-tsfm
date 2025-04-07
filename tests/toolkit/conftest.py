# Copyright contributors to the TSFM project
#

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from transformers import PatchTSTForPrediction

from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
from tsfm_public.toolkit.util import select_by_index

from ..util import nreps


@pytest.fixture(scope="module")
def ts_data():
    df = pd.DataFrame(
        {
            "id": nreps(["A", "B", "C"], 50),
            "id2": nreps(["XX", "YY", "ZZ"], 50),
            "timestamp": [datetime(2021, 1, 1) + timedelta(days=i) for i in range(50)] * 3,
            "value1": range(150),
            "value2": np.arange(150) ** 2 / 3 + 10,
        }
    )
    return df


@pytest.fixture(scope="module")
def sample_data():
    df = pd.DataFrame(
        {
            "val": range(10),
            "val2": [x + 100 for x in range(10)],
            "cat": ["A", "B"] * 5,
            "cat2": ["CC", "DD", "EE", "FF", "GG"] * 2,
        }
    )
    return df


@pytest.fixture(scope="module")
def ts_data_runs():
    df = pd.DataFrame(
        {
            "run_id": nreps(["1", "2", "3", "4"], 50),
            "asset_id": nreps(["foo", "bar", "foo", "bar"], 50),
            "timestamp": [datetime(2021, 1, 1) + timedelta(days=i) for i in range(50)] * 4,
            "value1": range(200),
        }
    )
    return df


@pytest.fixture(scope="package")
def etth_data_base():
    timestamp_column = "date"
    dataset_path = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv"
    data = pd.read_csv(
        dataset_path,
        parse_dates=[timestamp_column],
    )
    return data


@pytest.fixture(scope="package")
def etth_data(etth_data_base):
    timestamp_column = "date"
    id_columns = []
    target_columns = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
    prediction_length = 96

    data = etth_data_base.copy()
    train_end_index = 12 * 30 * 24

    context_length = 512  # model.config.context_length

    test_end_index = 12 * 30 * 24 + 8 * 30 * 24
    test_start_index = test_end_index - context_length - 4

    train_data = select_by_index(
        data,
        id_columns=id_columns,
        start_index=0,
        end_index=train_end_index,
    )
    test_data = select_by_index(
        data,
        id_columns=id_columns,
        start_index=test_start_index,
        end_index=test_end_index,
    )

    params = {
        "timestamp_column": timestamp_column,
        "id_columns": id_columns,
        "target_columns": target_columns,
        "prediction_length": prediction_length,
        "context_length": context_length,
    }

    return train_data, test_data, params


@pytest.fixture(scope="package")
def patchtst_base_model():
    model_path = "ibm/test-patchtst"
    model = PatchTSTForPrediction.from_pretrained(model_path)

    return model


@pytest.fixture(scope="module")
def ttm_base_model():
    model_path = "ibm/test-ttm-v1"

    return TinyTimeMixerForPrediction.from_pretrained(model_path)


@pytest.fixture(scope="module")
def ttm_model():
    model_path = "ibm/test-ttm-v1"

    def ttm_model_func(**kwargs):
        return TinyTimeMixerForPrediction.from_pretrained(model_path, **kwargs)

    return ttm_model_func
