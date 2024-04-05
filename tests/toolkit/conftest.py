# Copyright contributors to the TSFM project
#

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

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
            "cat2": ["CC", "DD"] * 5,
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
