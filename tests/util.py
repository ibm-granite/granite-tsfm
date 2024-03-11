# Copyright contributors to the TSFM project
#

"""Utilities for testing"""

from datetime import datetime, timedelta
from itertools import chain, repeat

import numpy as np
import pandas as pd
import pytest


def nreps(iterable, n):
    "Returns each element in the sequence repeated n times."
    return chain.from_iterable((repeat(i, n) for i in iterable))


@pytest.fixture(scope="module")
def ts_data():
    df = pd.DataFrame(
        {
            "id": nreps(["A", "B", "C"], 50),
            "id2": nreps(["XX", "YY", "ZZ"], 50),
            "timestamp": [datetime(2021, 1, 1) + timedelta(days=i) for i in range(50)]
            * 3,
            "value1": range(150),
            "value2": np.arange(150) / 3 + 10,
        }
    )
    return df
