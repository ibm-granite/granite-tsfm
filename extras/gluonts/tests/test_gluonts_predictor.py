# Copyright contributors to the TSFM project
#

"""Tests get_model"""

import os
import sys

import numpy as np
import pandas as pd
import pytest
from gluonts.dataset.common import ListDataset


sys.path.append(os.path.realpath("../../"))
from extras.gluonts.models.tinytimemixer import TTMGluonTSPredictor


@pytest.fixture(scope="module")
def gluonts_data_with_nan():
    # Step 1: Define the multivariate time series data
    num_time_series = 3  # Number of time series
    num_variables = 2  # Number of variables (dimensions) per time series

    # Create random multivariate time series data
    time_series_data = [
        {
            "item_id": f"ts{i+1}",
            "start": pd.Timestamp("2024-01-01"),  # Start time for each series
            "target": np.concatenate(
                (
                    np.array([[np.nan, np.nan, np.nan, np.nan], [0, 1, np.nan, 2]]),
                    np.random.rand(num_variables, 600),
                    np.array([[np.nan, np.nan, np.nan, np.nan], [np.nan, 1, np.nan, 2]]),
                    np.random.rand(num_variables, 4),
                ),
                axis=1,
            ),  # 2D array: (num_variables, length)
        }
        for i in range(num_time_series)
    ]

    # Step 2: Create the ListDataset
    freq = "D"  # Daily frequency
    dataset = ListDataset(
        time_series_data,
        freq=freq,
        one_dim_target=False,
    )
    return dataset


def test_ttm_gluonts_predictor(gluonts_data_with_nan):
    dataset = gluonts_data_with_nan
    predictor = TTMGluonTSPredictor(context_length=512, prediction_length=96)
    forecasts = predictor.predict(dataset)
    assert forecasts[0].samples.shape == (1, 96, 2)
