# Copyright contributors to the TSFM project
#

"""Tests get_model"""

import os
import sys

import numpy as np
import pandas as pd
import pytest
from gluonts.dataset.common import ListDataset
from gluonts.dataset.split import split


sys.path.append(os.path.realpath("../../../"))
from extras.gluonts.data.gluonts_data_wrapper import (
    StandardScalingGluonTSDataset,
    TorchDatasetFromGluonTSTestDataset,
    TorchDatasetFromGluonTSTrainingDataset,
)


@pytest.fixture(scope="module")
def gluonts_data():
    # Step 1: Define the multivariate time series data
    num_time_series = 3  # Number of time series
    length = 50  # Length of each time series
    num_variables = 2  # Number of variables (dimensions) per time series

    # Create random multivariate time series data
    time_series_data = [
        {
            "item_id": f"ts{i+1}",
            "start": pd.Timestamp("2024-01-01"),  # Start time for each series
            "target": np.random.rand(num_variables, length),  # 2D array: (num_variables, length)
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


def test_gluonts_standard_scaling(gluonts_data):
    dataset = gluonts_data

    # Split the dataset into train and test
    prediction_length = 10
    train_dataset, test_template = split(dataset, offset=-prediction_length)

    # Test shapes
    for entry in train_dataset:
        assert entry["target"].shape == (2, 40)

    test_dataset = test_template.generate_instances(
        prediction_length=prediction_length,
    )
    test_dataset_input = test_dataset.input
    test_dataset_label = test_dataset.label
    # Test shapes
    for entry in test_dataset_input:
        assert entry["target"].shape == (2, 40)
    for entry in test_dataset_label:
        assert entry["target"].shape == (2, 10)

    # Test scaler
    scaler = StandardScalingGluonTSDataset()
    scaler.fit(train_dataset)
    train_dataset_scaled = scaler.transform(train_dataset)
    test_dataset_scaled = scaler.transform(test_dataset_input)

    # Test scaling
    for entry in train_dataset_scaled:
        np.testing.assert_almost_equal(entry["target"].mean(axis=1), np.array([0.0, 0.0]), decimal=4)
        np.testing.assert_almost_equal(entry["target"].std(axis=1), np.array([1.0, 1.0]), decimal=4)

    for entry in test_dataset_scaled:
        np.testing.assert_almost_equal(entry["target"].mean(axis=1), np.array([0.0, 0.0]), decimal=4)
        np.testing.assert_almost_equal(entry["target"].std(axis=1), np.array([1.0, 1.0]), decimal=4)

    # inverse
    test_label_scaled = scaler.transform(test_dataset_label)
    Y = []
    for entry in test_label_scaled:
        Y.append(entry["target"].T)
    Y = np.array(Y)
    Y_inv = scaler.inverse_transform(Y)

    Y_org = []
    for entry in test_dataset_label:
        Y_org.append(entry["target"].T)
    Y_org = np.array(Y_org)

    np.testing.assert_almost_equal(Y_inv.mean(), Y_org.mean(), decimal=4)


def test_pytorch_data_wrappers(gluonts_data):
    dataset = gluonts_data

    # Split the dataset into train and test
    prediction_length = 10
    train_dataset, test_template = split(dataset, offset=-prediction_length)
    test_dataset = test_template.generate_instances(
        prediction_length=prediction_length,
    )
    test_dataset_input = test_dataset.input
    test_dataset_label = test_dataset.label

    torch_train_dset = TorchDatasetFromGluonTSTrainingDataset(train_dataset, seq_len=20, forecast_len=5)
    assert torch_train_dset[1]["past_values"].shape == (20, 2)
    assert torch_train_dset[1]["future_values"].shape == (5, 2)

    torch_train_dset = TorchDatasetFromGluonTSTrainingDataset(train_dataset, seq_len=35, forecast_len=5)
    assert torch_train_dset[1]["past_values"].shape == (35, 2)
    assert torch_train_dset[1]["future_values"].shape == (5, 2)
    assert len(torch_train_dset) == 3

    torch_test_dset = TorchDatasetFromGluonTSTestDataset(
        gluon_test_input=test_dataset_input, gluon_test_label=test_dataset_label, seq_len=20, forecast_len=5
    )
    assert torch_test_dset[0]["past_values"].shape == (20, 2)
    assert torch_test_dset[0]["future_values"].shape == (5, 2)
