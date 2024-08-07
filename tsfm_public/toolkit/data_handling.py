"""Utilities for handling datasets"""

import glob
import logging
import os
from importlib import resources
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from .time_series_preprocessor import TimeSeriesPreprocessor, get_datasets


LOGGER = logging.getLogger(__file__)


def load_dataset(
    dataset_name: str,
    context_length,
    forecast_length,
    fewshot_fraction=1.0,
    fewshot_location="first",
    dataset_root_path: str = "datasets/",
    dataset_path: Optional[str] = None,
):
    LOGGER.info(f"Dataset name: {dataset_name}, context length: {context_length}, prediction length {forecast_length}")

    config_path = resources.files("tsfm_public.resources.data_config")
    configs = glob.glob(os.path.join(config_path, "*.yaml"))

    names_to_config = {Path(p).stem: p for p in configs}
    config_path = names_to_config.get(dataset_name, None)

    if config_path is None:
        raise ValueError(
            f"Currently the `load_dataset()` function supports the following datasets: {names_to_config.keys()}\n \
                         For other datasets, please provide the proper configs to the TimeSeriesPreprocessor (TSP) module."
        )

    config = yaml.safe_load(open(config_path, "r"))

    tsp = TimeSeriesPreprocessor(
        id_columns=config["id_columns"],
        timestamp_column=config["timestamp_column"],
        target_columns=config["target_columns"],
        observable_columns=config["observable_columns"],
        control_columns=config["control_columns"],
        conditional_columns=config["conditional_columns"],
        static_categorical_columns=config["static_categorical_columns"],
        scaling=config["scale"]["scaling"],
        scaler_type=config["scale"]["scaler_type"],
        encode_categorical=config["encode_categorical"],
        freq=config["freq"],
        context_length=context_length,
        prediction_length=forecast_length,
    )

    split_config = config["split"]

    # if dataset_path is provided we will ignore the config file
    if dataset_path is None:
        dataset_path = Path(dataset_root_path) / config["data_path"] / config["data_file"]

    data = pd.read_csv(
        dataset_path,
        parse_dates=[config["timestamp_column"]],
    )

    train_dataset, valid_dataset, test_dataset = get_datasets(
        tsp,
        data,
        split_config=split_config,
        fewshot_fraction=fewshot_fraction,
        fewshot_location=fewshot_location,
    )
    LOGGER.info(f"Data lengths: train = {len(train_dataset)}, val = {len(valid_dataset)}, test = {len(test_dataset)}")

    return train_dataset, valid_dataset, test_dataset
