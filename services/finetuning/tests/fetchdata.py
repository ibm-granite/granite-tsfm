"""Utilities used by test cases in this folder"""

import json
import os

import pandas as pd
import wget


def fetch_etth():
    if not os.path.exists("data/ETTH1.csv"):
        dataset_path = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
        os.makedirs("data", exist_ok=True)
        dataset_path = wget.download(dataset_path, "data/ETTh1.csv")
        timestamp_column = "date"
        data = pd.read_csv(
            dataset_path,
            parse_dates=[timestamp_column],
        )
        data.to_feather("data/ETTh1.feather")
        data.to_csv("data/ETTh1.csv.gz", compression="gzip")


def fetch_finetune_json():
    """Doesn't really fetch, generates it from what fetch_etth"""
    payload = {
        "data": "file://./data/ETTh1.csv",
        "model_id": "mytest-tsfm/ttm-r1",
        "schema": {
            "timestamp_column": "date",
            "target_columns": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"],
        },
        "parameters": {
            "tune_prefix": "fine_tuned/",
            "trainer_args": {"num_train_epochs": 1, "per_device_train_batch_size": 256},
            "fewshot_fraction": 0.05,
        },
    }
    json.dump(payload, open("data/ftpayload.json", "w"))


if __name__ == "__main__":
    fetch_etth()
    fetch_finetune_json()
