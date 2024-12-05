"""Utilities used by test cases in this folder"""

import os

import pandas as pd
import wget


def fetch_etth():
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


if __name__ == "__main__":
    fetch_etth()
