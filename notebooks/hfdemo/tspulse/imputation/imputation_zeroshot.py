import csv
import os
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import mask_generate, mse

from tsfm_public import TimeSeriesPreprocessor, get_datasets
from tsfm_public.models.tspulse import TSPulseForReconstruction


warnings.filterwarnings("ignore")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
CONTEXT_LEN = 512
FORECAST_LEN = 0


def main(DATASET, mask_type, mask_ratio):
    if DATASET in ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]:
        batch_size = 64
    else:
        batch_size = 4

    # Dataset
    if DATASET in ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]:
        dataset_path = f"https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/{DATASET}.csv"
    else:
        dataset_path = f"datasets/{DATASET}/{DATASET}.csv"

    timestamp_column = "date"
    id_columns = []

    if DATASET in ["ETTh1", "ETTh2"]:
        split_config = {
            "train": [0, 8640],
            "valid": [8640, 11520],
            "test": [
                11520,
                14400,
            ],
        }
    elif DATASET in ["ETTm1", "ETTm2"]:
        split_config = {
            "train": [0, 34560],
            "valid": [34560, 46080],
            "test": [
                46080,
                57600,
            ],
        }
    else:
        split_config = {
            "train": 0.7,
            "test": 0.2,
        }

    data = pd.read_csv(
        dataset_path,
        parse_dates=[timestamp_column],
    )

    target_columns = data.columns.to_list()[1:]  # all the columns from the data except 'date'

    column_specifiers = {
        "timestamp_column": timestamp_column,
        "id_columns": id_columns,
        "target_columns": target_columns,
        "control_columns": [],
    }

    tsp = TimeSeriesPreprocessor(
        **column_specifiers,
        context_length=CONTEXT_LEN,
        prediction_length=FORECAST_LEN,
        scaling=True,
        encode_categorical=False,
        scaler_type="standard",
    )

    _, _, dset_test = get_datasets(tsp, data, split_config)

    def collate_only_past_values(batch):
        return torch.stack([item["past_values"] for item in batch])

    test_dataloader = DataLoader(dset_test, batch_size=batch_size, shuffle=False, collate_fn=collate_only_past_values)

    model = TSPulseForReconstruction.from_pretrained(
        "ibm-granite/granite-timeseries-tspulse-r1",
        revision="tspulse-hybrid-dualhead-512-p8-r1",
        num_input_channels=tsp.num_input_channels,
        mask_type="user",
    ).to(device)

    seed = 42
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    trues, preds, masks = [], [], []
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            batch_x = batch.to(device)  # b l c

            mask = mask_generate(g, batch_x, 8, mask_ratio, mask_type)

            output = model(past_values=batch_x, past_observed_mask=~mask)

            reconstructed_output = output.reconstruction_outputs

            trues.append(batch_x.detach().cpu().numpy())
            preds.append(reconstructed_output.detach().cpu().numpy())
            masks.append(mask.detach().cpu().numpy())

        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        masks = np.concatenate(masks)

        MSE = mse(y=trues[masks == 1], y_hat=preds[masks == 1], reduction="mean")
        print(f"Dataset = {DATASET}  : Mask Type = {mask_type}  : Mask Ratio = {mask_ratio}")
        print(f"Mean Squarred Error (MSE)={MSE:.3f}")

        output_file = "tspulse_zeroshot_imputation_results.csv"

        if not os.path.exists(output_file):
            with open(output_file, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Dataset", "Mask Type", "Mask Ratio", "MSE"])

        with open(output_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([DATASET, mask_type, mask_ratio, f"{MSE:.3f}"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        type=str,
        required=True,
        choices=["ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather", "electricity"],
        help="List of UEA dataset names",
    )
    parser.add_argument(
        "--mask_type",
        nargs="+",
        type=str,
        required=True,
        choices=["block", "hybrid"],
        help="Masking strategy to evaluate. Options available : 'block' and 'hybrid'",
    )
    parser.add_argument(
        "--mask_ratios",
        nargs="+",
        type=float,
        required=True,
        choices=[0.125, 0.25, 0.375, 0.5],
        help="Masking ratios to evaluate. Options available : 0.125, 0.25, 0.375, 0.5",
    )
    args = parser.parse_args()

    for m_t in args.mask_type:
        for DATASET in args.datasets:
            for m_r in args.mask_ratios:
                main(DATASET, m_t, m_r)
