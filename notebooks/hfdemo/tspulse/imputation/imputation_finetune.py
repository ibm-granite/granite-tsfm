import csv
import math
import os
import tempfile
import warnings

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed
from utils import mask_generate, mse

from tsfm_public import TimeSeriesPreprocessor, get_datasets
from tsfm_public.models.tspulse import TSPulseForReconstruction
from tsfm_public.toolkit.lr_finder import optimal_lr_finder


warnings.filterwarnings("ignore")

device = "cuda"
CONTEXT_LEN = 512
FORECAST_LEN = 0


def main(DATASET, mask_type, mask_ratio):
    seed = 42
    set_seed(seed)
    # Dataset
    if DATASET in ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]:
        dataset_path = f"https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/{DATASET}.csv"
    else:
        dataset_path = f"datasets/{DATASET}/{DATASET}.csv"

    timestamp_column = "date"
    id_columns = []  # mention the ids that uniquely identify a time-series.

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

    train_dataset, valid_dataset, test_dataset = get_datasets(tsp, data, split_config)

    model_dict = {
        "mask_ratio": mask_ratio,
        "mask_type": mask_type,
        "prediction_length": 0,
        "fft_time_add_forecasting_pt_loss": False,
        "enable_fft_prob_loss": False,
        "fft_time_consistent_masking": True,
        "fft_original_signal_loss_weight": 0,
        "loss_apply_mode": "mask",
        "fft_weight": 0,
        "num_full_patches_for_hybrid_mask": int((mask_ratio / 0.125) * 4),
        "decoder_mode": "mix_channel",
        "channel_consistent_masking": False,
        "dropout": 0,
        "head_dropout": 0,
    }

    model_dict["num_input_channels"] = tsp.num_input_channels

    model = TSPulseForReconstruction.from_pretrained(
        "ibm-granite/granite-timeseries-tspulse-r1", revision="tspulse-hybrid-dualhead-512-p8-r1", **model_dict
    ).to(device)

    OUT_DIR = "tspulse_finetuned_models/"

    model = model.to("cuda").float()

    for param in model.parameters():
        param.requires_grad = True

    temp_dir = tempfile.mkdtemp()

    suggested_lr = None

    train_dict = {
        "overwrite_output_dir": True,
        "learning_rate": 0.0001,
        "num_train_epochs": 100,
        "evaluation_strategy": "epoch",
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "dataloader_num_workers": 1,
        "eval_accumulation_steps": 50,
        "ddp_find_unused_parameters": False,
        "report_to": "tensorboard",
        "save_strategy": "epoch",
        "logging_strategy": "epoch",
        "save_total_limit": 1,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "seed": 42,
    }

    EPOCHS = train_dict["num_train_epochs"]
    BATCH_SIZE = train_dict["per_device_train_batch_size"]
    eval_accumulation_steps = train_dict["eval_accumulation_steps"]
    NUM_WORKERS = 1
    NUM_GPUS = 1

    set_seed(42)
    if suggested_lr is None:
        lr, model = optimal_lr_finder(
            model,
            train_dataset,
            batch_size=BATCH_SIZE,
        )
        suggested_lr = lr

    finetune_args = TrainingArguments(
        output_dir=temp_dir,
        overwrite_output_dir=True,
        learning_rate=suggested_lr,
        num_train_epochs=EPOCHS,
        do_eval=True,
        eval_strategy="epoch",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_accumulation_steps=eval_accumulation_steps,
        dataloader_num_workers=NUM_WORKERS,
        report_to="tensorboard",
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=1,
        logging_dir=os.path.join(OUT_DIR, "output"),  # Make sure to specify a logging directory
        load_best_model_at_end=True,  # Load the best model when training ends
        metric_for_best_model="eval_loss",  # Metric to monitor for early stopping
        greater_is_better=False,  # For loss
    )

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=10,  # Number of epochs with no improvement after which to stop
        early_stopping_threshold=0.0001,  # Minimum improvement required to consider as improvement
    )

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=suggested_lr)
    scheduler = OneCycleLR(
        optimizer,
        suggested_lr,
        epochs=EPOCHS,
        steps_per_epoch=math.ceil(len(train_dataset) / (BATCH_SIZE * NUM_GPUS)),
    )

    finetune_trainer = Trainer(
        model=model,
        args=finetune_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        callbacks=[early_stopping_callback],
        optimizers=(optimizer, scheduler),
    )

    # Fine tune
    finetune_trainer.train()

    # save the finetuned model
    os.makedirs("finetuned_models", exist_ok=True)
    path_to_save_model = f"finetuned_models/finetuned_model_{DATASET}_{mask_ratio}_{mask_type}"
    finetune_trainer.save_model(path_to_save_model)

    if DATASET in ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]:
        batch_size = 64
    else:
        batch_size = 4

    def collate_only_past_values(batch):
        return torch.stack([item["past_values"] for item in batch])

    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_only_past_values
    )

    model_path = path_to_save_model

    # load the finetuned model
    model = TSPulseForReconstruction.from_pretrained(
        model_path, fft_time_add_forecasting_pt_loss=False, num_input_channels=tsp.num_input_channels, mask_type="user"
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

        output_file = "tspulse_finetuned_imputation_results.csv"

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
