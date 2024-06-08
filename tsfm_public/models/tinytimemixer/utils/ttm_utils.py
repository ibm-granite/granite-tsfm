"""Utilities for TTM notebooks"""
# Standard
import argparse
import os

# Third Party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Local
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor


def get_ttm_args():
    parser = argparse.ArgumentParser(description="TTM pretrain arguments.")
    # Adding a positional argument
    parser.add_argument(
        "--forecast_length",
        "-fl",
        type=int,
        required=False,
        default=96,
        help="Forecast length",
    )
    parser.add_argument(
        "--context_length",
        "-cl",
        type=int,
        required=False,
        default=512,
        help="History context length",
    )
    parser.add_argument(
        "--patch_length",
        "-pl",
        type=int,
        required=False,
        default=64,
        help="Patch length",
    )
    parser.add_argument(
        "--adaptive_patching_levels",
        "-apl",
        type=int,
        required=False,
        default=3,
        help="Number of adaptive patching levels of TTM",
    )
    parser.add_argument(
        "--d_model_scale",
        "-dms",
        type=int,
        required=False,
        default=3,
        help="Model hidden dimension",
    )
    parser.add_argument(
        "--decoder_d_model_scale",
        "-ddms",
        type=int,
        required=False,
        default=2,
        help="Decoder hidden dimension",
    )
    parser.add_argument(
        "--num_gpus",
        "-ng",
        type=int,
        required=False,
        default=None,
        help="Number of GPUs",
    )
    parser.add_argument("--random_seed", "-rs", type=int, required=False, default=42, help="Random seed")
    parser.add_argument("--batch_size", "-bs", type=int, required=False, default=3000, help="Batch size")
    parser.add_argument(
        "--num_epochs",
        "-ne",
        type=int,
        required=False,
        default=25,
        help="Number of epochs",
    )

    parser.add_argument(
        "--num_workers",
        "-nw",
        type=int,
        required=False,
        default=8,
        help="Number of dataloader workers",
    )
    parser.add_argument(
        "--learning_rate",
        "-lr",
        type=float,
        required=False,
        default=0.001,
        help="Learning rate",
    )
    parser.add_argument(
        "--dataset",
        "-ds",
        type=str,
        required=False,
        default="etth1",
        help="Dataset",
    )
    parser.add_argument(
        "--data_root_path",
        "-drp",
        type=str,
        required=False,
        default="datasets/",
        help="Dataset",
    )
    parser.add_argument(
        "--save_dir",
        "-sd",
        type=str,
        required=False,
        default="tmp/",
        help="Data path",
    )
    parser.add_argument(
        "--early_stopping",
        "-es",
        type=int,
        required=False,
        default=1,
        help="Whether to use early stopping during finetuning.",
    )
    parser.add_argument(
        "--freeze_backbone",
        "-fb",
        type=int,
        required=False,
        default=1,
        help="Whether to freeze the backbone during few-shot finetuning.",
    )

    # Parsing the arguments
    args = parser.parse_args()
    args.early_stopping = int_to_bool(args.early_stopping)
    args.freeze_backbone = int_to_bool(args.freeze_backbone)
    args.d_model = args.patch_length * args.d_model_scale
    args.decoder_d_model = args.patch_length * args.decoder_d_model_scale

    # Calculate number of gpus
    if args.num_gpus is None:
        args.num_gpus = torch.cuda.device_count()
        print("Automatically calculated number of GPUs =", args.num_gpus)

    # Create save directory
    args.save_dir = os.path.join(
        args.save_dir,
        f"TTM_cl-{args.context_length}_fl-{args.forecast_length}_pl-{args.patch_length}_apl-{args.adaptive_patching_levels}_ne-{args.num_epochs}_es-{args.early_stopping}",
    )
    os.makedirs(args.save_dir, exist_ok=True)

    return args


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def int_to_bool(value):
    if value == 0:
        return False
    elif value == 1:
        return True
    else:
        raise argparse.ArgumentTypeError("Boolean value expected (0 or 1)")


# Utitlity: plot
def plot_preds(trainer, dset, plot_dir, num_plots=10, plot_prefix="valid", channel=-1):
    device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu")
    random_indices = np.random.choice(len(dset), size=num_plots, replace=False)
    random_samples = torch.stack([dset[i]["past_values"] for i in random_indices])
    trainer.model = trainer.model.to(device)
    output = trainer.model(random_samples.to(device=device))
    y_hat = output.prediction_outputs[:, :, channel].detach().cpu().numpy()
    pred_len = y_hat.shape[1]

    # Set a more beautiful style
    plt.style.use("seaborn-v0_8-whitegrid")

    # Account for 1 num_plots
    if num_plots == 1:
        for i, ri in enumerate(random_indices):
            batch = dset[ri]
            y = batch["future_values"][:pred_len, channel].squeeze().cpu().numpy()
            x = batch["past_values"][: 2 * pred_len, channel].squeeze().cpu().numpy()
            y = np.concatenate((x, y), axis=0)

            # Plot predicted values with a dashed line
            y_hat_plot = np.concatenate((x, y_hat[i, ...]), axis=0)
            plt.figure(figsize=(10, 2))

            plt.plot(y_hat_plot, label="Predicted", linestyle="--", color="orange", linewidth=2)

            # Plot true values with a solid line
            plt.plot(y, label="True", linestyle="-", color="blue", linewidth=2)

            # Plot horizon border
            plt.axvline(x=2 * pred_len, color="r", linestyle="-")

            plt.title(f"Example {random_indices[i]}")
            plt.legend()

    # Adjust overall layout
        plt.tight_layout()
        return

    # Adjust figure size and subplot spacing
    _, axs = plt.subplots(num_plots, 1, figsize=(10, 2 * num_plots))
    for i, ri in enumerate(random_indices):
        batch = dset[ri]

        y = batch["future_values"][:pred_len, channel].squeeze().cpu().numpy()
        x = batch["past_values"][: 2 * pred_len, channel].squeeze().cpu().numpy()
        y = np.concatenate((x, y), axis=0)

        # Plot predicted values with a dashed line
        y_hat_plot = np.concatenate((x, y_hat[i, ...]), axis=0)
        axs[i].plot(y_hat_plot, label="Predicted", linestyle="--", color="orange", linewidth=2)

        # Plot true values with a solid line
        axs[i].plot(y, label="True", linestyle="-", color="blue", linewidth=2)

        # Plot horizon border
        axs[i].axvline(x=2 * pred_len, color="r", linestyle="-")

        axs[i].set_title(f"Example {random_indices[i]}")
        axs[i].legend()

    # Adjust overall layout
    plt.tight_layout()

    # Save the plot
    plot_filename = f"synthetic_{plot_prefix}_ch_{str(channel)}.pdf"
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, plot_filename))


# Get data loaders using TSP
def get_data(
    dataset_name: str,
    context_length,
    forecast_length,
    fewshot_fraction=1.0,
    data_root_path: str = "datasets/",
):
    print(dataset_name, context_length, forecast_length)

    config_map = {
        "etth1": {
            "dataset_path": os.path.join(data_root_path, "ETT-small/ETTh1.csv"),
            "timestamp_column": "date",
            "id_columns": [],
            "target_columns": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"],
            "split_config": {
                "train": [0, 12 * 30 * 24],
                "valid": [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24],
                "test": [12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24],
            },
        },
        "etth2": {
            "dataset_path": os.path.join(data_root_path, "ETT-small/ETTh2.csv"),
            "timestamp_column": "date",
            "id_columns": [],
            "target_columns": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"],
            "split_config": {
                "train": [0, 12 * 30 * 24],
                "valid": [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24],
                "test": [12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24],
            },
        },
        "ettm1": {
            "dataset_path": os.path.join(data_root_path, "ETT-small/ETTm1.csv"),
            "timestamp_column": "date",
            "id_columns": [],
            "target_columns": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"],
            "split_config": {
                "train": [0, 12 * 30 * 24 * 4],
                "valid": [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4],
                "test": [
                    12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,
                    12 * 30 * 24 * 4 + 8 * 30 * 24 * 4,
                ],
            },
        },
        "ettm2": {
            "dataset_path": os.path.join(data_root_path, "ETT-small/ETTm2.csv"),
            "timestamp_column": "date",
            "id_columns": [],
            "target_columns": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"],
            "split_config": {
                "train": [0, 12 * 30 * 24 * 4],
                "valid": [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4],
                "test": [
                    12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,
                    12 * 30 * 24 * 4 + 8 * 30 * 24 * 4,
                ],
            },
        },
        "weather": {
            "dataset_path": os.path.join(data_root_path, "weather/weather.csv"),
            "timestamp_column": "date",
            "id_columns": [],
            "target_columns": [],
            "split_config": {
                "train": 0.7,
                "test": 0.2,
            },
        },
        "electricity": {
            "dataset_path": os.path.join(data_root_path, "electricity/electricity.csv"),
            "timestamp_column": "date",
            "id_columns": [],
            "target_columns": [],
            "split_config": {
                "train": 0.7,
                "test": 0.2,
            },
        },
        "traffic": {
            "dataset_path": os.path.join(data_root_path, "traffic/traffic.csv"),
            "timestamp_column": "date",
            "id_columns": [],
            "target_columns": [],
            "split_config": {
                "train": 0.7,
                "test": 0.2,
            },
        },
    }
    if dataset_name not in config_map.keys():
        raise ValueError(
            f"Currently `get_data()` function supports the following datasets: {config_map.keys()}\n \
                         For other datasets, please provide the proper configs to the TimeSeriesPreprocessor (TSP) module."
        )

    dataset_path = config_map[dataset_name]["dataset_path"]
    timestamp_column = config_map[dataset_name]["timestamp_column"]
    id_columns = config_map[dataset_name]["id_columns"]
    target_columns = config_map[dataset_name]["target_columns"]
    split_config = config_map[dataset_name]["split_config"]
    dataset_path = config_map[dataset_name]["dataset_path"]

    if target_columns == []:
        df_tmp_ = pd.read_csv(dataset_path)
        target_columns = list(df_tmp_.columns)
        target_columns.remove(timestamp_column)

    data = pd.read_csv(
        dataset_path,
        parse_dates=[timestamp_column],
    )

    column_specifiers = {
        "timestamp_column": timestamp_column,
        "id_columns": id_columns,
        "target_columns": target_columns,
        "control_columns": [],
    }

    tsp = TimeSeriesPreprocessor(
        **column_specifiers,
        context_length=context_length,
        prediction_length=forecast_length,
        scaling=True,
        encode_categorical=False,
        scaler_type="standard",
    )

    train_dataset, valid_dataset, test_dataset = tsp.get_datasets(
        data, split_config, fewshot_fraction=fewshot_fraction, fewshot_location="first"
    )
    print(f"Data lengths: train = {len(train_dataset)}, val = {len(valid_dataset)}, test = {len(test_dataset)}")

    return train_dataset, valid_dataset, test_dataset
