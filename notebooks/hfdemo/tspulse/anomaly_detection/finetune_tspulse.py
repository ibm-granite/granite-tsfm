# Copyright contributors to the TSFM project
#

import argparse
import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import ConcatDataset, random_split
from tqdm import tqdm
from transformers.trainer import Trainer
from transformers.trainer_callback import EarlyStoppingCallback
from transformers.training_args import TrainingArguments
from utility.datasets import TSPulseReconstructionDataset

from tsfm_public import count_parameters
from tsfm_public.models.tspulse import TSPulseForReconstruction
from tsfm_public.models.tspulse.utils.helpers import PatchMaskingDatasetWrapper


plt.rcParams.update(
    {
        "font.size": 24,  # Set the desired font size
    }
)

# seeding
seed = 2024
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def get_args():
    ## ArgumentParser
    parser = argparse.ArgumentParser(description="Running TSPulse AD Finetuning Script.")

    parser.add_argument(
        "--filename", type=str, metavar="CSV_FILE", required=True, help="filename with training dataset file name!"
    )

    parser.add_argument(
        "--skip_columns",
        "-sc",
        type=str,
        metavar="COLUMN_IDS",
        default=None,
        help="Specify the columns (by position) to drop from the CSV data, comma separated list.",
    )

    parser.add_argument(
        "--target_columns",
        "-tc",
        type=str,
        metavar="COLUMN_IDS",
        default=None,
        help="Specify the target columns (by position) for model finetuning, comma separated list. "
        "This overrides skip_columns.",
    )

    parser.add_argument(
        "--data_direc",
        type=str,
        metavar="DIRECTORY",
        required=True,
        help="Data directory that contains all the data file!",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        metavar="URL/FILE",
        default="ibm-granite/granite-timeseries-tspulse-r1",
        help="base model that will be finetuned",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        metavar="DIRECTORY",
        required=True,
        help="Output directory where the finetuned model will be saved.",
    )

    parser.add_argument(
        "--aggr_win_size",
        "-wsz",
        type=int,
        metavar="INTEGER",
        default=64,
        help="Size of the scoring window, default is 64.",
    )

    parser.add_argument(
        "--tspulse_encoder_mode",
        "-tem",
        type=str,
        metavar="STRING",
        default="common_channel",
        help="Activate encoder channel mixing by selecting the appropriate mode. Default is common_channel.",
    )

    parser.add_argument(
        "--tspulse_decoder_mode",
        "-tdm",
        type=str,
        metavar="STRING",
        default="common_channel",
        help="Activate decoder channel mixing by selecting the appropriate mode. Default is common_channel.",
    )

    parser.add_argument(
        "--num_workers",
        "-nw",
        type=int,
        metavar="INTEGER",
        default=4,
        help="Number of workers for data loader. Default value is 4.",
    )

    parser.add_argument(
        "--freeze_backbone",
        "-fb",
        type=int,
        metavar="INT",
        default=1,
        help="Freeze backbone while finetuning (0 = tune backbone, 1 = freeze backbone). Default is 1.",
    )

    parser.add_argument(
        "--dataset_name",
        "-dn",
        type=str,
        metavar="STRING",
        required=False,
        default=None,
        help="Dataset name, or filter for selecting target files for tuning!",
    )

    parser.add_argument(
        "--device",
        "-d",
        type=str,
        metavar="STRING",
        required=False,
        default=None,
        help="Device to be used for compute. Supported values are (cpu, cuda, mps)",
    )

    parser.add_argument(
        "--enable_fft_prob_loss",
        "-efpl",
        type=int,
        metavar="INT",
        default=1,
        help="Use fft probability loss in the finetuning (1 = enable, 0 = disable). Default is 1.",
    )

    parser.add_argument(
        "--batch_size",
        "-bs",
        type=int,
        metavar="INTEGER",
        default=1024,
        help="Batch size to be used for model finetuning. Default value is 1024.",
    )

    parser.add_argument(
        "--stride",
        "-s",
        type=int,
        metavar="INTEGER",
        default=1,
        help="Stride for train and validation data generation for model finetuning. Default is 1.",
    )

    parser.add_argument(
        "--epochs",
        "-fne",
        type=int,
        metavar="INTEGER",
        default=20,
        help="Number of epochs for model finetuning. Default is 20.",
    )

    args = parser.parse_args()

    if args.skip_columns is not None:
        try:
            columns = [int(f.strip()) for f in str(args.skip_columns).split(",")]
            columns = list(set(columns))
            args.skip_columns = columns
        except ValueError:
            print("Error: invalid column specification")
            args.skip_columns = None

    if args.target_columns is not None:
        try:
            columns = [int(f.strip()) for f in str(args.target_columns).split(",")]
            columns = list(set(columns))
            args.target_columns = columns
        except ValueError:
            print("Error: invalid column specification")
            args.target_columns = None

    args.freeze_backbone = bool(args.freeze_backbone)
    args.enable_fft_prob_loss = bool(args.enable_fft_prob_loss)

    if args.device not in ["cpu", "cuda", "mps"]:
        args.device = None

    if args.tspulse_decoder_mode not in ("common_channel", "mix_channel"):
        args.tspulse_decoder_mode = "common_channel"

    if args.tspulse_encoder_mode not in ("common_channel", "mix_channel"):
        args.tspulse_encoder_mode = "common_channel"

    return args


if __name__ == "__main__":
    args = get_args()

    if not os.path.exists(args.filename):
        raise ValueError("Error: can not access the model list files!")

    df_eval = pd.read_csv(args.filename)

    file_column = "file_name" if "file_name" in df_eval else df_eval.columns[0]
    all_files = df_eval[file_column].values.tolist()

    # Filter all_files based on dataset name
    if args.dataset_name is not None:
        all_files = [f for f in all_files if args.dataset_name in f]
    print("Files for dataset = ", args.dataset_name)
    print(all_files)

    all_train_dsets = []
    dataframes = []
    print("Creating dataset from all data.")
    input_c = None
    for filename in tqdm(all_files):
        df = pd.read_csv(os.path.join(args.data_direc, filename), index_col=None).dropna()
        col_names = df.columns

        if args.target_columns is not None:
            col_names = col_names[args.target_columns]
        elif args.skip_columns is not None:
            skip_columns = args.skip_columns
            skip_column_names = col_names[skip_columns]
            col_names = [c for c in col_names if c not in skip_column_names]

        data = df[col_names].values.astype(float)

        if input_c is None:
            input_c = data.shape[-1]

        if input_c != data.shape[-1]:
            raise ValueError(
                f"Error: input channels not consistent across finetuned datasets [{input_c}!= {data.shape[-1]}]!"
            )

        # Create a dataset with this data only
        dataframes.append(data)

    print(f"Number of channels: {input_c}")
    # Load model
    model = TSPulseForReconstruction.from_pretrained(
        args.model_path,
        num_input_channels=input_c,
        mode=args.tspulse_encoder_mode,
        decoder_mode=args.tspulse_decoder_mode,
        mask_type="user",
        enable_fft_prob_loss=args.enable_fft_prob_loss,
    )

    context_length = model.config.context_length
    forecast_length = model.config.prediction_length
    min_length = context_length if forecast_length is None else context_length + forecast_length

    for data in dataframes:
        dset_train = TSPulseReconstructionDataset(
            data,
            return_dict=True,
            window_size=context_length,
            aggr_window_size=args.aggr_win_size,
            normalize=True,
            stride=args.stride,
        )
        all_train_dsets.append(dset_train)

    train_val_dset = ConcatDataset(all_train_dsets)

    # Define the split ratio
    train_size = int(0.8 * len(train_val_dset))
    val_size = len(train_val_dset) - train_size

    # Split the dataset
    train_dataset, valid_dataset = random_split(train_val_dset, [train_size, val_size])

    # Get some metadata
    input_c = train_dataset[0]["past_values"].shape[-1]

    # Wrap the datasets to perform masked stiched finetuning
    train_dataset = PatchMaskingDatasetWrapper(
        train_dataset,
        window_length=args.aggr_win_size,
        patch_length=model.config.patch_length,
        window_position="last",
    )

    valid_dataset = PatchMaskingDatasetWrapper(
        valid_dataset,
        window_length=args.aggr_win_size,
        patch_length=model.config.patch_length,
        window_position="last",
    )

    # Freeze the backbone
    if args.freeze_backbone:
        print(
            "Number of params before freezing backbone",
            count_parameters(model),
        )

        # Freeze the backbone of the model
        for param in model.backbone.parameters():
            param.requires_grad = False

        # Count params
        print(
            "Number of params after freezing the backbone",
            count_parameters(model),
        )

    suggested_lr = 0.0001
    finetune_num_epochs = args.epochs
    batch_size = args.batch_size
    num_workers = args.num_workers
    num_gpus = max(torch.cuda.device_count(), 1)
    max_train_samples = 250_000

    num_train_samples = len(train_dataset)
    num_valid_sampels = len(valid_dataset)

    print(f"Fine-tune: Train samples = {num_train_samples}, Valid Samples = {num_valid_sampels}")
    save_dir = args.save_dir

    if input_c > 1:
        # Reduce batch size to avoid OOMs (A100-80GB Gpu)
        batch_size_ = int(batch_size // input_c)
        batch_size = max(16, batch_size_)
        print("Setting batch size to be", batch_size)

    print("Batch Size is set to = ", batch_size)

    extra_args = {}
    if args.device == "cpu":
        extra_args.update(use_cpu=True, use_mps_device=False)
    elif args.device == "cuda":
        extra_args.update(use_cpu=False, use_mps_device=False)
        print("CUDA Available: ", torch.cuda.is_available())
        print("cuDNN Version: ", torch.backends.cudnn.version())
    elif args.device == "mps":
        extra_args.update(use_cpu=False, use_mps_device=True)

    finetune_args = TrainingArguments(
        output_dir=f"{save_dir}/outputs",
        overwrite_output_dir=True,
        learning_rate=suggested_lr,
        num_train_epochs=finetune_num_epochs,
        do_eval=True,
        eval_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers=num_workers,
        report_to="tensorboard",
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=1,
        logging_dir=f"{save_dir}/logs",  # Make sure to specify a logging directory
        load_best_model_at_end=True,  # Load the best model when training ends
        metric_for_best_model="eval_loss",  # Metric to monitor for early stopping
        greater_is_better=False,  # For loss
        **extra_args,
    )

    # Create the early stopping callback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=5,  # Number of epochs with no improvement after which to stop
        early_stopping_threshold=0.0001,  # Minimum improvement required to consider as improvement
    )

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=suggested_lr)
    scheduler = OneCycleLR(
        optimizer,
        suggested_lr,
        epochs=finetune_num_epochs,
        steps_per_epoch=math.ceil(len(train_dataset) / (batch_size * num_gpus)),
    )

    finetune_trainer = Trainer(
        model=model,
        args=finetune_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        callbacks=[early_stopping_callback],
        optimizers=(optimizer, scheduler),
    )

    print(f"Device used for training = {finetune_trainer.args.device}")

    # Fine tune
    finetune_trainer.train()

    # save model
    finetune_trainer.save_model(f"{save_dir}/tspulse_finetuned_model")
