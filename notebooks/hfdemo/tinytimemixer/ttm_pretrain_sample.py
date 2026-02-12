#!/usr/bin/env python
# coding: utf-8

import logging
import math
import os
import random
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed
from tsfm_public import TimeSeriesPreprocessor, get_datasets
from tsfm_public.models.tinytimemixer.utils import get_ttm_args
from tsfm_public.toolkit.get_model import get_model
from tsfm_public.toolkit.lr_finder import optimal_lr_finder
from tsfm_public.toolkit.visualization import plot_predictions

logger = logging.getLogger(__file__)

# TTM pre-training example.
# This scrips provides a toy example to pretrain a Tiny Time Mixer (TTM) model on
# the `etth1` dataset. For pre-training TTM on a much large set of datasets, please
# have a look at our paper: https://arxiv.org/pdf/2401.03955.pdf
# If you want to directly utilize the pre-trained models. Please use them from the
# Hugging Face Hub: https://huggingface.co/ibm-granite/granite-timeseries-ttm-v1
# Have a look at the fine-tune scripts for example usecases of the pre-trained
# TTM models.

# Basic usage:
# python ttm_pretrain_sample.py --data_root_path datasets/
# See the get_ttm_args() function to know more about other TTM arguments


def get_base_model(args):
    # Pre-train a `TTM` forecasting model
    config = TinyTimeMixerConfig(
        context_length=args.context_length,
        prediction_length=args.forecast_length,
        patch_length=args.patch_length,
        num_input_channels=1,
        patch_stride=args.patch_length,
        d_model=args.d_model,
        num_layers=args.num_layers,  # increase the number of layers if we want more complex models
        mode="common_channel",
        expansion_factor=2,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        scaling="std",
        gated_attn=True,
        adaptive_patching_levels=args.adaptive_patching_levels,
        # decoder params
        decoder_num_layers=args.decoder_num_layers,  # increase the number of layers if we want more complex models
        decoder_adaptive_patching_levels=0,
        decoder_mode="common_channel",
        decoder_raw_residual=False,
        use_decoder=True,
        decoder_d_model=args.decoder_d_model,
        multi_scale=args.multi_scale,
        register_tokens=args.register_tokens,
        fft_length=args.fft_length,
        multi_quantile_head=args.multi_quantile_head,
        point_extra_weight=args.point_extra_weight,
        use_fft_embedding=True,
    )

    model = TinyTimeMixerForPrediction(config)
    return model


def pretrain(args, model, dset_train, dset_val):
    # Find optimal learning rate
    # Use with caution: Set it manually if the suggested learning rate is not suitable

    learning_rate, model = optimal_lr_finder(
        model,
        dset_train,
        batch_size=args.batch_size,
    )
    print("OPTIMAL SUGGESTED LEARNING RATE =", learning_rate)

    # learning_rate = args.learning_rate

    trainer_args = TrainingArguments(
        output_dir=os.path.join(args.save_dir, "checkpoint"),
        overwrite_output_dir=True,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        seed=args.random_seed,
        eval_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        dataloader_num_workers=args.num_workers,
        ddp_find_unused_parameters=False,
        report_to="tensorboard",
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=1,
        logging_dir=os.path.join(
            args.save_dir, "logs"
        ),  # Make sure to specify a logging directory
        load_best_model_at_end=True,  # Load the best model when training ends
        metric_for_best_model="eval_loss",  # Metric to monitor for early stopping
        greater_is_better=False,  # For loss
    )

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = OneCycleLR(
        optimizer,
        learning_rate,
        epochs=args.num_epochs,
        steps_per_epoch=math.ceil(len(dset_train) / args.batch_size),
        # steps_per_epoch=math.ceil(len(dset_train) / (args.batch_size * args.num_gpus)),
    )

    # Create the early stopping callback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=10,  # Number of epochs with no improvement after which to stop
        early_stopping_threshold=0.0,  # Minimum improvement required to consider as improvement
    )

    # Set trainer
    if args.early_stopping:
        trainer = Trainer(
            model=model,
            args=trainer_args,
            train_dataset=dset_train,
            eval_dataset=dset_val,
            optimizers=(optimizer, scheduler),
            callbacks=[early_stopping_callback],
        )
    else:
        trainer = Trainer(
            model=model,
            args=trainer_args,
            train_dataset=dset_train,
            eval_dataset=dset_val,
            optimizers=(optimizer, scheduler),
        )

    # Train
    trainer.train()

    # Save the pretrained model

    model_save_path = os.path.join(args.save_dir, "ttm_pretrained")
    trainer.save_model(model_save_path)
    return model_save_path


def print_model_summary(model, verbose=False):
    """
    Prints full model structure and parameter counts.

    Args:
        model: torch.nn.Module
        verbose (bool): if True, prints every parameter name and shape
    """
    print("\n" + "=" * 80)
    print("MODEL STRUCTURE")
    print("=" * 80)
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    print("\n" + "=" * 80)
    print("PARAMETER COUNT")
    print("=" * 80)
    print(f"Total params      : {total_params:,}")
    print(f"Trainable params  : {trainable_params:,}")
    print(f"Non-trainable     : {non_trainable_params:,}")

    print("\n" + "=" * 80)
    print("TOP-LEVEL MODULE PARAMS")
    print("=" * 80)
    for name, module in model.named_children():
        mod_total = sum(p.numel() for p in module.parameters())
        mod_train = sum(p.numel() for p in module.parameters() if p.requires_grad)
        if mod_total > 0:
            print(f"{name:30s} total={mod_total:12,}  trainable={mod_train:12,}")

    # HuggingFace-style parameter count (if available)
    if hasattr(model, "num_parameters"):
        try:
            print("\n" + "=" * 80)
            print("HF MODEL NUM PARAMETERS")
            print("=" * 80)
            print(
                "num_parameters(trainable_only=False) =",
                model.num_parameters(trainable_only=False),
            )
            print(
                "num_parameters(trainable_only=True ) =",
                model.num_parameters(trainable_only=True),
            )
        except Exception:
            pass

    if verbose:
        print("\n" + "=" * 80)
        print("NAMED PARAMETERS (VERBOSE)")
        print("=" * 80)
        for name, param in model.named_parameters():
            print(
                f"{name:80s} "
                f"shape={tuple(param.shape)} "
                f"numel={param.numel():,} "
                f"trainable={param.requires_grad}"
            )


def inference(args, model_path, dset_test):
    # model = get_model(model_path=model_path)
    model = TinyTimeMixerForPrediction.from_pretrained(model_path)

    temp_dir = tempfile.mkdtemp()
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=temp_dir,
            per_device_eval_batch_size=args.batch_size,
            seed=args.random_seed,
            report_to="none",
        ),
    )
    # evaluate = zero-shot performance
    print("+" * 20, "Test MSE output:", "+" * 20)
    output = trainer.evaluate(dset_test)
    print(output)

    # get predictions

    predictions_dict = trainer.predict(dset_test)

    predictions_output = predictions_dict.predictions[
        0
    ]  # [N, F, C] combined (inverse-scaled)

    if args.use_internal_tsfm:
        input_data = predictions_dict.predictions[1]  # [N, L, C] (raw scale)
    else:
        input_data = predictions_dict.predictions[-3]
    forecast_groundtruth = predictions_dict.predictions[-2]  # [N, F, C] (raw scale)
    has_quantiles = model.config.multi_quantile_head

    if has_quantiles:
        comb_q = predictions_dict.predictions[-1]

    mse = np.mean((predictions_output - forecast_groundtruth) ** 2)
    print("MSE =", mse)

    save_folder = os.path.join(args.save_dir, "plots")

    os.makedirs(save_folder, exist_ok=True)

    num_samples = predictions_output.shape[0]
    num_plots = min(40, num_samples)

    ch = 0  # plot first channel by default
    for i in range(num_plots):
        idx = random.randint(0, num_samples - 1)

        forecast_main = predictions_output[idx, :, ch]
        forecast_ori = forecast_groundtruth[idx, :, ch]

        input_main = input_data[idx, :, ch]

        plt.figure(figsize=(12, 8))

        # Inputs
        plt.subplot(2, 1, 1)
        plt.plot(input_main, label="Input", linewidth=1.5)
        plt.title(f"Inputs (Sample {idx}, Channel {ch})")
        plt.legend()

        # Forecasts
        plt.subplot(2, 1, 2)
        plt.plot(forecast_main, label="Forecast", linewidth=1.5)
        plt.plot(forecast_ori, label="Ground Truth", linewidth=1)
        plt.title(f"Forecasts (Sample {idx}, Channel {ch})")
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, f"plot_{i}.png"))
        plt.close()

        # ---------------- Extra quantile plots ----------------
        if has_quantiles:
            # Helper to extract q10, q50, q90 lines
            def q1090(qarr, idx, ch):
                # qarr: [N, 9, F, C]; quantile order = [0.1,...,0.9], median at index 4
                q10 = qarr[idx, 0, :, ch]
                q50 = qarr[idx, 4, :, ch]
                q90 = qarr[idx, 8, :, ch]
                return q10, q50, q90

            # --- Combined quantiles ---
            q10, q50, q90 = q1090(comb_q, idx, ch)
            plt.figure(figsize=(12, 6))
            plt.plot(q50, label="Combined q50", linewidth=1.8)
            plt.plot(q10, label="Combined q10", linewidth=1.0)
            plt.plot(q90, label="Combined q90", linewidth=1.0)
            # ground truth for reference
            plt.plot(forecast_ori, label="Ground Truth", linewidth=1.0)
            # (optional) band shading
            plt.fill_between(
                np.arange(len(q50)), q10, q90, alpha=0.15, label="P80 band"
            )
            plt.title(f"Combined Quantiles (Sample {idx}, Channel {ch})")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_folder, f"quant_combined_{i}.png"))
            plt.close()

    print(f"Saved {num_plots} plots to: {save_folder}")

    print_model_summary(model, verbose=True)

    # # plot
    # plot_predictions(
    #     model=trainer.model,
    #     dset=dset_test,
    #     plot_dir=plot_path,
    #     plot_prefix="test_inference",
    #     channel=0,
    # )
    # print("Plots saved in location:", plot_path)


if __name__ == "__main__":
    # Arguments
    args = get_ttm_args()

    if args.use_internal_tsfm:
        print("internal TSFM")
        from tsfm.models.tinytimemixer import (
            TinyTimeMixerConfig,
            TinyTimeMixerForPrediction,
        )
    else:
        print("public TSFM")
        from tsfm_public.models.tinytimemixer import (
            TinyTimeMixerConfig,
            TinyTimeMixerForPrediction,
        )

    # Set seed
    set_seed(args.random_seed)

    logger.info(
        f"{'*' * 20} Pre-training a TTM for context len = {args.context_length}, forecast len = {args.forecast_length} {'*' * 20}"
    )

    # Data prep
    # Dataset
    TARGET_DATASET = "etth1"
    dataset_path = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"  # mention the dataset path
    timestamp_column = "date"
    id_columns = []  # mention the ids that uniquely identify a time-series.

    target_columns = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]

    # mention the train, valid and split config.
    split_config = {
        "train": [0, 8640],
        "valid": [8640, 11520],
        "test": [
            11520,
            14400,
        ],
    }

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
        context_length=args.context_length,
        prediction_length=args.forecast_length,
        scaling=True,
        encode_categorical=False,
        scaler_type="standard",
    )

    dset_train, dset_valid, dset_test = get_datasets(tsp, data, split_config)

    # Get model
    model = get_base_model(args)

    # Pretrain
    model_save_path = pretrain(args, model, dset_train, dset_valid)
    print("=" * 20, "Pretraining Completed!", "=" * 20)
    print("Model saved in location:", model_save_path)

    # inference

    inference(args=args, model_path=model_save_path, dset_test=dset_test)

    print("inference completed..")
