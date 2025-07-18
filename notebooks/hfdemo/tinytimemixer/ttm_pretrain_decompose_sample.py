#!/usr/bin/env python
# coding: utf-8

import logging
import math
import os
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import tempfile

import pandas as pd
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from torch.utils.data import ConcatDataset
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed

from tsfm_public import TimeSeriesPreprocessor, get_datasets, load_dataset
from tsfm_public.models.tinytimemixer import (
    TinyTimeMixerConfig,
    TinyTimeMixerForDecomposedPrediction,
)
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
        scaling=args.scaling,
        gated_attn=True,
        adaptive_patching_levels=args.adaptive_patching_levels,
        # decoder params
        decoder_num_layers=args.decoder_num_layers,  # increase the number of layers if we want more complex models
        decoder_adaptive_patching_levels=0,
        decoder_mode=args.decoder_mode,
        decoder_raw_residual=False,
        use_decoder=True,
        decoder_d_model=args.decoder_d_model,
        multi_scale=args.multi_scale,
        register_tokens=args.register_tokens,
        fft_length=args.fft_length,
        patch_gating=args.patch_gating,
        multi_scale_loss=args.multi_scale_loss,
        use_fft_embedding=args.use_fft_embedding,
        self_attn=args.self_attn,
        enable_fourier_attention=args.enable_fourier_attention,
        trend_patch_length=args.trend_patch_length,
        trend_patch_stride=args.trend_patch_stride,
        trend_d_model=args.trend_d_model,
        trend_decoder_d_model=args.trend_decoder_d_model,
        trend_num_layers=args.trend_num_layers,
        trend_decoder_num_layers=args.trend_decoder_num_layers,
    )

    model = TinyTimeMixerForDecomposedPrediction(config)

    for name, p in model.named_parameters():
        def hook(grad, n=name):
            if grad is None:
                print(f"[WARN] No grad for {n}")
        p.register_hook(hook)

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

    learning_rate = args.learning_rate

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

    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    # scheduler = OneCycleLR(
    #     optimizer,
    #     learning_rate,
    #     epochs=args.num_epochs,
    #     steps_per_epoch=math.ceil(len(dset_train) / args.batch_size),
    #     # steps_per_epoch=math.ceil(len(dset_train) / (args.batch_size * args.num_gpus)),
    # )

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
            # optimizers=(optimizer),
            callbacks=[early_stopping_callback],
        )
    else:
        trainer = Trainer(
            model=model,
            args=trainer_args,
            train_dataset=dset_train,
            eval_dataset=dset_val,
            optimizers=(optimizer, scheduler),
            # optimizers=(optimizer,),
        )

    # Train
    trainer.train()

    # Save the pretrained model

    model_save_path = os.path.join(args.save_dir, "ttm_pretrained")
    trainer.save_model(model_save_path)
    return model_save_path


def inference(args, model_path, dset_test):
    model = TinyTimeMixerForDecomposedPrediction.from_pretrained(model_path)
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

    predictions_output = predictions_dict.predictions[0] # samples x forecast_len x channels
    trend_prediction_outputs = predictions_dict.predictions[1] # samples x forecast_len x channels
    residual_prediction_outputs = predictions_dict.predictions[2] # samples x forecast_len x channels
    input_data = predictions_dict.predictions[3] # samples x input_len x channels
    trend_input = predictions_dict.predictions[4] # samples x input_len x channels
    residual_input = predictions_dict.predictions[5] # samples x input_len x channels
    forecast_groundtruth = predictions_dict.predictions[6] # samples x input_len x channels
    combined_input = predictions_dict.predictions[7]

    # Create folder
    save_folder = "random_plots"
    os.makedirs(save_folder, exist_ok=True)

    num_samples = predictions_output.shape[0]
    num_plots = 10

    for i in range(num_plots):
        idx = random.randint(0, num_samples - 1)

        forecast_main = predictions_output[idx, :, 0]
        forecast_trend = trend_prediction_outputs[idx, :, 0]
        forecast_residual = residual_prediction_outputs[idx, :, 0]
        forecast_ori = forecast_groundtruth[idx,:,0]

        input_main = input_data[idx, :, 0]
        input_trend = trend_input[idx, :, 0]
        input_residual = residual_input[idx, :, 0]
        input_combined = combined_input[idx, :, 0]

        plt.figure(figsize=(12, 8))

        # Inputs
        plt.subplot(2, 1, 1)
        plt.plot(input_main, label='Input', linewidth=1.5)
        plt.plot(input_trend, label='Trend Input', linewidth=1)
        plt.plot(input_residual, label='Residual Input', linewidth=1)
        plt.plot(input_combined, '--', label='Trend+Residual', linewidth=1)
        plt.title(f"Inputs (Sample {idx})")
        plt.legend()

        # Forecasts
        plt.subplot(2, 1, 2)
        plt.plot(forecast_main, label='Forecast', linewidth=1.5)
        plt.plot(forecast_trend, label='Trend Forecast', linewidth=1)
        plt.plot(forecast_residual, label='Residual Forecast', linewidth=1)
        plt.plot(forecast_ori, label='Original Forecast', linewidth=1)
        plt.title(f"Forecasts (Sample {idx})")
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, f"plot_{i}.png"))
        plt.close()

    # get backbone embeddings (if needed for further analysis)

    # backbone_embedding = predictions_dict.predictions[0]

    # print(backbone_embedding.shape)

    # plot_path = os.path.join(args.save_dir, "plots")
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

    # Set seed
    set_seed(args.random_seed)

    logger.info(
        f"{'*' * 20} Pre-training a TTM for context len = {args.context_length}, forecast len = {args.forecast_length} {'*' * 20}"
    )

    # Data prep
    # Dataset
    # TARGET_DATASET = "etth1"
    # dataset_path = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"  # mention the dataset path
    # timestamp_column = "date"
    # id_columns = []  # mention the ids that uniquely identify a time-series.

    # target_columns = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]

    # # mention the train, valid and split config.
    # split_config = {
    #     "train": [0, 8640],
    #     "valid": [8640, 11520],
    #     "test": [
    #         11520,
    #         14400,
    #     ],
    # }

    # data = pd.read_csv(
    #     dataset_path,
    #     parse_dates=[timestamp_column],
    # )

    # column_specifiers = {
    #     "timestamp_column": timestamp_column,
    #     "id_columns": id_columns,
    #     "target_columns": target_columns,
    #     "control_columns": [],
    # }

    # tsp = TimeSeriesPreprocessor(
    #     **column_specifiers,
    #     context_length=args.context_length,
    #     prediction_length=args.forecast_length,
    #     scaling=True,
    #     encode_categorical=False,
    #     scaler_type="standard",
    # )

    dset_train, dset_valid, dset_test = load_dataset(dataset_name = "ettm2",
                                                    context_length = args.context_length,
                                                    forecast_length = args. forecast_length,
                                                    dataset_root_path = "/dccstor/tsfm23/datasets")
    
    # get_datasets(tsp, data, split_config)
    # dset_combined = ConcatDataset([dset_train, dset_test])

    # Get model
    model = get_base_model(args)

    # Pretrain
    model_save_path = pretrain(args, model, dset_train, dset_valid)
    print("=" * 20, "Pretraining Completed!", "=" * 20)
    print("Model saved in location:", model_save_path)

    # inference

    inference(args=args, model_path=model_save_path, dset_test=dset_test)

    print("inference completed..")
