#!/usr/bin/env python
# coding: utf-8

import logging
import math
import os
import tempfile

import pandas as pd
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import ConcatDataset
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed

from tsfm_public import TimeSeriesPreprocessor, get_datasets,load_dataset
from tsfm_public.models.tinytimemixer import (
    TinyTimeMixerConfig,
    TinyTimeMixerForPrediction,
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

import torch
from torch.utils.data import Dataset

class PrependZeroContextWrapper(Dataset):
    def __init__(self, base_dataset: Dataset, context_len: int):
        """
        Args:
            base_dataset: Any dataset whose __getitem__ returns a dict with:
                - 'past_values': [seq_len, c]
            context_len: Desired sequence length after prepending zeros.
        """
        self.base_dataset = base_dataset
        self.context_len = context_len

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]

        past_values = item["past_values"]  # [seq_len, c]
        if past_values.dim() != 2:
            raise ValueError(f"'past_values' must be [seq_len, c], got {past_values.shape}")

        seq_len, c = past_values.shape
        if self.context_len < seq_len:
            raise ValueError(
                f"context_len ({self.context_len}) < seq_len ({seq_len})"
            )
        pad_len = self.context_len - seq_len

        # ---- pad past_values ----
        padded_past_values = torch.cat([
            torch.zeros((pad_len, c), dtype=past_values.dtype),
            past_values
        ], dim=0)

        # ---- create past_observed_mask ----
        # mask for actual values: [seq_len, c] of ones
        actual_mask = torch.ones((seq_len, c), dtype=torch.float32)
        # prepend zeros for padded part: [pad_len, c]
        pad_mask = torch.zeros((pad_len, c), dtype=torch.float32)
        # combine: [context_len, c]
        padded_mask = torch.cat([pad_mask, actual_mask], dim=0)

        # ---- build output dict ----
        out = {**item}
        out["past_values"] = padded_past_values
        out["past_observed_mask"] = padded_mask

        return out
    
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
        decoder_mode="common_channel",
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
        disable_pad_activations = args.disable_pad_activations,
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
    # scheduler = OneCycleLR(
    #     optimizer,
    #     learning_rate,
    #     epochs=args.num_epochs,
    #     steps_per_epoch=math.ceil(len(dset_train) / args.batch_size),
    #     # steps_per_epoch=math.ceil(len(dset_train) / (args.batch_size * args.num_gpus)),
    # )
    # scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)

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
    model = get_model(model_path=model_path)

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

    predictions_np = predictions_dict.predictions[0]

    print(predictions_np.shape)

    # get backbone embeddings (if needed for further analysis)

    backbone_embedding = predictions_dict.predictions[1]

    print(backbone_embedding.shape)

    plot_path = os.path.join(args.save_dir, "plots")
    # plot
    plot_predictions(
        model=trainer.model,
        dset=dset_test,
        plot_dir=plot_path,
        plot_prefix="test_inference",
        channel=0,
        plot_context = 512,
        num_plots = 100,
    )
    print("Plots saved in location:", plot_path)


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


        # Example: try multiple context lengths
    # k_values = [128, 64, 336]  # replace with your desired Ks
    k_values = [128, 64, 336]  # replace with your desired Ks
    train_wrapped_list = []
    valid_wrapped_list = []
    test_wrapped_list = []

    for K in k_values:
        dset_train, dset_valid, dset_test = load_dataset(
            dataset_name="etth1",
            context_length=K,
            forecast_length=args.forecast_length,
            dataset_root_path="/dccstor/tsfm23/datasets"
        )

        dset_train = PrependZeroContextWrapper(dset_train, args.context_length)
        dset_valid = PrependZeroContextWrapper(dset_valid, args.context_length)
        dset_test  = PrependZeroContextWrapper(dset_test,  args.context_length)

        train_wrapped_list.append(dset_train)
        valid_wrapped_list.append(dset_valid)
        test_wrapped_list.append(dset_test)

    # Combine them
    final_train_dataset = ConcatDataset(train_wrapped_list)
    final_valid_dataset = ConcatDataset(valid_wrapped_list)
    final_test_dataset  = ConcatDataset(test_wrapped_list)


    dset_train, dset_valid, dset_test = load_dataset(dataset_name = "etth1",
                                                    context_length = args.context_length,
                                                    forecast_length = args. forecast_length,
                                                    dataset_root_path = "/dccstor/tsfm23/datasets")
    
    # dset_train = PrependZeroContextWrapper(dset_train,args.context_length)
    # dset_valid = PrependZeroContextWrapper(dset_valid,args.context_length)
    # dset_test = PrependZeroContextWrapper(dset_test,args.context_length)

    # dset_train, dset_valid, dset_test = get_datasets(tsp, data, split_config)
    # dset_combined = ConcatDataset([dset_train, dset_test])

    # Get model
    model = get_base_model(args)

    # Pretrain
    # model_save_path = pretrain(args, model, final_train_dataset, final_valid_dataset)
    model_save_path = pretrain(args, model, dset_train, dset_valid)

    # model_save_path = "/dccstor/tsfm-irl/vijaye12/hacking/ttm_var/ttm_variable_big/TTM_cl-512_fl-48_ns-200000_pl-64_apl-3_es-True_dr-0.1_hdr-0.2_ept-True_lr-0.001/checkpoint-step-60000"
    print("=" * 20, "Pretraining Completed!", "=" * 20)
    print("Model saved in location:", model_save_path)

    # inference

    inference(args=args, model_path=model_save_path, dset_test=final_test_dataset)

    print("inference completed..")
