#!/usr/bin/env python
# coding: utf-8

# TTM pre-training example.
# This scrips provides a toy example to pretrain a Tiny Time Mixer (TTM) model on
# the `etth1` dataset. For pre-training TTM on a much large set of datasets, please
# have a look at our paper: https://arxiv.org/pdf/2401.03955.pdf
# If you want to directly utilize the pre-trained models. Please use them from the
# Hugging Face Hub: https://huggingface.co/ibm/TTM
# Have a look at the fine-tune scripts for example usecases of the pre-trained
# TTM models.

# Basic usage:
# python ttm_pretrain_sample.py --data_root_path datasets/
# See the get_ttm_args() function to know more about other TTM arguments

# Standard
import math
import os

# Third Party
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed

# Local
from tsfm_public.models.tinytimemixer import (
    TinyTimeMixerConfig,
    TinyTimeMixerForPrediction,
)
from tsfm_public.models.tinytimemixer.utils import get_data, get_ttm_args


# Arguments
args = get_ttm_args()


def get_model(args):
    # Pre-train a `TTM` forecasting model
    config = TinyTimeMixerConfig(
        context_length=args.context_length,
        prediction_length=args.forecast_length,
        patch_length=args.patch_length,
        num_input_channels=1,
        patch_stride=args.patch_length,
        d_model=args.d_model,
        num_layers=2,
        mode="common_channel",
        expansion_factor=2,
        dropout=0.2,
        head_dropout=0.2,
        scaling="std",
        gated_attn=True,
        adaptive_patching_levels=args.adaptive_patching_levels,
        # decoder params
        decoder_num_layers=2,
        decoder_adaptive_patching_levels=0,
        decoder_mode="common_channel",
        decoder_raw_residual=False,
        use_decoder=True,
        decoder_d_model=args.decoder_d_model,
    )

    model = TinyTimeMixerForPrediction(config)
    return model


def pretrain(args, model, dset_train, dset_val):
    trainer_args = TrainingArguments(
        output_dir=os.path.join(args.save_dir, "checkpoint"),
        overwrite_output_dir=True,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        evaluation_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        dataloader_num_workers=args.num_workers,
        ddp_find_unused_parameters=False,
        report_to="tensorboard",
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=1,
        logging_dir=os.path.join(args.save_dir, "logs"),  # Make sure to specify a logging directory
        load_best_model_at_end=True,  # Load the best model when training ends
        metric_for_best_model="eval_loss",  # Metric to monitor for early stopping
        greater_is_better=False,  # For loss
    )

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = OneCycleLR(
        optimizer,
        args.learning_rate,
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
    trainer.save_model(os.path.join(args.save_dir, "ttm_pretrained"))


if __name__ == "__main__":
    # Set seed
    set_seed(args.random_seed)

    print(
        "*" * 20,
        f"Pre-training a TTM for context len = {args.context_length}, forecast len = {args.forecast_length}",
        "*" * 20,
    )

    # Data prep
    dset_train, dset_val, dset_test = get_data(
        args.dataset,
        args.context_length,
        args.forecast_length,
        data_root_path=args.data_root_path,
    )
    print("Length of the train dataset =", len(dset_train))

    # Get model
    model = get_model(args)

    # Pretrain
    pretrain(args, model, dset_train, dset_val)
    print("=" * 20, "Pretraining Completed!", "=" * 20)
