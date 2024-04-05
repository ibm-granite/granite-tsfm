#!/usr/bin/env python
# coding: utf-8

# TTM zero-shot and few-shot scenarios on multiple datasets.

# Standard
import math

# Third Party
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed
import numpy as np
import pandas as pd

# First Party
from notebooks.hfdemo.tinytimemixer.utils import (
    TrackingCallback,
    count_parameters,
    get_data,
    get_ttm_args,
    plot_preds,
)

# Local
from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction

# Set seed
set_seed(42)

# Get args
args = get_ttm_args()

list_datasets = [
    "etth1",
    "etth2",
    "ettm1",
    "ettm2",
    "weather",
    "electricity",
    "traffic",
]

EPOCHS = 50
NUM_WORKERS = 16
OUT_DIR = args.save_dir

# get model path
hf_model_path = "ibm/TTM"
if args.context_length == 512:
    hf_model_branch = "main"
elif args.context_length == 1024:
    hf_model_branch = "1024_96_v1"
else:
    raise ValueError(
        "Current supported context lengths are 512 and 1024. Stay tuned for more TTMs!"
    )

all_results = {
    "dataset": [],
    "zs_mse": [],
    "fs5_mse": [],
    "fs10_mse": [],
    "zs_eval_time": [],
    "fs5_mean_epoch_time": [],
    "fs5_total_train_time": [],
    "fs10_mean_epoch_time": [],
    "fs10_total_train_time": [],
    "fs5_best_val_metric": [],
    "fs10_best_val_metric": [],
}
# Loop over data
for DATASET in list_datasets:
    print()
    print("=" * 100)
    print(
        f"Running zero-shot/few-shot for TTM-{args.context_length} on dataset = {DATASET}, forecast_len = {args.forecast_length}"
    )
    print(f"Model will be loaded from {hf_model_path}/{hf_model_branch}")
    SUBDIR = f"{OUT_DIR}/{DATASET}"

    # Set batch size
    if DATASET == "traffic":
        BATCH_SIZE = 8
    elif DATASET == "electricity":
        BATCH_SIZE = 32
    else:
        BATCH_SIZE = 64

    # Data prep: Get dataset
    _, _, dset_test = get_data(DATASET, args.context_length, args.forecast_length)

    #############################################################
    ##### Use the pretrained model in zero-shot forecasting #####
    #############################################################
    # Load model
    zeroshot_model = TinyTimeMixerForPrediction.from_pretrained(
        hf_model_path, revision=hf_model_branch
    )

    # zeroshot_trainer
    zeroshot_trainer = Trainer(
        model=zeroshot_model,
        args=TrainingArguments(
            output_dir=f"{SUBDIR}/zeroshot",
            per_device_eval_batch_size=BATCH_SIZE,
        ),
        eval_dataset=dset_test,
    )

    # evaluate = zero-shot performance
    print("+" * 20, "Test MSE zero-shot", "+" * 20)
    zeroshot_output = zeroshot_trainer.evaluate(dset_test)
    print(zeroshot_output)
    print("+" * 60)
    all_results["zs_eval_time"].append(zeroshot_output["eval_runtime"])

    # Plot
    plot_preds(
        zeroshot_trainer,
        dset_test,
        SUBDIR,
        num_plots=10,
        plot_prefix="test_zeroshot",
        channel=0,
    )

    # write results
    all_results["dataset"].append(DATASET)
    all_results["zs_mse"].append(zeroshot_output["eval_loss"])

    ################################################################
    ## Use the pretrained model in few-shot 5% and 10% forecasting #
    ################################################################
    for fewshot_percent in [5, 10]:
        print("-" * 20, f"Running few-shot {fewshot_percent}%", "-" * 20)
        # Data prep: Get dataset
        dset_train, dset_val, dset_test = get_data(
            DATASET,
            args.context_length,
            args.forecast_length,
            fewshot_fraction=fewshot_percent / 100,
        )

        # change head dropout to 0.7 for ett datasets
        if "ett" in DATASET:
            finetune_forecast_model = TinyTimeMixerForPrediction.from_pretrained(
                hf_model_path, revision=hf_model_branch, head_dropout=0.7
            )
        else:
            finetune_forecast_model = TinyTimeMixerForPrediction.from_pretrained(
                hf_model_path, revision=hf_model_branch
            )

        if args.freeze_backbone:
            print(
                "Number of params before freezing backbone",
                count_parameters(finetune_forecast_model),
            )

            # Freeze the backbone of the model
            for param in finetune_forecast_model.backbone.parameters():
                param.requires_grad = False

            # Count params
            print(
                "Number of params after freezing the backbone",
                count_parameters(finetune_forecast_model),
            )

        print(f"Using learning rate = {args.learning_rate}")
        finetune_forecast_args = TrainingArguments(
            output_dir=f"{SUBDIR}/fewshot_{fewshot_percent}",
            overwrite_output_dir=True,
            learning_rate=args.learning_rate,
            num_train_epochs=EPOCHS,
            do_eval=True,
            evaluation_strategy="epoch",
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            dataloader_num_workers=NUM_WORKERS,
            report_to=None,
            save_strategy="epoch",
            logging_strategy="epoch",
            save_total_limit=1,
            logging_dir=f"{SUBDIR}/fewshot_{fewshot_percent}",  # Make sure to specify a logging directory
            load_best_model_at_end=True,  # Load the best model when training ends
            metric_for_best_model="eval_loss",  # Metric to monitor for early stopping
            greater_is_better=False,  # For loss
        )

        # Create the early stopping callback
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=10,  # Number of epochs with no improvement after which to stop
            early_stopping_threshold=0.0,  # Minimum improvement required to consider as improvement
        )
        tracking_callback = TrackingCallback()

        # Optimizer and scheduler
        optimizer = AdamW(finetune_forecast_model.parameters(), lr=args.learning_rate)
        scheduler = OneCycleLR(
            optimizer,
            args.learning_rate,
            epochs=EPOCHS,
            steps_per_epoch=math.ceil(len(dset_train) / (BATCH_SIZE)),
        )

        finetune_forecast_trainer = Trainer(
            model=finetune_forecast_model,
            args=finetune_forecast_args,
            train_dataset=dset_train,
            eval_dataset=dset_val,
            callbacks=[early_stopping_callback, tracking_callback],
            optimizers=(optimizer, scheduler),
        )

        # Fine tune
        finetune_forecast_trainer.train()

        # Evaluation
        print(
            "+" * 20,
            f"Test MSE after few-shot {fewshot_percent}% fine-tuning",
            "+" * 20,
        )
        fewshot_output = finetune_forecast_trainer.evaluate(dset_test)
        print(fewshot_output)
        print("+" * 60)

        # Plot
        plot_preds(
            finetune_forecast_trainer,
            dset_test,
            SUBDIR,
            num_plots=10,
            plot_prefix=f"test_fewshot_{fewshot_percent}",
            channel=0,
        )

        # write results
        all_results[f"fs{fewshot_percent}_mse"].append(fewshot_output["eval_loss"])
        all_results[f"fs{fewshot_percent}_mean_epoch_time"].append(
            tracking_callback.mean_epoch_time
        )
        all_results[f"fs{fewshot_percent}_total_train_time"].append(
            tracking_callback.total_train_time
        )
        all_results[f"fs{fewshot_percent}_best_val_metric"].append(
            tracking_callback.best_eval_metric
        )

    df_out = pd.DataFrame(all_results).round(3)
    print(df_out[["dataset", "zs_mse", "fs5_mse", "fs10_mse"]])
    df_out.to_csv(f"{OUT_DIR}/results_zero_few.csv")
