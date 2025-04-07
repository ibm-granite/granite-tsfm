"""
# TTM zero-shot and few-shot benchmarking on multiple datasets
Pre-trained TTM models will be fetched from the HuggingFace TTM Model Repositories as described below.

1. TTM-Granite-R1 pre-trained models can be found here: [TTM-R1 Model Card](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r1)
2. TTM-Granite-R2 pre-trained models can be found here: [TTM-R2 Model Card](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2)
3. TTM-Research-Use pre-trained models can be found here: [TTM-Research-Use Model Card](https://huggingface.co/ibm-research/ttm-research-r2)

Every model card has a suite of TTM models. Please read the respective model cards for usage instructions.
"""

## Imports
import math
import tempfile
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed
from transformers.integrations import INTEGRATION_TO_CALLBACK

from tsfm_public import TrackingCallback, count_parameters, load_dataset
from tsfm_public.models.tinytimemixer.utils import get_ttm_args
from tsfm_public.toolkit.get_model import get_model
from tsfm_public.toolkit.lr_finder import optimal_lr_finder
from tsfm_public.toolkit.visualization import plot_predictions


warnings.filterwarnings("ignore")

# Arguments
args = get_ttm_args()

# Set seed
set_seed(args.random_seed)

## Important arguments
# Specify model parameters
CONTEXT_LENGTH = args.context_length
FORECAST_LENGTH = args.forecast_length
FREEZE_BACKBONE = True

# Other args
EPOCHS = args.num_epochs
NUM_WORKERS = args.num_workers

# Make sure all the datasets in the following `list_datasets` are
# saved in the `DATA_ROOT_PATH` folder. Or, change it accordingly.
# Refer to the load_datasets() function
# in notebooks/hfdemo/tinytimemixer/utils/ttm_utils.py
# to see how it is used.
DATA_ROOT_PATH = args.data_root_path

# This is where results will be saved
OUT_DIR = args.save_dir

MODEL_PATH = args.hf_model_path

print(f"{'*' * 20} Pre-training a TTM for context len = {CONTEXT_LENGTH}, forecast len = {FORECAST_LENGTH} {'*' * 20}")

## List of benchmark datasets (TTM was not pre-trained on any of these)

if args.datasets is None:
    list_datasets = [
        "etth1",
        "etth2",
        "ettm1",
        "ettm2",
        "weather",
        "electricity",
        "traffic",
        # "exchange",
        # "zafnoo",
        # "solar" # please note that, solar is part of TTM pre-training.
        #         # But, adding here to do in-distribution testing.
        #         # solar results should be ignored for TTM for zero-shot ranking.
    ]

else:
    list_datasets = [dataset.strip() for dataset in args.datasets.split(",")]


all_results = {
    "dataset": [],
    "zs_mse": [],
    "fs5_mse": [],
}

# Loop over data
for DATASET in list_datasets:
    try:
        print()
        print("=" * 100)
        print(
            f"Running zero-shot/few-shot for TTM-{CONTEXT_LENGTH} on dataset = {DATASET}, forecast_len = {FORECAST_LENGTH}"
        )

        print(f"Model will be loaded from {MODEL_PATH}")
        SUBDIR = f"{OUT_DIR}/{DATASET}"

        # Set batch size
        if DATASET == "traffic":
            BATCH_SIZE = 8
        elif DATASET == "electricity":
            BATCH_SIZE = 32
        else:
            BATCH_SIZE = 64

        # Data prep: Get dataset
        _, _, dset_test = load_dataset(
            DATASET,
            CONTEXT_LENGTH,
            FORECAST_LENGTH,
            dataset_root_path=DATA_ROOT_PATH,
            use_frequency_token=args.enable_prefix_tuning,
            enable_padding=False,
        )

        #############################################################
        ##### Use the pretrained model in zero-shot forecasting #####
        #############################################################
        # Load model
        zeroshot_model = get_model(
            model_path=MODEL_PATH, context_length=CONTEXT_LENGTH, prediction_length=FORECAST_LENGTH
        )

        # zeroshot_trainer
        zeroshot_trainer = Trainer(
            model=zeroshot_model,
            args=TrainingArguments(
                output_dir=tempfile.mkdtemp(),
                per_device_eval_batch_size=BATCH_SIZE,
                seed=args.random_seed,
            ),
            eval_dataset=dset_test,
        )

        # evaluate = zero-shot performance
        print("+" * 20, "Test MSE zero-shot", "+" * 20)
        zeroshot_output = zeroshot_trainer.evaluate(dset_test)
        print(zeroshot_output)
        print("+" * 60)

        # Plot

        if args.plot:
            plot_predictions(
                model=zeroshot_trainer.model,
                dset=dset_test,
                plot_dir=SUBDIR,
                num_plots=10,
                plot_prefix="test_zeroshot",
                channel=0,
            )
            plt.close()

        # write results
        all_results["dataset"].append(DATASET)
        all_results["zs_mse"].append(zeroshot_output["eval_loss"])

    except Exception as e:
        print(f"Reason for exception: {e}")
        # write dummy results
        all_results["dataset"].append(DATASET)
        all_results["zs_mse"].append(np.nan)

    ################################################################
    ## Use the pretrained model in few-shot 5% and 10% forecasting #
    ################################################################
    try:
        if args.fewshot == 0:
            raise Exception("fewshot is not enabled")
        for fewshot_percent in [5]:
            # Set learning rate
            learning_rate = None  # `None` value indicates that the optimal_lr_finder() will be used

            print("-" * 20, f"Running few-shot {fewshot_percent}%", "-" * 20)
            # Data prep: Get dataset
            dset_train, dset_val, dset_test = load_dataset(
                DATASET,
                CONTEXT_LENGTH,
                FORECAST_LENGTH,
                fewshot_fraction=fewshot_percent / 100,
                dataset_root_path=DATA_ROOT_PATH,
                use_frequency_token=args.enable_prefix_tuning,
                enable_padding=False,
            )

            # change head dropout to 0.7 for ett datasets
            if "ett" in DATASET:
                finetune_forecast_model = get_model(
                    model_path=MODEL_PATH,
                    context_length=CONTEXT_LENGTH,
                    prediction_length=FORECAST_LENGTH,
                    head_dropout=0.7,
                )
            else:
                finetune_forecast_model = get_model(
                    model_path=MODEL_PATH,
                    context_length=CONTEXT_LENGTH,
                    prediction_length=FORECAST_LENGTH,
                )

            if FREEZE_BACKBONE:
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

            if learning_rate is None:
                learning_rate, finetune_forecast_model = optimal_lr_finder(
                    finetune_forecast_model,
                    dset_train,
                    batch_size=BATCH_SIZE,
                    enable_prefix_tuning=args.enable_prefix_tuning,
                )
                print("OPTIMAL SUGGESTED LEARNING RATE =", learning_rate)

            print(f"Using learning rate = {learning_rate}")

            # This is to save space during exhaustive benchmarking, use specific directory if the saved models are needed
            tmp_dir = tempfile.mkdtemp()

            finetune_forecast_args = TrainingArguments(
                output_dir=tmp_dir,
                overwrite_output_dir=True,
                learning_rate=learning_rate,
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
                logging_dir=tmp_dir,  # Make sure to specify a logging directory
                load_best_model_at_end=True,  # Load the best model when training ends
                metric_for_best_model="eval_loss",  # Metric to monitor for early stopping
                greater_is_better=False,  # For loss
                seed=args.random_seed,
            )

            # Create the early stopping callback
            early_stopping_callback = EarlyStoppingCallback(
                early_stopping_patience=10,  # Number of epochs with no improvement after which to stop
                early_stopping_threshold=0.0,  # Minimum improvement required to consider as improvement
            )
            tracking_callback = TrackingCallback()

            # Optimizer and scheduler
            optimizer = AdamW(finetune_forecast_model.parameters(), lr=learning_rate)
            scheduler = OneCycleLR(
                optimizer,
                learning_rate,
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
            finetune_forecast_trainer.remove_callback(INTEGRATION_TO_CALLBACK["codecarbon"])

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

            if args.plot:
                # Plot
                plot_predictions(
                    model=finetune_forecast_trainer.model,
                    dset=dset_test,
                    plot_dir=SUBDIR,
                    num_plots=10,
                    plot_prefix=f"test_fewshot_{fewshot_percent}",
                    channel=0,
                )
                plt.close()

            # write results
            all_results[f"fs{fewshot_percent}_mse"].append(fewshot_output["eval_loss"])

    except Exception as e:
        print(f"Reason for exception: {e}")
        fewshot_percent = 5
        # write dummy results
        all_results[f"fs{fewshot_percent}_mse"].append(np.nan)

    df_out = pd.DataFrame(all_results).round(3)
    print(df_out[["dataset", "zs_mse", "fs5_mse"]])
    df_out.to_csv(f"{OUT_DIR}/results_zero_few.csv")
    df_out.to_csv(f"{OUT_DIR}/results_zero_few.csv")
