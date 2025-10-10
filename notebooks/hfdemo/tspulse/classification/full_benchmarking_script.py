import csv
import json
import math
import os
import tempfile
import warnings

import numpy as np
import torch
import transformers
from packaging import version
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Subset, random_split
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed
from transformers.data.data_collator import default_data_collator
from transformers.trainer_utils import RemoveColumnsCollator

from tsfm_public.models.tspulse import TSPulseForClassification
from tsfm_public.toolkit.dataset import ClassificationDFDataset
from tsfm_public.toolkit.lr_finder import optimal_lr_finder
from tsfm_public.toolkit.time_series_classification_preprocessor import TimeSeriesClassificationPreprocessor
from tsfm_public.toolkit.util import convert_tsfile_to_dataframe


warnings.filterwarnings("ignore")

REQUIRED_TORCH_VERSION = "2.4.0+cu121"
REQUIRED_TRANSFORMERS_VERSION = "4.44.0"

if torch.__version__ != REQUIRED_TORCH_VERSION:
    print("torch.__version__ : ", torch.__version__)
    raise RuntimeError(
        f"Torch version mismatch: Please use {REQUIRED_TORCH_VERSION} for reproducibility of classification scores."
    )

MIN_TRANSFORMERS_VERSION = "4.44.0"
MAX_TRANSFORMERS_VERSION = "4.50.3"

current_version = transformers.__version__

if not (
    version.parse(MIN_TRANSFORMERS_VERSION)
    <= version.parse(current_version)
    <= version.parse(MAX_TRANSFORMERS_VERSION)
):
    raise RuntimeError(
        f"Transformers version mismatch: Please use a version between {MIN_TRANSFORMERS_VERSION} and {MAX_TRANSFORMERS_VERSION} "
        f"for reproducibility of classification scores. Found version: {current_version}"
    )

OUT_DIR = "tspulse_finetuned_models/"

json_file = "config.json"
with open(json_file, "r") as file:
    clf_params = json.load(file)


def main(dataset_name):
    seed = 42
    set_seed(seed)
    path = f"datasets/{dataset_name}/{dataset_name}_TRAIN.ts"  # train

    df_base = convert_tsfile_to_dataframe(
        path,
        return_separate_X_and_y=False,
    )
    label_column = "class_vals"
    input_columns = [f"dim_{i}" for i in range(df_base.shape[1] - 1)]

    tsp = TimeSeriesClassificationPreprocessor(
        input_columns=input_columns,
        label_column=label_column,
        scaling=True,
    )

    tsp.train(df_base)

    df_prep = tsp.preprocess(df_base)
    base_dataset = ClassificationDFDataset(
        df_prep,
        id_columns=[],
        timestamp_column=None,
        input_columns=input_columns,
        label_column=label_column,
        context_length=512,
        static_categorical_columns=[],
        stride=1,
        enable_padding=False,
        full_series=True,
    )

    path = f"datasets/{dataset_name}/{dataset_name}_TEST.ts"  # test

    df_test = convert_tsfile_to_dataframe(
        path,
        return_separate_X_and_y=False,
    )
    label_column = "class_vals"
    input_columns = [f"dim_{i}" for i in range(df_test.shape[1] - 1)]

    tsp = TimeSeriesClassificationPreprocessor(
        input_columns=input_columns,
        label_column=label_column,
        scaling=True,
    )

    tsp.train(df_test)

    df_prep = tsp.preprocess(df_test)

    test_dataset = ClassificationDFDataset(
        df_prep,
        id_columns=[],
        timestamp_column=None,
        input_columns=input_columns,
        label_column=label_column,
        context_length=512,
        static_categorical_columns=[],
        stride=1,
        enable_padding=False,
        full_series=True,
    )

    # validation set
    if dataset_name == "DuckDuckGeese":
        data_dict = clf_params[dataset_name]["DATA_PARAMS"]

        def k_fold_cv(skf, dataset, kth_fold):
            y = []
            for idx in range(len(dataset)):
                y.append(dataset[idx]["target_values"])

            for i, (train_indices, val_indices) in enumerate(skf.split(np.zeros(len(y)), y)):
                if i == kth_fold:
                    train_dataset = Subset(dataset, train_indices)
                    val_dataset = Subset(dataset, val_indices)
            return train_dataset, val_dataset

        skf = StratifiedKFold(n_splits=data_dict["num_folds"], shuffle=True)
        train_dataset, valid_dataset = k_fold_cv(skf, base_dataset, data_dict["kth_fold"])
    else:
        dataset_size = len(base_dataset)
        print(dataset_size)
        split_valid_ratio = 0.1
        val_size = int(split_valid_ratio * dataset_size)  # 10% valid split
        train_size = dataset_size - val_size
        train_dataset, valid_dataset = random_split(base_dataset, [train_size, val_size])

    config_dict = clf_params[dataset_name]["MODEL_PARAMS"]
    config_dict["loss"] = "cross_entropy"
    config_dict["ignore_mismatched_sizes"] = True

    config_dict["num_input_channels"] = tsp.num_input_channels
    config_dict["num_targets"] = df_base["class_vals"].nunique()

    model = TSPulseForClassification.from_pretrained(
        "ibm-granite/granite-timeseries-tspulse-r1", revision="tspulse-block-dualhead-512-p16-r1", **config_dict
    )
    model = model.to("cuda").float()

    # Freezing Backbone except patch embedding layer....
    for param in model.backbone.parameters():
        param.requires_grad = False

    for param in model.backbone.time_encoding.parameters():
        param.requires_grad = True
    for param in model.backbone.fft_encoding.parameters():
        param.requires_grad = True

    temp_dir = tempfile.mkdtemp()

    suggested_lr = None

    train_dict = clf_params[dataset_name]["TRAINING_PARAMS"]
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
    print("Suggested LR : ", suggested_lr)
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

    # Create the early stopping callback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=clf_params[dataset_name]["EARYL_STOPPING_PARAMS"][
            "early_stopping_patience"
        ],  # Number of epochs with no improvement after which to stop
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

    predictions_dict = finetune_trainer.predict(test_dataset)
    preds_np = predictions_dict.predictions[0]

    remove_columns_collator = RemoveColumnsCollator(
        data_collator=default_data_collator,
        signature_columns=["target_values"],
        logger=None,
        description=None,
        model_name="temp",
    )

    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=remove_columns_collator)
    target_list = []
    for batch in test_dataloader:
        batch_labels = batch["target_values"].numpy()
        target_list.append(batch_labels)
    targets_np = np.concatenate(target_list, axis=0)
    test_accuracy = np.mean(targets_np == np.argmax(preds_np, axis=1))
    print("test_accuracy : ", test_accuracy)

    output_file = "tspulse_uea_classification_accuracies.csv"

    if not os.path.exists(output_file):
        with open(output_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Dataset", "Accuracy"])

    with open(output_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([dataset_name, test_accuracy])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", required=True, help="List of UEA dataset names")
    args = parser.parse_args()

    print("Datasets : ", args.datasets)

    for dataset_name in args.datasets:
        main(dataset_name)
