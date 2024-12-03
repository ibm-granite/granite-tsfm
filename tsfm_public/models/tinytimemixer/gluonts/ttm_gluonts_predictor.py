# Copyright contributors to the TSFM project
#
"""Tools for building TTM Predictor that works with GluonTS datasets"""

import copy
import math
import os
import tempfile
from typing import List

import numpy as np
import pandas as pd
import torch
from gluonts.dataset.split import InputDataset, LabelDataset, TrainingDataset
from gluonts.itertools import batcher
from gluonts.model.forecast import SampleForecast
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import ConcatDataset, Subset
from tqdm.auto import tqdm
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments
from transformers.integrations import INTEGRATION_TO_CALLBACK
from transformers.utils import logging

from tsfm_public import (
    TrackingCallback,
    count_parameters,
)
from tsfm_public.toolkit.get_model import get_model
from tsfm_public.toolkit.gluonts_data_wrapper import (
    StandardScalingGluonTSDataset,
    TorchDatasetFromGluonTSTestDataset,
    TorchDatasetFromGluonTSTrainingDataset,
    impute_series,
)
from tsfm_public.toolkit.lr_finder import optimal_lr_finder
from tsfm_public.toolkit.visualization import plot_predictions


logger = logging.get_logger(__name__)

# TTM Constants
TTM_MAX_FORECAST_HORIZON = 720
TTM_AVAILABLE_CONTEXTS = [1536, 1024, 512]

# Fewshot max allowed number of samples
# For example, if 5% few-shot for a dataset exceeds this number,
# this `FEWSHOT_MAX_NUM_SAMPLES` upper bound will be used.
FEWSHOT_MAX_NUM_SAMPLES = 500_000


class TTMGluonTSPredictor:
    """Wrapper to TTM that can be directly trained, validated, and tested with GluonTS datasets."""

    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        model_path: str = "ibm-granite/granite-timeseries-ttm-r2",
        test_data_label: LabelDataset = None,  # provide this for plotting
        scale: bool = False,
        random_seed: int = 42,
        term: str = None,
        ds_name: str = None,
        out_dir: str = None,
        plot_test_forecast: bool = False,
        upper_bound_fewshot_samples: bool = True,
        **kwargs,
    ):
        """Initialize a TTMGluonTSPredictor object.

        Args:
            context_length (int): Context length.
            prediction_length (int): Forecast length.
            model_path (str, optional): TTM Model path.. Defaults to "ibm-granite/granite-timeseries-ttm-r2".
            test_data_label (LabelDataset, optional): Test data label object. Only used for plotting. Defaults to None.
            scale (bool, optional): To scale the data or not. Defaults to False. (Recommended to set to `True` for fine-tuning workflow.)
            random_seed (int, optional): Seed. Defaults to 42.
            term (str, optional): Term (short/medium/long). Defaults to None.
            ds_name (str, optional): Dataset name. Only used for plotting. Defaults to None.
            out_dir (str, optional): Out directory. Defaults to None.
            plot_test_forecast (bool, optional): Whether to plot forecasts. Defaults to False.
            upper_bound_fewshot_samples (bool, optional): If True, number of x% fewshot will be upper-bounded
                to FEWSHOT_MAX_NUM_SAMPLES. Defaults to True.
        """
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.test_data_label = test_data_label
        self.scale = scale
        self.scaler = None
        self.random_seed = random_seed
        self.term = term
        self.ds_name = ds_name
        self.out_dir = out_dir
        self.plot_test_forecast = plot_test_forecast
        self.upper_bound_fewshot_samples = upper_bound_fewshot_samples

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if "dropout" in kwargs and kwargs["dropout"] is None:
            del kwargs["dropout"]
        if "head_dropout" in kwargs and kwargs["head_dropout"] is None:
            del kwargs["head_dropout"]

        # Call get_model() function to load TTM model automatically.
        self.ttm = get_model(
            model_path,
            context_length=self.context_length,
            prediction_length=min(self.prediction_length, TTM_MAX_FORECAST_HORIZON),
            **kwargs,
        ).to(self.device)

    def _process_time_series(self, dataset: TrainingDataset) -> List:
        """
        Processes a time series by truncating initial NaNs and forward filling intermittent NaNs.
        Returns a new truncated dataset, and does not modify the original one.

        Args:
            dataset (TrainingDataset): Every series of of shape [channels, length].

        Returns:
            List: Processed time series, each of shape [channels, truncated_length].
        """
        truncated_dataset = list(copy.deepcopy(dataset))

        for i, item in enumerate(truncated_dataset):
            data = item["target"]
            if len(data.shape) == 1:
                data = data.reshape(1, -1)  # [channels, length]

            # Step 1: Determine the longest stretch of initial NaNs across all channels
            valid_mask = ~np.isnan(data)  # Mask of valid (non-NaN) values

            if valid_mask.all():
                continue  # Continue if no NaN

            first_valid = np.argmax(valid_mask.any(axis=0))  # First col with any valid value across channels
            data = data[:, first_valid:]  # Truncate cols before the first valid col

            # Step 2: Perform forward fill for NaNs
            df = pd.DataFrame(data.T, columns=range(data.shape[0]))
            df = df.ffill(axis=0)

            data = df.values.T
            if data.shape[0] == 1:  # [1, truncated_length]
                data = data.reshape(-1)  # [lentruncated_lengthgth]

            truncated_dataset[i]["target"] = data

        return truncated_dataset

    def train(
        self,
        train_dataset: TrainingDataset,
        valid_dataset: TrainingDataset,
        batch_size: int = 64,
        freeze_backbone: bool = True,
        learning_rate: float = None,
        num_epochs: int = 30,
        num_workers: int = 8,
        fewshot_fraction: int = 1.0,
        use_valid_from_train: bool = True,
        save_model: bool = False,
    ):
        """Finetune the TTM.

        Args:
            train_dataset (TrainingDataset): Training dataset.
            valid_dataset (TrainingDataset): Validation dataset.
            batch_size (int, optional): Batch size. Defaults to 64.
            freeze_backbone (bool, optional): To freeze TTM backbone. Defaults to True.
            learning_rate (float, optional): Learning rate. Defaults to None.
            num_epochs (int, optional): Number of epochs. Defaults to 30.
            num_workers (int, optional): Number of workers. Defaults to 8.
            fewshot_fraction (int, optional): Few-shot fraction. Defaults to 1.0.
            use_valid_from_train (bool, optional): Utilize unused train data for validation. Defaults to True.
            save_model (bool, optional): Save model to `self.out_dir`. Defaults to False.

        Raises:
            ValueError: _description_
        """
        train_dataset_scaled = self._process_time_series(train_dataset)
        valid_dataset_scaled = self._process_time_series(valid_dataset)

        # Standard scale
        if self.scale:
            self.scaler = StandardScalingGluonTSDataset()
            self.scaler.fit(train_dataset_scaled)
            train_dataset_scaled = self.scaler.transform(train_dataset_scaled)
            valid_dataset_scaled = self.scaler.transform(valid_dataset_scaled)

        temp_dir = tempfile.mkdtemp()
        dset_train = TorchDatasetFromGluonTSTrainingDataset(
            train_dataset_scaled, self.context_length, self.prediction_length
        )

        dset_valid_from_train = None
        if fewshot_fraction < 1.0:
            # Choose randomly
            rng = np.random.default_rng(seed=self.random_seed)
            if self.upper_bound_fewshot_samples:
                list_size = min(int(fewshot_fraction * len(dset_train)), FEWSHOT_MAX_NUM_SAMPLES)
            else:
                list_size = int(fewshot_fraction * len(dset_train))

            lst_fewshot_indx = rng.integers(
                low=0,
                high=len(dset_train),
                size=list_size,
            )

            logger.info(f"Length of orginal train set = {len(dset_train)}")
            org_dset_train = copy.deepcopy(dset_train)
            dset_train = Subset(org_dset_train, lst_fewshot_indx)
            logger.info(f"Length of {fewshot_fraction*100} % train set = {len(dset_train)}")

            if len(dset_train) < 1:
                raise ValueError(
                    f"Data too small for finetuning in fewshot {fewshot_fraction*100}%. Resulting in 0 samples."
                )

            if use_valid_from_train:
                all_indx = list(range(0, len(org_dset_train)))
                valid_indx = list(set(all_indx) - set(lst_fewshot_indx))

                # we don't use a huge validation set
                valid_size = min(len(dset_train), len(org_dset_train) - len(dset_train))

                valid_indx = np.random.choice(valid_indx, valid_size, replace=False)
                dset_valid_from_train = Subset(org_dset_train, valid_indx)

        dset_valid = TorchDatasetFromGluonTSTrainingDataset(
            valid_dataset_scaled,
            self.context_length,
            self.prediction_length,
            last_window_only=True,
        )

        if dset_valid_from_train is not None:
            dset_valid = ConcatDataset((dset_valid_from_train, dset_valid))

        self.train_num_samples = len(dset_train)
        self.valid_num_samples = len(dset_valid)

        if freeze_backbone:
            print(
                "Number of params before freezing backbone",
                count_parameters(self.ttm),
            )

            # Freeze the backbone of the model
            for param in self.ttm.backbone.parameters():
                param.requires_grad = False

            # Count params
            print(
                "Number of params after freezing the backbone",
                count_parameters(self.ttm),
            )

        # Find optimal learning rate
        # Use with caution: Set it manually if the suggested learning rate is not suitable
        if learning_rate is None:
            learning_rate, self.ttm = optimal_lr_finder(
                self.ttm,
                dset_train,
                batch_size=batch_size,
            )
            print("OPTIMAL SUGGESTED LEARNING RATE =", learning_rate)

        print(f"Using learning rate = {learning_rate}")
        finetune_forecast_args = TrainingArguments(
            output_dir=os.path.join(temp_dir, "output"),
            overwrite_output_dir=True,
            learning_rate=learning_rate,
            num_train_epochs=num_epochs,
            do_eval=True,
            evaluation_strategy="epoch",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            dataloader_num_workers=num_workers,
            report_to="none",
            save_strategy="epoch",
            logging_strategy="epoch",
            save_total_limit=1,
            logging_dir=os.path.join(temp_dir, "logs"),  # Make sure to specify a logging directory
            load_best_model_at_end=True,  # Load the best model when training ends
            metric_for_best_model="eval_loss",  # Metric to monitor for early stopping
            greater_is_better=False,  # For loss
            seed=self.random_seed,
        )

        # Create the early stopping callback
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=5,  # Number of epochs with no improvement after which to stop
            early_stopping_threshold=1e-5,  # Minimum improvement required to consider as improvement
        )
        tracking_callback = TrackingCallback()

        # Optimizer and scheduler
        optimizer = AdamW(self.ttm.parameters(), lr=learning_rate)
        scheduler = OneCycleLR(
            optimizer,
            learning_rate,
            epochs=num_epochs,
            steps_per_epoch=math.ceil(len(dset_train) / (batch_size)),
        )

        hf_trainer = Trainer(
            model=self.ttm,
            args=finetune_forecast_args,
            train_dataset=dset_train,
            eval_dataset=dset_valid,
            callbacks=[early_stopping_callback, tracking_callback],
            optimizers=(optimizer, scheduler),
        )
        hf_trainer.remove_callback(INTEGRATION_TO_CALLBACK["codecarbon"])

        # Fine tune
        hf_trainer.train()

        # Save model
        if save_model:
            hf_trainer.save_model(os.path.join(self.out_dir, "ttm_model"))

    def validate(
        self,
        valid_dataset: TrainingDataset,
        batch_size: int = 64,
    ):
        """(Optionally) Validate.

        Args:
            valid_dataset (TrainingDataset): Validation dataset.
            batch_size (int, optional): Batch size. Defaults to 64.

        Returns:
            flat: Validation loss.
        """
        valid_dataset_scaled = self._process_time_series(valid_dataset)
        if self.scale:
            if self.scaler is None:
                self.scaler = StandardScalingGluonTSDataset()
                self.scaler.fit(valid_dataset_scaled)

            valid_dataset_scaled = self.scaler.transform(valid_dataset_scaled)
        else:
            valid_dataset_scaled = valid_dataset

        temp_dir = tempfile.mkdtemp()
        dset_valid = TorchDatasetFromGluonTSTrainingDataset(
            valid_dataset_scaled,
            self.context_length,
            self.prediction_length,
            last_window_only=True,
        )

        # hf_trainer
        hf_trainer = Trainer(
            model=self.ttm,
            args=TrainingArguments(
                output_dir=temp_dir,
                per_device_eval_batch_size=batch_size,
                seed=self.random_seed,
                report_to="none",
                eval_accumulation_steps=10,
            ),
        )

        # evaluate = zero-shot performance
        print("+" * 20, "Zero-shot Test Loss", "+" * 20)
        zeroshot_output = hf_trainer.predict(dset_valid)
        print(zeroshot_output)
        return zeroshot_output["eval_loss"]

    def predict(
        self,
        test_data_input: InputDataset,
        batch_size: int = 64,
    ):
        """Predict.

        Args:
            test_data_input (InputDataset): Test input dataset.
            batch_size (int, optional): Batch size. Defaults to 64.

        Returns:
            float: Eval loss.
        """
        # Standard scale
        if self.scale:
            # We do not truncate the initial NaNs during testing since it sometimes
            # results in extremely short length, and inference fails.
            # Hence, in the current implementation the initial NaNs will be converted
            # to zeros.
            # (not used currently) test_data_input_scaled = self._process_time_series(test_data_input)

            # A new Standard Scaler is defined
            # Note: Issue with using the train scaler directly...number of series mismatch!
            test_data_input_scaled = copy.deepcopy(test_data_input)
            scaler = StandardScalingGluonTSDataset()
            scaler.fit(test_data_input_scaled)
            test_data_input_scaled = scaler.transform(test_data_input_scaled)
        else:
            test_data_input_scaled = test_data_input

        while True:
            try:
                # Generate forecast samples
                forecast_samples = []
                for batch in tqdm(batcher(test_data_input_scaled, batch_size=batch_size)):
                    batch_ttm = {}
                    adjusted_batch_raw = []
                    for idx, entry in enumerate(batch):
                        # univariate array of shape (time,)
                        # multivariate array of shape (var, time)
                        # TTM supports multivariate time series
                        if len(entry["target"].shape) == 1:
                            entry["target"] = entry["target"].reshape(1, -1)

                        entry_context_length = entry["target"].shape[1]
                        num_channels = entry["target"].shape[0]
                        # Pad
                        if entry_context_length < self.ttm.config.context_length:
                            padding = torch.zeros(
                                (
                                    num_channels,
                                    self.ttm.config.context_length - entry_context_length,
                                )
                            )
                            adjusted_entry = torch.cat((padding, torch.tensor(impute_series(entry["target"]))), dim=1)
                            # observed_mask[idx, :, :(ttm.config.context_length - entry_context_length)] = 0
                        # Truncate
                        elif entry_context_length > self.ttm.config.context_length:
                            adjusted_entry = torch.tensor(
                                impute_series(entry["target"][:, -self.ttm.config.context_length :])
                            )
                        # Take full context
                        else:
                            adjusted_entry = torch.tensor(impute_series(entry["target"]))
                        adjusted_batch_raw.append(adjusted_entry)

                    # For TTM channel dimension comes at the end
                    batch_ttm["past_values"] = torch.stack(adjusted_batch_raw).permute(0, 2, 1).to(self.device)

                    if self.prediction_length > TTM_MAX_FORECAST_HORIZON:
                        recursive_steps = int(np.ceil(self.prediction_length / self.ttm.config.prediction_length))
                        predict_outputs = torch.empty(len(batch), 0, num_channels).to(self.device)
                        with torch.no_grad():
                            for i in range(recursive_steps):
                                model_outputs = self.ttm(**batch_ttm)
                                batch_ttm["past_values"] = torch.cat(
                                    [
                                        batch_ttm["past_values"],
                                        model_outputs["prediction_outputs"],
                                    ],
                                    dim=1,
                                )[:, -self.ttm.config.context_length :, :]
                                predict_outputs = torch.cat(
                                    [
                                        predict_outputs,
                                        model_outputs["prediction_outputs"][:, : self.ttm.config.prediction_length, :],
                                    ],
                                    dim=1,
                                )
                        predict_outputs = predict_outputs[:, : self.prediction_length, :]
                    else:
                        model_outputs = self.ttm(**batch_ttm)
                        predict_outputs = model_outputs.prediction_outputs

                    # Accumulate all forecasts
                    forecast_samples.append(predict_outputs.detach().cpu().numpy())

                # list to np.ndarray
                forecast_samples = np.concatenate(forecast_samples)

                if self.scale:
                    # inverse scale
                    forecast_samples = scaler.inverse_transform(forecast_samples)

                if forecast_samples.shape[2] == 1:
                    forecast_samples = np.squeeze(forecast_samples, axis=2)
                break
            except torch.cuda.OutOfMemoryError:
                print(f"OutOfMemoryError at batch_size {batch_size}, reducing to {batch_size // 2}")
                batch_size //= 2

        # Convert forecast samples into gluonts SampleForecast objects
        #   Array of size (num_samples, prediction_length) (1D case) or
        #   (num_samples, prediction_length, target_dim) (multivariate case)
        sample_forecasts = []
        for item, ts in zip(forecast_samples, test_data_input):
            forecast_start_date = ts["start"] + len(ts["target"])
            sample_forecasts.append(
                SampleForecast(
                    item_id=ts["item_id"],
                    samples=np.expand_dims(item, axis=0),
                    start_date=forecast_start_date,
                )
            )

        if self.out_dir is None:
            self.out_dir = tempfile.mkdtemp()

        if self.plot_test_forecast and self.prediction_length <= TTM_MAX_FORECAST_HORIZON:
            # Create torch dataset for plotting
            torch_dset_test = TorchDatasetFromGluonTSTestDataset(
                gluon_test_input=test_data_input,
                gluon_test_label=self.test_data_label,
                seq_len=self.ttm.config.context_length,
                forecast_len=self.prediction_length,
            )
            # Plot random samples
            plot_predictions(
                dset=torch_dset_test,
                model=self.ttm,
                plot_dir=f"{self.out_dir}/{self.ds_name}_{self.term}",
                channel=0,
                plot_context=int(0.5 * self.prediction_length),
            )

        return sample_forecasts
