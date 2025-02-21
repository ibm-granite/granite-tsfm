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
from gluonts.model.forecast import QuantileForecast
from scipy.stats import norm
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import ConcatDataset, DataLoader, Subset
from tqdm.auto import tqdm
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments
from transformers.integrations import INTEGRATION_TO_CALLBACK
from transformers.utils import logging

from tsfm_public import (
    TrackingCallback,
    count_parameters,
)
from tsfm_public.toolkit.get_model import TTM_LOW_RESOLUTION_MODELS_MAX_CONTEXT, get_model
from tsfm_public.toolkit.lr_finder import optimal_lr_finder

from .gluonts_data_wrapper import (
    RESOLUTION_MAP,
    TTM_MAX_FORECAST_HORIZON,
    ForecastDataset,
    StandardScalingGluonTSDataset,
    TorchDatasetFromGluonTSTrainingDataset,
    get_freq_mapping,
    impute_series,
)
from .utils import CustomMASETrainer, plot_forecast


logger = logging.get_logger(__name__)

# TTM Constants:
# Fewshot max allowed number of samples
# This is only used when `upper_bound_fewshot_samples=True`.
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
        upper_bound_fewshot_samples: bool = False,
        force_short_context: bool = False,
        min_context_mult: int = 4,
        past_feat_dynamic_real_exist: bool = False,
        num_prediction_channels: int = None,
        use_mask: bool = False,
        freq: str = None,
        use_valid_from_train: bool = True,
        insample_forecast: bool = True,
        insample_use_train: bool = False,
        plot_predictions: bool = False,
        **kwargs,
    ):
        """Initialize a TTMGluonTSPredictor object.

        Args:
            context_length (int): Context length / history length.
            prediction_length (int): Prediction length / forecast horizon.
            model_path (str, optional): Model path / model card link. Defaults to "ibm-granite/granite-timeseries-ttm-r2".
            test_data_label (LabelDataset, optional): Test data label object. Only used for plotting. Defaults to None.
            random_seed (int, optional): Random seed. Defaults to 42.
            term (str, optional): Term (short/medium/long). Defaults to None.
            ds_name (str, optional): Dataset name. Defaults to None.
            out_dir (str, optional): Output directory. Defaults to None.
            upper_bound_fewshot_samples (bool, optional): If True, number of x% fewshot will be upper-bounded to
                FEWSHOT_MAX_NUM_SAMPLES. Defaults to False (recommended).
            force_short_context (bool, optional): If True, it forces to use short context. Defaults to False (recommended).
            min_context_mult (int, optional): If set to n, minimum context of length `prediction_length x n` is needed
                to create a finetune sample. Defaults to 4.
            past_feat_dynamic_real_exist (bool, optional): Use past exogeneous features. Defaults to False.
            num_prediction_channels (int, optional): Number of prediction channels. Defaults to None.
            use_mask (bool, optional): Use observed mask. Defaults to False.
            freq (str, optional): Frequency. Defaults to None.
            use_valid_from_train (bool, optional): Use unused training data for validation (from 100-x% fewshot).
                Defaults to True.
            insample_forecast (bool, optional): Calculate insample statistics for quantile forecasts. Defaults to True.
            insample_use_train (bool, optional): Use training error statistics along with validation error statistics
                for insample statistics calculation. Defaults to False.
            plot_predictions (bool, optional): Plot predictions or not. Defaults to False.
        """
        self.prediction_length = prediction_length
        self.test_data_label = test_data_label
        self.scale = scale
        self.scaler = None
        self.random_seed = random_seed
        self.term = term
        self.ds_name = ds_name
        self.out_dir = out_dir
        self.upper_bound_fewshot_samples = upper_bound_fewshot_samples
        self.force_short_context = force_short_context
        self.min_context_mult = min_context_mult
        self.past_feat_dynamic_real_exist = past_feat_dynamic_real_exist
        self.num_prediction_channels = num_prediction_channels
        self.freq = freq
        self.use_mask = use_mask
        self.use_valid_from_train = use_valid_from_train
        self.insample_forecast = insample_forecast
        self.insample_use_train = insample_use_train
        self.plot_predictions = plot_predictions
        self.quantile_keys = [
            "0.1",
            "0.2",
            "0.3",
            "0.4",
            "0.5",
            "0.6",
            "0.7",
            "0.8",
            "0.9",
            "mean",
        ]

        self.insample_errors = None

        if force_short_context:
            logger.info(f"Forcing short context: H = {prediction_length}, CL={prediction_length*min_context_mult}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if "dropout" in kwargs and kwargs["dropout"] is None:
            del kwargs["dropout"]
        if "head_dropout" in kwargs and kwargs["head_dropout"] is None:
            del kwargs["head_dropout"]
        if "loss" in kwargs and kwargs["loss"] is not None:
            self.loss = kwargs["loss"]
            if kwargs["loss"] not in ["mse", "mae", "huber", "pinball"]:
                del kwargs["loss"]
        if "prediction_channel_indices" in kwargs:
            self.prediction_channel_indices = kwargs["prediction_channel_indices"]

        # Get model
        self._get_gift_model(model_path, context_length, prediction_length, freq, **kwargs)

    def _get_gift_model(self, model_path: str, context_length: int, prediction_length: int, freq: str, **kwargs):
        """Get suitable TTM model based on context and forecast lengths.

        Args:
            model_path (str): Model card link.
            context_length (int): Context length.
        """
        self.ttm = None

        prefer_l1_loss = False
        prefer_longer_context = True
        freq_prefix_tuning = False
        force_return = "zeropad"
        if self.term == "short" and (
            str(self.freq).startswith("W")
            or str(self.freq).startswith("M")
            or str(self.freq).startswith("Q")
            or str(self.freq).startswith("A")
        ):
            prefer_l1_loss = True
            prefer_longer_context = False
            freq_prefix_tuning = True

        if self.term == "short" and str(self.freq).startswith("D"):
            prefer_l1_loss = True
            freq_prefix_tuning = True
            if context_length < 2 * TTM_LOW_RESOLUTION_MODELS_MAX_CONTEXT:
                prefer_longer_context = False
            else:
                prefer_longer_context = True

        if self.term == "short" and str(self.freq).startswith("A"):
            self.insample_use_train = False
            self.use_valid_from_train = False
            force_return = "random_init_small"

        if prediction_length > TTM_MAX_FORECAST_HORIZON:
            force_return = "rolling"

        self.ttm = get_model(
            model_path=model_path,
            context_length=context_length,
            prediction_length=prediction_length,
            prefer_l1_loss=prefer_l1_loss,
            prefer_longer_context=prefer_longer_context,
            resolution=RESOLUTION_MAP.get(freq, "oov"),
            freq_prefix_tuning=freq_prefix_tuning,
            force_return=force_return,
            **kwargs,
        ).to(self.device)

        self.context_length = self.ttm.config.context_length

        self.enable_prefix_tuning = False
        if hasattr(self.ttm.config, "resolution_prefix_tuning"):
            self.enable_prefix_tuning = self.ttm.config.resolution_prefix_tuning
        logger.info(f"The TTM has Prefix Tuning = {self.enable_prefix_tuning}")

    def _process_time_series(self, dataset: TrainingDataset, truncate: bool = True) -> List:
        """
        Processes a time series by truncating initial NaNs and forward filling intermittent NaNs.
        Returns a new truncated dataset, and does not modify the original one.

        Args:
            dataset (TrainingDataset): Every series of of shape [channels, length].
            truncate (bool, optional): Truncate the dataset if True. Defaults to True.

        Returns:
            List: Processed time series, each of shape [channels, truncated_length].
        """
        truncated_dataset = list(copy.deepcopy(dataset))
        for i, item in enumerate(truncated_dataset):
            data = item["target"]

            if data.ndim == 1:
                data = data.reshape(1, -1)  # [channels, length]

            if self.past_feat_dynamic_real_exist:
                if item["past_feat_dynamic_real"].ndim == 1:
                    item["past_feat_dynamic_real"] = item["past_feat_dynamic_real"].reshape(1, -1)
                data = np.vstack((data, item["past_feat_dynamic_real"]))

            truncated_dataset[i]["target"] = data

            if not truncate:
                continue

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
                data = data.reshape(-1)  # [truncated_length]

            truncated_dataset[i]["target"] = data

        return truncated_dataset

    def compute_quantile_forecasts(self, loader, quantiles):
        all_quantile_forecasts = []

        for batch in tqdm(loader, desc="Processing Batches"):
            forecast_samples, insample_errors, point_forecasts = batch

            insample_errors[insample_errors == 0] = 1e-5  # To prevent division by zero

            # Expand scales for quantiles
            batch_size, seq_len, no_channels = forecast_samples.shape
            num_quantiles = len(quantiles)

            scales = np.expand_dims(insample_errors, axis=1)  # Shape: (batch_size, 1, H, C)
            scales = np.tile(scales, (1, num_quantiles, 1, 1))  # Shape: (batch_size, num_quantiles, H, C)

            # Expand quantiles
            quantiles_expanded = np.reshape(quantiles, (1, num_quantiles, 1, 1))  # Shape: (1, num_quantiles, 1, 1)
            quantiles_expanded = np.tile(
                quantiles_expanded, (batch_size, 1, seq_len, no_channels)
            )  # Shape: (batch_size, num_quantiles, H, C)

            # Expand forecasts
            forecasts_expanded = np.expand_dims(forecast_samples, axis=1)  # Shape: (batch_size, 1, H, C)
            forecasts_expanded = np.tile(
                forecasts_expanded, (1, num_quantiles, 1, 1)
            )  # Shape: (batch_size, num_quantiles, H, C)

            # Compute quantile forecasts
            quantile_forecasts = norm.ppf(
                quantiles_expanded, loc=forecasts_expanded, scale=scales
            )  # Shape: (batch_size, num_quantiles, H, C)

            # Append point forecasts
            final_forecasts = np.concatenate(
                (quantile_forecasts, point_forecasts), axis=1
            )  # Shape: (batch_size, num_quantiles+1, H, C)

            # Collect results for the batch
            all_quantile_forecasts.extend(final_forecasts)

        return all_quantile_forecasts

    def train(
        self,
        train_dataset: TrainingDataset,
        valid_dataset: TrainingDataset,
        batch_size: int = 64,
        optimize_batch_size: bool = True,
        freeze_backbone: bool = False,
        learning_rate: float = None,
        num_epochs: int = 30,
        num_workers: int = 8,
        fewshot_fraction: int = 1.0,
        automate_fewshot_fraction: bool = True,
        automate_fewshot_fraction_threshold: int = 200,
        fewshot_location: str = "rand",  # rand/start/end
        save_model: bool = False,
    ):
        """Finetune the TTM.

        Args:
            train_dataset (TrainingDataset): Training dataset.
            valid_dataset (TrainingDataset): Validation dataset.
            batch_size (int, optional): Batch size. Defaults to 64.
            optimize_batch_size (bool, optional): Optimize batch size based on data size to speed up training.
                Defaults to True.
            freeze_backbone (bool, optional): To freeze TTM backbone. Defaults to False.
            learning_rate (float, optional): Learning rate. Defaults to None.
            num_epochs (int, optional): Number of epochs. Defaults to 30.
            num_workers (int, optional): Number of workers. Defaults to 8.
            fewshot_fraction (int, optional): Few-shot fraction. Defaults to 1.0.
            automate_fewshot_fraction (bool, optional): Automate few-shot fraction. Helps for very small dataset.
                Defaults to True.
            automate_fewshot_fraction_threshold (int, optional): Consider this many samples as very small.
            fewshot_location (str, optional): Fewshot location "rand"/"start"/"end". Default to "rand" (recommended).
            save_model (bool, optional): Save model to `self.out_dir`. Defaults to False.

        Raises:
            ValueError: _description_
        """
        train_dataset_scaled = self._process_time_series(train_dataset)
        valid_dataset_scaled = self._process_time_series(valid_dataset)
        logger.info(f"Number of series: Train = {len(train_dataset_scaled)}, Valid = {len(valid_dataset_scaled)}")

        # Standard scale
        if self.scale:
            self.scaler = StandardScalingGluonTSDataset()
            self.scaler.fit(train_dataset_scaled)
            train_dataset_scaled = self.scaler.transform(train_dataset_scaled)
            valid_dataset_scaled = self.scaler.transform(valid_dataset_scaled)
            logger.info("Global scaling done successfully.")

        temp_dir = tempfile.mkdtemp()

        # create train dataset
        dset_train_actual = TorchDatasetFromGluonTSTrainingDataset(
            train_dataset_scaled,
            self.context_length,
            self.prediction_length,
            force_short_context=self.force_short_context,
            min_context_mult=self.min_context_mult,
            send_freq=self.enable_prefix_tuning,
            freq=self.freq,
        )

        if automate_fewshot_fraction:
            fewshot_data_size = int(fewshot_fraction * len(dset_train_actual))
            if fewshot_data_size < automate_fewshot_fraction_threshold and fewshot_data_size > 10:
                fewshot_fraction = 0.9
                num_epochs = 50  # easy to run large epochs since datasets are small
                logger.info(f"Increasing fewshot fraction to {fewshot_fraction} due to small dataset size.")

        dset_valid_from_train = None
        if fewshot_fraction < 1.0:
            if fewshot_location == "rand":
                dset_train = copy.deepcopy(dset_train_actual)

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

                dset_train = Subset(dset_train_actual, lst_fewshot_indx)
                logger.info(f"Length of {fewshot_fraction*100} % train set = {len(dset_train)}")

                if len(dset_train) < 1:
                    raise ValueError(
                        f"Data too small for finetuning in fewshot {fewshot_fraction*100}%. Resulting in 0 samples."
                    )

                if self.use_valid_from_train:
                    all_indx = list(range(0, len(dset_train_actual)))
                    valid_indx = list(set(all_indx) - set(lst_fewshot_indx))

                    valid_size = min(50_000, len(dset_train_actual) - len(dset_train))

                    valid_indx = np.random.choice(valid_indx, valid_size, replace=False)
                    dset_valid_from_train = Subset(dset_train_actual, valid_indx)
            elif fewshot_location == "end" or fewshot_location == "start":
                # re-define train with fewshot calculated
                dset_train = TorchDatasetFromGluonTSTrainingDataset(
                    train_dataset_scaled,
                    self.context_length,
                    self.prediction_length,
                    fewshot_fraction=fewshot_fraction,
                    fewshot_location=fewshot_location,
                    force_short_context=self.force_short_context,
                    min_context_mult=self.min_context_mult,
                    send_freq=self.enable_prefix_tuning,
                    freq=self.freq,
                )
            else:
                raise ValueError("Wrong fewshot_location.")
        else:
            logger.info("Using 100% train data to finetune the model.")
            # create train dataset
            dset_train = TorchDatasetFromGluonTSTrainingDataset(
                train_dataset_scaled,
                self.context_length,
                self.prediction_length,
                force_short_context=self.force_short_context,
                min_context_mult=self.min_context_mult,
                send_freq=self.enable_prefix_tuning,
                freq=self.freq,
            )

        dset_valid = TorchDatasetFromGluonTSTrainingDataset(
            valid_dataset_scaled,
            self.context_length,
            self.prediction_length,
            last_window_only=True,
            gen_more_samples_for_short_series=False,
            force_short_context=self.force_short_context,
            min_context_mult=self.min_context_mult,
            send_freq=self.enable_prefix_tuning,
            freq=self.freq,
        )
        dset_valid_actual = copy.deepcopy(dset_valid)

        if dset_valid_from_train is not None:
            # Use valid for training since it is more recent data,
            # and use random valid chosen earlier
            dset_train = ConcatDataset((dset_train, dset_valid))
            dset_valid = dset_valid_from_train

        self.train_num_samples = len(dset_train)
        self.valid_num_samples = len(dset_valid)
        logger.info(f"Number of train samples = {self.train_num_samples}, valid samples = {self.valid_num_samples}")

        if freeze_backbone:
            logger.info(
                f"Number of params before freezing backbone = {count_parameters(self.ttm)}",
            )

            # Freeze the backbone of the model
            for param in self.ttm.backbone.parameters():
                param.requires_grad = False

            # Count params
            logger.info(
                f"Number of params after freezing the backbone = {count_parameters(self.ttm)}",
            )

        if optimize_batch_size:
            # Set custom batch size to speed up the training
            # for large data and for better convergence for small data
            if self.ttm.config.num_input_channels < 10:
                batch_size = 64
            else:
                batch_size = 16

            if len(dset_train) <= 1_000:
                batch_size = 8
            elif len(dset_train) > 100_000:
                batch_size = 512

            logger.info(
                f"Using a batch size of {batch_size}, based on number of training samples = {len(dset_train)} and number of channels = {self.ttm.config.num_input_channels}."
            )

        # Find optimal learning rate
        # Use with caution: Set it manually if the suggested learning rate is not suitable
        if learning_rate is None:
            learning_rate, self.ttm = optimal_lr_finder(
                self.ttm,
                dset_train,
                batch_size=batch_size,
                enable_prefix_tuning=self.enable_prefix_tuning,
            )
            logger.info(f"OPTIMAL SUGGESTED LEARNING RATE = {learning_rate}")

        logger.info(f"Using learning rate = {learning_rate}")

        finetune_forecast_args = TrainingArguments(
            output_dir=os.path.join(temp_dir, "output"),
            overwrite_output_dir=True,
            learning_rate=learning_rate,
            num_train_epochs=num_epochs,
            do_eval=True,
            eval_strategy="epoch",
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
            data_seed=self.random_seed,
            eval_accumulation_steps=10,
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

        if self.loss == "mase":
            underlying_dataset = train_dataset.dataset.iterable
            first_entry = next(iter(underlying_dataset))
            freq_ = first_entry["freq"]
            hf_trainer = CustomMASETrainer(
                model=self.ttm,
                args=finetune_forecast_args,
                train_dataset=dset_train,
                eval_dataset=dset_valid,
                callbacks=[early_stopping_callback, tracking_callback],
                optimizers=(optimizer, scheduler),
                freq=freq_,
            )
        else:
            hf_trainer = Trainer(
                model=self.ttm,
                args=finetune_forecast_args,
                train_dataset=dset_train,
                eval_dataset=dset_valid,
                callbacks=[early_stopping_callback, tracking_callback],
                optimizers=(optimizer, scheduler),
            )
        hf_trainer.remove_callback(INTEGRATION_TO_CALLBACK["codecarbon"])

        # Force zeroshot if very few finetune samples
        if self.train_num_samples > 10:
            # Finetune
            hf_trainer.train()

        # Save model
        if save_model:
            if self.out_dir is not None:
                hf_trainer.save_model(os.path.join(self.out_dir, "ttm_model"))
            else:
                raise ValueError("`out_dir` should not be `None` when `save_model=True`.")
        if self.insample_forecast:
            self.insample_errors = self.get_insample_stats(
                hf_trainer=hf_trainer,
                dset_valid=dset_valid_actual,
                dset_train=dset_train_actual,
                global_scaler=self.scaler,
                use_train=self.insample_use_train,
            )

        if self.train_num_samples <= 10:
            raise Exception("Forcing zeroshot since number of finetune samples is very low.")

    def get_insample_stats(
        self,
        hf_trainer,
        dset_valid,
        dset_train,
        global_scaler,
        batch_size=128,
        use_train=False,
    ):
        if len(dset_valid) == 1:
            dataset = ConcatDataset((dset_train, dset_valid))
        elif use_train:
            max_subset_size = 50_000
            if len(dset_train) > max_subset_size:
                rng = np.random.default_rng(seed=self.random_seed)
                lst_fewshot_indx = rng.integers(
                    low=0,
                    high=len(dset_train),
                    size=max_subset_size,
                )
                dset_train_subset = Subset(dset_train, lst_fewshot_indx)
                dataset = ConcatDataset((dset_train_subset, dset_valid))
            else:
                dataset = ConcatDataset((dset_train, dset_valid))
        else:
            dataset = dset_valid

        # Get ground truth stacked
        dl_valid = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        y_true = []
        series_ids = []
        for batch in dl_valid:
            y_true.append(batch["future_values"])
            series_ids.extend(batch["item_id"])
        y_true = torch.cat(y_true).detach().cpu().numpy()
        y_true_unscaled = global_scaler.inverse_transform(y_true, series_ids)

        # Get validation predictions
        valid_preds_out = hf_trainer.predict(dataset)
        y_pred = valid_preds_out.predictions[0]
        y_pred_unscaled = global_scaler.inverse_transform(y_pred, series_ids)

        # Create a pands dataframe
        # Flatten (H, C) into 2D arrays
        # L = y_pred_unscaled.shape[0]
        flattened_predictions = list(y_pred_unscaled)
        flattened_ground_truth = list(y_true_unscaled)

        df = pd.DataFrame(
            {
                "item_id": series_ids,
                "y_true": flattened_ground_truth,
                "y_pred": flattened_predictions,
            }
        )

        df["errors"] = (df["y_true"] - df["y_pred"]).abs()  # absolute error pointwise
        errors = df.groupby(by="item_id")["errors"].mean()  # mean over all samples from a particular series

        logger.info("Successfully, calculated the in-sample statistics.")

        return errors

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
            force_short_context=self.force_short_context,
            min_context_mult=self.min_context_mult,
            send_freq=self.enable_prefix_tuning,
            freq=self.freq,
        )

        # hf_trainer
        hf_trainer = Trainer(
            model=self.ttm,
            args=TrainingArguments(
                output_dir=temp_dir,
                per_device_eval_batch_size=batch_size,
                report_to="none",
                eval_accumulation_steps=10,
                seed=self.random_seed,
                data_seed=self.random_seed,
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
        # We do not truncate the initial NaNs during testing since it sometimes
        # results in extremely short length, and inference fails.
        # Hence, in the current implementation the initial NaNs will be converted to zeros.
        test_data_input_scaled = self._process_time_series(test_data_input, truncate=False)
        # Standard scale
        if self.scale:
            test_data_input_scaled = self.scaler.transform(test_data_input_scaled)

        while True:
            try:
                # Generate forecast samples
                forecast_samples = []
                series_ids = []
                for batch in tqdm(batcher(test_data_input_scaled, batch_size=batch_size)):
                    batch_ttm = {}
                    adjusted_batch_raw = []
                    past_observed_mask = []
                    for idx, entry in enumerate(batch):
                        series_ids.append(entry["item_id"])

                        # univariate array of shape (time,)
                        # multivariate array of shape (var, time)
                        # TTM supports multivariate time series
                        if len(entry["target"].shape) == 1:
                            entry["target"] = entry["target"].reshape(1, -1)

                        if self.force_short_context:
                            entry["target"] = entry["target"][:, -self.min_context_mult * self.prediction_length :]

                        entry_context_length = entry["target"].shape[1]
                        num_channels = entry["target"].shape[0]

                        # Pad
                        if entry_context_length < self.ttm.config.context_length:
                            logger.debug("Using zero filling for padding.")
                            # Zero-padding
                            padding = torch.zeros(
                                (
                                    num_channels,
                                    self.ttm.config.context_length - entry_context_length,
                                )
                            )
                            adjusted_entry = torch.cat(
                                (
                                    padding,
                                    torch.tensor(impute_series(entry["target"])),
                                ),
                                dim=1,
                            )
                            mask = torch.ones(adjusted_entry.shape)
                            mask[:, : padding.shape[1]] = 0

                            # observed_mask[idx, :, :(ttm.config.context_length - entry_context_length)] = 0
                        # Truncate
                        elif entry_context_length > self.ttm.config.context_length:
                            adjusted_entry = torch.tensor(
                                impute_series(entry["target"][:, -self.ttm.config.context_length :])
                            )
                            mask = torch.ones(adjusted_entry.shape)
                        # Take full context
                        else:
                            adjusted_entry = torch.tensor(impute_series(entry["target"]))
                            mask = torch.ones(adjusted_entry.shape)

                        adjusted_batch_raw.append(adjusted_entry)
                        past_observed_mask.append(mask.bool())

                    # For TTM channel dimension comes at the end
                    batch_ttm["past_values"] = torch.stack(adjusted_batch_raw).permute(0, 2, 1).to(self.device)
                    if self.use_mask:
                        batch_ttm["past_observed_mask"] = (
                            torch.stack(past_observed_mask).permute(0, 2, 1).to(self.device)
                        )
                    if self.ttm.config.resolution_prefix_tuning:
                        freq_map = get_freq_mapping()
                        batch_ttm["freq_token"] = (
                            torch.ones((batch_ttm["past_values"].shape[0])) * freq_map[self.freq]
                        ).to("cuda")

                    if self.prediction_length > TTM_MAX_FORECAST_HORIZON:
                        batch_ttm["return_loss"] = False

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
                                if self.use_mask:
                                    batch_ttm["past_observed_mask"] = torch.cat(
                                        [
                                            batch_ttm["past_observed_mask"],
                                            torch.ones(model_outputs["prediction_outputs"].shape)
                                            .bool()
                                            .to(self.device),
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
                    if self.past_feat_dynamic_real_exist:
                        forecast_samples = self.scaler.inverse_transform(
                            forecast_samples,
                            series_ids,
                            self.prediction_channel_indices,
                        )
                    else:
                        forecast_samples = self.scaler.inverse_transform(forecast_samples, series_ids)

                if self.prediction_length > TTM_MAX_FORECAST_HORIZON:
                    forecast_samples = forecast_samples[:, :, : self.num_prediction_channels]

                if self.insample_forecast:
                    point_forecasts = np.expand_dims(forecast_samples, 1)

                    self.quantiles = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

                    # Assuming forecasts, scale, and self.quantiles are defined
                    b, seq_len, no_channels = forecast_samples.shape

                    if self.insample_errors is None:
                        dummy_errors_ = []
                        unq_series_ids = list(np.unique(series_ids))
                        for _ in unq_series_ids:
                            dummy_errors_.append(np.ones((seq_len, no_channels)))
                        self.insample_errors = pd.DataFrame(
                            {"item_id": unq_series_ids, "errors": dummy_errors_}
                        ).set_index("item_id")["errors"]
                        logger.warning("`insample_errors` is `None`. Using a dummy error of `np.ones()`")

                    # happens for H > 720
                    if self.insample_errors.iloc[0].shape[0] < self.prediction_length:
                        for i in range(len(self.insample_errors)):
                            self.insample_errors.iloc[i] = np.concatenate(
                                (
                                    self.insample_errors.iloc[i],
                                    self.insample_errors.iloc[i][
                                        -(self.prediction_length - self.insample_errors.iloc[i].shape[0]) :,
                                        :,
                                    ],
                                )
                            )

                    logger.info(f"Making quantile forecasts for quantiles {self.quantiles}")
                    all_quantile_forecasts = []

                    dataset = ForecastDataset(
                        forecast_samples,
                        series_ids,
                        self.insample_errors,
                        point_forecasts,
                        self.quantiles,
                    )
                    dataloader = DataLoader(
                        dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        collate_fn=lambda x: (
                            np.stack([i[0] for i in x]),  # forecast_samples
                            np.stack([i[1] for i in x]),  # insample_errors
                            np.stack([i[2] for i in x]),  # point_forecasts
                        ),
                    )

                    all_quantile_forecasts = self.compute_quantile_forecasts(dataloader, self.quantiles)

                forecast_samples = np.array(all_quantile_forecasts)
                if forecast_samples.shape[-1] == 1:
                    forecast_samples = np.squeeze(forecast_samples, axis=-1)
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
                QuantileForecast(
                    forecast_arrays=item,
                    start_date=forecast_start_date,
                    forecast_keys=self.quantile_keys,
                    item_id=ts["item_id"],
                )
            )

        if self.plot_predictions:
            # plot random samples
            plot_forecast(
                test_data_input,
                self.test_data_label,
                forecast_samples,
                self.prediction_length,
                self.ds_name,
                self.term,
                self.out_dir,
                probabilistic=self.insample_forecast,
                quantile_keys=self.quantile_keys,
            )

        return sample_forecasts
