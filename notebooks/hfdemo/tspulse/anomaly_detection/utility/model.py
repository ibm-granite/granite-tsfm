# Copyright contributors to the TSFM project
#
from typing import Optional

import sys
import math
import random
import tempfile
import numpy as np
import pandas as pd
from torch.optim import AdamW
from torch.utils.data import random_split
from torch.optim.lr_scheduler import OneCycleLR
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments
from tsfm_public.models.tspulse.utils.helpers import PatchMaskingDatasetWrapper
from tsfm_public.models.tspulse.modeling_tspulse import TSPulseForReconstruction
from tsfm_public.toolkit.time_series_anomaly_detection_pipeline import TimeSeriesAnomalyDetectionPipeline

from .base_model import BaseDetector
from .datasets import TSPulseReconstructionDataset

MODEL_PATH = "ibm-granite/granite-timeseries-tspulse-r1"


def attach_timestamp_column(
    df: pd.DataFrame, time_col: str = "timestamp", freq: str = "5s", start_date: str = "2002-01-01"
):
    n = df.shape[0]
    if time_col not in df:
        df[time_col] = pd.date_range(start_date, freq=freq, periods=n)
    return df


class TSAD_Pipeline(BaseDetector):
    def __init__(
        self,
        model_path: Optional[str] = None,
        batch_size: int = 256,
        aggr_win_size: int = 96,
        num_input_channels: int = 1,
        smoothing_window: int = 8,
        prediction_mode: str = "forecast+time+fft",
        finetune_epochs: int = 20,
        finetune_validation: float = 0.2,
        finetune_lr: float = 1e-4,
        finetune_seed: int = 42,
        finetune_freeze_backbone: bool = False,
        finetune_decoder_mode: str = 'common_channel',
        **kwargs,
    ):
        self._batch_size = batch_size
        self._headers = [f"x{i + 1}" for i in range(num_input_channels)]
        if model_path is None:
            model_path = MODEL_PATH
        
        if num_input_channels == 1:
            finetune_decoder_mode = 'common_channel'
            
        if finetune_decoder_mode is None:
            if num_input_channels > 1:
                self.decoder_mode = "mix_channel"
            else:
                self.decoder_mode = "common_channel"
        else:
            self.decoder_mode = finetune_decoder_mode
        
        random.seed(finetune_seed)
        np.random.seed(finetune_seed)

        self._model = TSPulseForReconstruction.from_pretrained(
            model_path, 
            num_input_channels=num_input_channels, 
            decoder_mode=self.decoder_mode,
            scaling="revin", 
            mask_type="user"
        )
        
        p_length = self._model.config.patch_length
        if (aggr_win_size < p_length) or (aggr_win_size % p_length != 0) :
            raise ValueError(f"Error: aggregation window must be greater than and multiple of patch_length={p_length}")
        prediction_mode_array = [s_.strip() for s_ in str(prediction_mode).split("+")]
        
        self._pipeline_config = dict(timestamp_column="timestamp",
                                     target_columns=self._headers.copy(), 
                                     prediction_mode=prediction_mode_array.copy(),
                                     aggregation_length=aggr_win_size,
                                     smoothing_window=smoothing_window,
                                     least_significant_scale=0.0,
                                     least_significant_score=1.0)
        
        self._finetune_params = dict(finetune_epochs=finetune_epochs,
                                     finetune_validation=finetune_validation,
                                     finetune_lr=finetune_lr,
                                     finetune_seed=finetune_seed,
                                     finetune_freeze_backbone=finetune_freeze_backbone)
        
        self._scorer = TimeSeriesAnomalyDetectionPipeline(
            self._model,
            target_columns=self._pipeline_config.get('target_columns'), 
            prediction_mode=prediction_mode_array,
            aggregation_length=aggr_win_size,
            smoothing_window=self._pipeline_config.get('smoothing_window'),
            least_significant_scale=self._pipeline_config.get('least_significant_scale'),
            least_significant_score=self._pipeline_config.get('least_significant_score')
        )

    def zero_shot(self, x, label=None):
        self.decision_scores_ = self.decision_function(x)

    def fit(self, X, y=None):
        try:
            print("Fine-tuning TSPulse.")
            create_valid = True
            if X.shape[0] < 3000:  # 20% of this should be > context_len
                print("Data too small to create a validation set.")
                create_valid = False
                self.validation_size = 0.0

            if X.shape[0] < self._model.config.context_length:
                print("Skipping fine-tuning due to very short length")
                return

            tsTrain = X[: int((1 - self.validation_size) * len(X))]
            
            train_dataset = PatchMaskingDatasetWrapper(
                TSPulseReconstructionDataset(
                    tsTrain, 
                    window_size=self._pipeline_config.get('aggregation_length'), 
                    return_dict=True
                ),
                window_length=self._pipeline_config.get('aggregation_length'),
                patch_length=self._model.config.patch_length,
                window_position='last'
            )
            if len(train_dataset) < 100:
                print("Skipping fine-tuning due to very few training samples")
                return

            if create_valid:
                tsValid = X[int((1 - self.validation_size) * len(X)) :]
                valid_dataset = PatchMaskingDatasetWrapper(
                    TSPulseReconstructionDataset(
                        tsValid, 
                        window_size=self._pipeline_config.get('aggregation_length'), 
                        return_dict=True
                    ),
                    window_length=self._pipeline_config.get('aggregation_length'),
                    patch_length=self._model.config.patch_length,
                    window_position='last',
                )
            else:
                valid_dataset = train_dataset

            max_finetune_samples = 100_000
            if len(train_dataset) > max_finetune_samples:
                use_fraction = max_finetune_samples / len(train_dataset)
                # Randomly select use_fraction samples to make finetuning faster
                train_dataset, _ = random_split(
                    train_dataset, [use_fraction, 1 - use_fraction]
                )
                valid_dataset, _ = random_split(
                    valid_dataset, [use_fraction, 1 - use_fraction]
                )
                print(
                    f"Training samples are > max_finetune_samples ({max_finetune_samples}), using {round(use_fraction*100)}% for faster fine-tuning."
                )

            freeze_backbone = self._finetune_params.get('finetune_freeze_backbone')
            # Freeze the backbone
            if freeze_backbone:
                # Freeze the backbone of the model
                for param in self._model.backbone.parameters():
                    param.requires_grad = False

            temp_dir = tempfile.mkdtemp()

            suggested_lr = self._finetune_params.get('finetune_lr', 1e-4)
            finetune_num_epochs: int = int(self._finetune_params.get('finetune_epochs', 20))
            if not create_valid:
                finetune_num_epochs = min(5, finetune_num_epochs)

            finetune_batch_size = self._batch_size
            if len(train_dataset) < 500:
                finetune_batch_size = 8
            num_workers = 4
            num_gpus = 1

            print(
                f"Fine-tune: Train samples = {len(train_dataset)}, Valid Samples = {len(valid_dataset)}"
            )

            finetune_args = TrainingArguments(
                output_dir=temp_dir,
                overwrite_output_dir=True,
                learning_rate=suggested_lr,
                num_train_epochs=finetune_num_epochs,
                do_eval=True,
                eval_strategy="epoch",
                per_device_train_batch_size=finetune_batch_size,
                per_device_eval_batch_size=finetune_batch_size * 10,
                dataloader_num_workers=num_workers,
                report_to="tensorboard",
                save_strategy="epoch",
                logging_strategy="epoch",
                save_total_limit=1,
                logging_dir=temp_dir,  # Make sure to specify a logging directory
                load_best_model_at_end=True,  # Load the best model when training ends
                metric_for_best_model="eval_loss",  # Metric to monitor for early stopping
                greater_is_better=False,  # For loss
            )

            # Create the early stopping callback
            early_stopping_callback = EarlyStoppingCallback(
                early_stopping_patience=5,  # Number of epochs with no improvement after which to stop
                early_stopping_threshold=1e-5,  # Minimum improvement required to consider as improvement
            )

            # Optimizer and scheduler
            optimizer = AdamW(self._model.parameters(), 
                              lr=suggested_lr)
            scheduler = OneCycleLR(
                optimizer,
                suggested_lr,
                epochs=finetune_num_epochs,
                steps_per_epoch=math.ceil(
                    len(train_dataset) / (finetune_batch_size * num_gpus)
                ),
            )

            finetune_trainer = Trainer(
                model=self._model,
                args=finetune_args,
                train_dataset=train_dataset,
                eval_dataset=valid_dataset,
                callbacks=[early_stopping_callback],
                optimizers=(optimizer, scheduler),
            )

            # Fine tune
            finetune_trainer.train()

        except Exception as e:
            print("Error occured in finetune. Error =", e)
            sys.exit(-1)


    def decision_function(self, X):
        """
        Not used, present for API consistency by convention.
        """
        data = attach_timestamp_column(pd.DataFrame(X, columns=self._headers))
        score = self._scorer(data, batch_size=self._batch_size)
        if not isinstance(score, pd.DataFrame) or ("anomaly_score" not in score):
            raise ValueError("Error: expect anomaly_score column in the output!")

        score = score["anomaly_score"].values.ravel()
        norm_value = np.nanmax(np.asarray(score), axis=0, keepdims=True) + 1e-5
        anomaly_score = score / norm_value
        return anomaly_score
