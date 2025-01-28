"""Service handler for HuggingFace models"""

import logging
import os
from pathlib import Path

import pandas as pd
import torch
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from tsfm_public import ForecastDFDataset
from tsfm_public.toolkit.util import select_by_fixed_fraction

from .filelogging_tracker import FileLoggingCallback
from .ftpayloads import TuneTypeEnum
from .hf_service_handler import ForecastingHuggingFaceHandler, TinyTimeMixerForecastingHandler
from .inference_payloads import ForecastingMetadataInput, ForecastingParameters
from .tuning_handler import (
    ForecastingTuningHandler,
)


LOGGER = logging.getLogger(__file__)


class ForecastingHuggingFaceTuningHandler(ForecastingHuggingFaceHandler, ForecastingTuningHandler):
    def _train(
        self,
        data: pd.DataFrame,
        schema: ForecastingMetadataInput,
        parameters: ForecastingParameters,
        tuned_model_name: str,
        tmp_dir: Path,
    ) -> str:
        # to do: data selection here
        train_data = data
        validation_data = None

        # optional freezing
        if parameters.tune_type == TuneTypeEnum.linear_probe.value:
            # Freeze the backbone of the model
            for param in self.model.backbone.parameters():
                param.requires_grad = False

        # create datasets
        dataset_spec = {
            "id_columns": self.preprocessor.id_columns,
            "timestamp_column": self.preprocessor.timestamp_column,
            "target_columns": self.preprocessor.target_columns,
            "observable_columns": self.preprocessor.observable_columns,
            "control_columns": self.preprocessor.control_columns,
            "conditional_columns": self.preprocessor.conditional_columns,
            "static_categorical_columns": self.preprocessor.static_categorical_columns,
            "prediction_length": self.model.config.prediction_length,
            "context_length": self.model.config.context_length,
        }
        # to do: check prediction length. Three options (1) prediction length is passed, (2) prediction length is in preprocessor, (3) prediction length is in model config
        # we should properly set the prediction/context lengths in the preprocessor
        # we also need to make sure prediction_filter length is set in the model config

        if parameters.fewshot_fraction < 1:
            train_data = select_by_fixed_fraction(
                train_data,
                id_columns=self.preprocessor.id_columns,
                fraction=parameters.fewshot_fraction,
                location="last",  # to do: expose this parameter
                minimum_size=self.preprocessor.context_length,
            )

        train_dataset = ForecastDFDataset(train_data, **dataset_spec)
        validation_dataset = ForecastDFDataset(validation_data, **dataset_spec) if validation_data else train_dataset

        # Configure trainer
        parameters = parameters
        training_tmp_dir = Path(tmp_dir)
        training_args = TrainingArguments(
            output_dir=training_tmp_dir / "output",
            overwrite_output_dir=True,
            learning_rate=parameters.trainer_args.learning_rate,
            num_train_epochs=parameters.trainer_args.num_train_epochs,
            eval_strategy="epoch",
            per_device_train_batch_size=parameters.trainer_args.per_device_train_batch_size,
            per_device_eval_batch_size=parameters.trainer_args.per_device_eval_batch_size,
            dataloader_num_workers=4,  # to do: auto config this parameter
            save_strategy="epoch",
            logging_strategy="epoch",
            save_total_limit=3,
            logging_dir=training_tmp_dir / "logs",  # Make sure to specify a logging directory
            load_best_model_at_end=True,  # Load the best model when training ends
            metric_for_best_model="eval_loss",  # Metric to monitor for early stopping
            greater_is_better=False,  # For loss
            label_names=self.training_label_names,
            use_cpu=not torch.cuda.is_available(),  # is MPS possible now?
        )

        callbacks = [
            FileLoggingCallback(logs_filename=os.environ.get("TSFM_TRAINING_TRACKER_LOGFILE", "training_logs.jsonl"))
        ]
        if parameters.trainer_args.early_stopping and validation_dataset:
            # Create the early stopping callback
            early_stopping_callback = EarlyStoppingCallback(
                early_stopping_patience=parameters.trainer_args.early_stopping_patience,  # Number of epochs with no improvement after which to stop
                early_stopping_threshold=parameters.trainer_args.early_stopping_threshold,  # Minimum improvement required to consider as improvement
            )
            callbacks.append(early_stopping_callback)
        trainer_extra_args = {}
        if validation_dataset:
            trainer_extra_args = {"eval_dataset": validation_dataset}

        # define trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            callbacks=callbacks,
            **trainer_extra_args,
        )

        # To do: do we provide feedback to the user?
        LOGGER.info("calling trainer.train")
        trainer.train()
        LOGGER.info("done with training")

        save_path = training_tmp_dir / tuned_model_name
        trainer.save_model(save_path)
        self.preprocessor.save_pretrained(save_path)
        return save_path


class TinyTimeMixerForecastingTuningeHandler(TinyTimeMixerForecastingHandler, ForecastingHuggingFaceTuningHandler): ...
