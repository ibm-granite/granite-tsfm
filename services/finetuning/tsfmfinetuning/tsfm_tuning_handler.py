"""Tuning handler for TSFM models"""

import copy
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
import torch
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from tsfm_public import ForecastDFDataset
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor
from tsfm_public.toolkit.util import select_by_fixed_fraction

from .filelogging_tracker import FileLoggingCallback
from .ftpayloads import TuneTypeEnum
from .inference_payloads import ForecastingMetadataInput, ForecastingParameters
from .tsfm_config import TSFMConfig
from .tsfm_util import load_config, load_model, register_config


LOGGER = logging.getLogger(__file__)


class TSFMForecastingTuningHandler:
    def __init__(
        self,
        model_id: str,
        model_path: Union[str, Path],
        handler_config: TSFMConfig,
    ):
        self.training_label_names = ["future_values"]

        if (
            getattr(handler_config, "model_type", None)
            and getattr(handler_config, "model_config_name", None)
            and getattr(handler_config, "module_path", None)
        ):
            register_config(
                handler_config.model_type,
                handler_config.model_config_name,
                handler_config.module_path,
            )
            LOGGER.info(f"registered {handler_config.model_type}")

        self.model_id = model_id
        self.model_path = model_path
        self.handler_config = handler_config

        # set during prepare
        self.config = None
        self.model = None
        self.preprocessor = None

    def _get_config_kwargs(
        self,
        parameters: Optional[ForecastingParameters] = None,
        preprocessor: Optional[TimeSeriesPreprocessor] = None,
    ) -> Dict[str, Any]:
        """Helper function to return additional configuration arguments that are used during config load.
        Can be overridden in a subclass to allow specialized model functionality.

        Args:
            parameters (Optional[ForecastingParameters], optional): Request parameters. Defaults to None.
            preprocessor (Optional[TimeSeriesPreprocessor], optional): Time seres preprocessor. Defaults to None.

        Returns:
            Dict[str, Any]: Dictionary of additional arguments that are used later as keyword arguments to the config.
        """
        return {"num_input_channels": preprocessor.num_input_channels}

    def prepare(
        self,
        data: pd.DataFrame,
        future_data: Optional[pd.DataFrame] = None,
        schema: Optional[ForecastingMetadataInput] = None,
        parameters: Optional[ForecastingParameters] = None,
        **kwargs,
    ) -> "TSFMForecastingTuningHandler":
        """Implementation of _prepare for HF-like models. We assume the model will make use of the TSFM
        preprocessor and forecasting pipeline. This method:
        1) loades the preprocessor, creating a new one if the model does not already have a preprocessor
        2) updates model configuration arguments by calling _get_config_kwargs
        3) loads the HuggingFace model, passing the updated config object

        Args:
            data (pd.DataFrame): A pandas dataframe containing historical data.
            future_data (Optional[pd.DataFrame], optional): A pandas dataframe containing future data. Defaults to None.
            schema (Optional[ForecastingMetadataInput], optional): Schema information from the original inference
                request. Includes information about columns and their role. Defaults to None.
            parameters (Optional[ForecastingParameters], optional): Parameters from the original inference
                request. Defaults to None.

        Returns:
            ForecastingHuggingFaceHandler: The updated service handler object.
        """

        preprocessor_params = copy.deepcopy(schema.model_dump())
        preprocessor_params["prediction_length"] = parameters.prediction_length

        LOGGER.info(f"Preprocessor params: {preprocessor_params}")

        # load preprocessor
        try:
            preprocessor = TimeSeriesPreprocessor.from_pretrained(self.model_path)
            LOGGER.info("Successfully loaded preprocessor")
        except OSError:
            preprocessor = None
            LOGGER.info("No preprocessor found")
        except Exception as ex:
            raise ex

        if self.handler_config.is_finetuned and preprocessor is None:
            raise ValueError("Model indicates that it is finetuned but no preprocessor was found.")

        if not self.handler_config.is_finetuned and preprocessor is not None:
            raise ValueError("Unexpected: model indicates that it is not finetuned but a preprocessor was found.")

        if preprocessor is None:
            to_check = ["conditional_columns", "control_columns", "observable_columns", "static_categorical_columns"]

            for param in to_check:
                if param in preprocessor_params and preprocessor_params[param]:
                    raise ValueError(
                        f"Unexpected parameter {param} for a zero-shot model, please confirm you have the correct model_id and schema."
                    )

            preprocessor = TimeSeriesPreprocessor(
                **preprocessor_params,
                scaling=False,
                encode_categorical=False,
            )
            # train to estimate freq
            preprocessor.train(data)
            LOGGER.info(f"Data frequency determined: {preprocessor.freq}")
        else:
            # check payload, but only certain parameters
            to_check = [
                "freq",
                "timestamp_column",
                "target_columns",
                "conditional_columns",
                "control_columns",
                "observable_columns",
            ]

            for param in to_check:
                param_val = preprocessor_params[param]
                param_val_saved = getattr(preprocessor, param)
                if param_val != param_val_saved:
                    raise ValueError(
                        f"Attempted to use a fine-tuned model with a different schema, please confirm you have the correct model_id and schema. Error in parameter {param}: received {param_val} but expected {param_val_saved}."
                    )

        model_config_kwargs = self._get_config_kwargs(
            parameters=parameters,
            preprocessor=preprocessor,
        )
        LOGGER.info(f"model_config_kwargs: {model_config_kwargs}")
        model_config = load_config(self.model_path, **model_config_kwargs)

        model = load_model(
            self.model_path,
            config=model_config,
            module_path=self.handler_config.module_path,
        )

        self.config = model_config
        self.model = model
        self.preprocessor = preprocessor

    def train(
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


class TinyTimeMixerForecastingTuningHandler(TSFMForecastingTuningHandler):
    """Service handler for the tiny time mixer model"""

    def _get_config_kwargs(
        self,
        parameters: Optional[ForecastingParameters] = None,
        preprocessor: Optional[TimeSeriesPreprocessor] = None,
    ) -> Dict[str, Any]:
        config_kwargs = {
            "num_input_channels": preprocessor.num_input_channels,
            "prediction_filter_length": parameters.prediction_length,
            "exogenous_channel_indices": preprocessor.exogenous_channel_indices,
            "prediction_channel_indices": preprocessor.prediction_channel_indices,
        }
        return config_kwargs
