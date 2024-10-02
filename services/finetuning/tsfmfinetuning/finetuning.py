# Copyright contributors to the TSFM project
#
"""Tsfmfinetuning Runtime"""

import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from fastapi import APIRouter, HTTPException
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed

from tsfm_public import TimeSeriesPreprocessor
from tsfm_public.toolkit.dataset import ForecastDFDataset
from tsfm_public.toolkit.util import select_by_fixed_fraction

from . import TSFM_ALLOW_LOAD_FROM_HF_HUB
from .constants import API_VERSION
from .ftpayloads import (
    AsyncCallReturn,
    BaseTuneInput,
    TinyTimeMixerForecastingTuneInput,
    TuneTypeEnum,
)
from .hfutil import load_config, load_model, register_config
from .ioutils import to_pandas


LOGGER = logging.getLogger(__file__)


class FinetuningRuntime:
    def __init__(self, config: Dict[str, Any] = {}):
        self.config = config
        model_map = {}

        if "custom_modules" in config:
            for custom_module in config["custom_modules"]:
                register_config(
                    custom_module["model_type"],
                    custom_module["model_config_name"],
                    custom_module["module_path"],
                )
                LOGGER.info(f"registered {custom_module['model_type']}")

                model_map[custom_module["model_config_name"]] = custom_module["module_path"]

        self.model_to_module_map = model_map

    def add_routes(self, app):
        self.router = APIRouter(prefix=f"/{API_VERSION}/finetuning", tags=["finetuning"])
        self.router.add_api_route(
            "/tinytimemixer/forecasting",
            self.finetuning,
            methods=["POST"],
            response_model=AsyncCallReturn,
        )
        app.include_router(self.router)

    def load(self, model_path: str):
        try:
            preprocessor = TimeSeriesPreprocessor.from_pretrained(model_path)
            LOGGER.info("Successfully loaded preprocessor")
        except OSError:
            preprocessor = None
            LOGGER.info("No preprocessor found")

        # load config and model
        conf = load_config(
            model_path,
        )

        model = load_model(
            model_path,
            config=conf,
            module_path=self.model_to_module_map.get(conf.__class__.__name__, None),
        )
        LOGGER.info("Successfully loaded model")
        return model, preprocessor

    def finetuning(self, input: TinyTimeMixerForecastingTuneInput, tuned_model_name: str, output_dir: Path):
        try:
            LOGGER.info("calling forecast_common")
            answer = self._finetuning_common(input, tuned_model_name=tuned_model_name, tmp_dir=output_dir)
            LOGGER.info("done, returning.")
            return answer
        except Exception as e:
            LOGGER.exception(e)
            raise HTTPException(status_code=500, detail=repr(e))

    @classmethod
    def _validation_data(cls, input: BaseTuneInput) -> pd.DataFrame:
        """Returns validation data. At present we're not sure
        how this is going to work so for now punt and always return
        None"""

        # @TODO fixme

        return None

    def _finetuning_common(self, input: BaseTuneInput, tuned_model_name: str, tmp_dir: Path) -> Path:
        LOGGER.info("in _forecasting_tuning_workflow")

        # set seed
        if input.parameters.random_seed:
            set_seed(input.parameters.random_seed)

        data_schema = input.schema
        train_data: pd.DataFrame = to_pandas(uri=input.data, **data_schema.model_dump())

        validation_data = FinetuningRuntime._validation_data(input)

        model_path = Path(self.config["model_dir"]) / input.model_id

        if not model_path.is_dir():
            LOGGER.info(f"Could not find model at path: {model_path}")
            if TSFM_ALLOW_LOAD_FROM_HF_HUB:
                model_path = input.model_id
                LOGGER.info(f"Using HuggingFace Hub: {model_path}")
            else:
                raise RuntimeError(
                    f"""Could not load model {input.model_id} from {self.config['model_dir']}.
                    If trying to load directly from the HuggingFace Hub please ensure that
                    `TSFM_ALLOW_LOAD_FROM_HF_HUB=1`"""
                )

        model, preprocessor = self.load(model_path)

        # @TODO (correct?)
        base_config = model.config

        # Load existing preprocessor for asset-class specific case
        # tsp = load_preprocessor(model_id_load_path, bucket_name=model_bucket, s3creds=input.s3credentials)
        # model_class, base_config = load_model_config(Path(model_id_load_path))

        # If no preprocessor, then we are finetuning on a new dataset -- no existing preprocessor
        if preprocessor is None:
            preprocessor = TimeSeriesPreprocessor(
                timestamp_column=data_schema.timestamp_column,
                id_columns=data_schema.id_columns,
                target_columns=data_schema.target_columns,
                control_columns=data_schema.control_columns,
                conditional_columns=data_schema.conditional_columns,
                observable_columns=data_schema.observable_columns,
                static_categorical_columns=data_schema.static_categorical_columns,
                context_length=base_config.context_length,
                prediction_length=base_config.prediction_length,
                scaling=False,
                encode_categorical=False,
                freq=data_schema.freq if data_schema.freq else None,
            )
            # Should we set prediction length and context length above?

            # train to estimate freq if not available
            preprocessor.train(train_data)

        # @TODO: if we support different context length
        # we need to dynamically load an appropriate model

        # optional freezing
        if input.parameters.tune_type == TuneTypeEnum.linear_probe.value:
            # Freeze the backbone of the model
            for param in model.backbone.parameters():
                param.requires_grad = False

        # create datasets
        dataset_spec = {
            "id_columns": preprocessor.id_columns,
            "timestamp_column": preprocessor.timestamp_column,
            "target_columns": preprocessor.target_columns,
            "observable_columns": preprocessor.observable_columns,
            "control_columns": preprocessor.control_columns,
            "conditional_columns": preprocessor.conditional_columns,
            "static_categorical_columns": preprocessor.static_categorical_columns,
            "prediction_length": preprocessor.prediction_length,
            "context_length": preprocessor.context_length,
        }

        if input.parameters.fewshot_fraction < 1:
            train_data = select_by_fixed_fraction(
                train_data,
                id_columns=preprocessor.id_columns,
                fraction=input.parameters.fewshot_fraction,
                location="last",
                minimum_size=preprocessor.context_length,
            )
        train_dataset = ForecastDFDataset(train_data, **dataset_spec)

        validation_dataset = ForecastDFDataset(validation_data, **dataset_spec) if validation_data else train_dataset

        # Configure trainer
        parameters = input.parameters
        training_tmp_dir = Path(tmp_dir)
        training_args = TrainingArguments(
            output_dir=training_tmp_dir / "output",
            overwrite_output_dir=True,
            learning_rate=parameters.trainer_args.learning_rate,
            num_train_epochs=parameters.trainer_args.num_train_epochs,
            evaluation_strategy="epoch",
            per_device_train_batch_size=parameters.trainer_args.per_device_train_batch_size,
            per_device_eval_batch_size=parameters.trainer_args.per_device_eval_batch_size,
            dataloader_num_workers=4,
            save_strategy="epoch",
            logging_strategy="epoch",
            save_total_limit=3,
            logging_dir=training_tmp_dir / "logs",  # Make sure to specify a logging directory
            load_best_model_at_end=True,  # Load the best model when training ends
            metric_for_best_model="eval_loss",  # Metric to monitor for early stopping
            greater_is_better=False,  # For loss
            label_names=["future_values"],
            use_cpu=True,  # only needed for testing on Mac :(
        )

        callbacks = []
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
            model=model,
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
        preprocessor.save_pretrained(save_path)
        return save_path
