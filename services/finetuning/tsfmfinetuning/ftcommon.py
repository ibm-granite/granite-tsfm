""" "Common implementations for fine-tuning"""

import logging
import tempfile
import uuid
from pathlib import Path

import pandas as pd
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed

from tsfm_public.toolkit.dataset import ForecastDFDataset
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor
from tsfm_public.toolkit.util import select_by_fixed_fraction

from .ftpayloads import BaseTuneInput, TuneOutput, TuneTypeEnum
from .ioutils import copy_dir_to_s3, getminio, to_pandas


LOGGER = logging.getLogger(__file__)


def _forecasting_tuning_workflow(
    input: BaseTuneInput,
    tmp_dir: str,
    model_bucket: str = "tsfm-services",
    tuned_model_name: str = "my_tuned_model",
):
    LOGGER.info("in _forecasting_tuning_workflow")

    # set seed
    if input.parameters.random_seed:
        set_seed(input.parameters.random_seed)

    # get the data
    # @todo discuss with wx.ai why we want to use "data_schema" instead
    schema = input.schema.model_dump()
    train_data: pd.DataFrame = to_pandas(uri=input.data, **schema)
    if input.validation_data is not None and len(input.validation_data) > 0:
        validation_data: pd.DataFrame = to_pandas(uri=input.validation_data, **schema)
    else:
        validation_data = None

    model_id_load_path = prepare_model_and_preprocessor(
        model_id=input.model_id,
        bucket_name=model_bucket,
        s3creds=input.s3credentials,
    )

    # Load existing preprocessor for asset-class specific case
    tsp = load_preprocessor(model_id_load_path, bucket_name=model_bucket, s3creds=input.s3credentials)
    model_class, base_config = load_model_config(Path(model_id_load_path))

    # If no preprocessor, then we are finetuning on a new dataset -- no existing preprocessor
    if tsp is None:
        tsp = TimeSeriesPreprocessor(
            timestamp_column=schema.timestamp_column,
            id_columns=schema.id_columns,
            target_columns=schema.target_columns,
            control_columns=schema.control_columns,
            conditional_columns=schema.conditional_columns,
            observable_columns=schema.observable_columns,
            static_categorical_columns=schema.static_categorical_columns,
            context_length=base_config.context_length,
            prediction_length=base_config.prediction_length,
            scaling=False,
            encode_categorical=False,
            freq=input.freq if input.freq else None,
        )
        # Should we set prediction length and context length above?

        # train to estimate freq if not available
        tsp.train(train_data)

    config_kwargs = {
        # context_length=base_config.context_length,  # do not reset
        # prediction_length=base_config.prediction_length,  # do not reset
        "num_input_channels": tsp.num_input_channels,
        # d_model=3 * 64,
        # decoder_d_model=2 * 64,
        "prediction_channel_indices": tsp.prediction_channel_indices,
        "exogenous_channel_indices": tsp.exogenous_channel_indices,
        "loss": input.model_parameters.loss,  # check
        "dropout": input.model_parameters.dropout,
        "head_dropout": input.model_parameters.head_dropout,
        "distribution_output": input.model_parameters.distribution_output,  # check
        "num_parallel_samples": input.model_parameters.num_parallel_samples,
        "decoder_num_layers": input.model_parameters.decoder_num_layers,  # check
        "decoder_adaptive_patching_levels": input.model_parameters.decoder_adaptive_patching_levels,
        "decoder_raw_residual": input.model_parameters.decoder_raw_residual,
        "decoder_mode": input.model_parameters.decoder_mode,
        "use_decoder": input.model_parameters.use_decoder,
        "enable_forecast_channel_mixing": input.model_parameters.enable_forecast_channel_mixing,
        "fcm_gated_attn": input.model_parameters.fcm_gated_attn,
        "fcm_context_length": input.model_parameters.fcm_context_length,
        "fcm_use_mixer": input.model_parameters.fcm_use_mixer,
        "fcm_mix_layers": input.model_parameters.fcm_mix_layers,
        "fcm_prepend_past": input.model_parameters.fcm_prepend_past,
    }

    # to do: if we support different context length we need to dynamically load an appropriate model

    # to do: update this to work with preloaded_model and possible config changes
    model = load_model(
        model_id=model_id_load_path,
        bucket_name=model_bucket,
        s3creds=input.s3credentials,
        **config_kwargs,
    )

    # optional freezing
    if input.tune_type == TuneTypeEnum.linear_probe.value:
        # Freeze the backbone of the model
        for param in model.backbone.parameters():
            param.requires_grad = False

    # create datasets
    dataset_spec = {
        "id_columns": tsp.id_columns,
        "timestamp_column": tsp.timestamp_column,
        "target_columns": tsp.target_columns,
        "observable_columns": tsp.observable_columns,
        "control_columns": tsp.control_columns,
        "conditional_columns": tsp.conditional_columns,
        "static_categorical_columns": tsp.static_categorical_columns,
        "prediction_length": tsp.prediction_length,
        "context_length": tsp.context_length,
    }

    if input.fewshot_fraction < 1:
        train_data = select_by_fixed_fraction(
            train_data,
            id_columns=tsp.id_columns,
            fraction=input.fewshot_fraction,
            location="last",
            minimum_size=tsp.context_length,
        )
    train_dataset = ForecastDFDataset(train_data, **dataset_spec)

    validation_dataset = ForecastDFDataset(validation_data, **dataset_spec) if validation_data else train_dataset

    # Configure trainer
    training_tmp_dir = Path(tmp_dir)
    training_args = TrainingArguments(
        output_dir=training_tmp_dir / "output",
        overwrite_output_dir=True,
        learning_rate=input.trainer_args.learning_rate,
        num_train_epochs=input.trainer_args.num_train_epochs,
        evaluation_strategy="epoch",
        per_device_train_batch_size=input.trainer_args.per_device_train_batch_size,
        per_device_eval_batch_size=input.trainer_args.per_device_eval_batch_size,
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

    # tmpdir.cleanup() ?
    callbacks = []
    if input.trainer_args.early_stopping and validation_dataset:
        # Create the early stopping callback
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=input.trainer_args.early_stopping_patience,  # Number of epochs with no improvement after which to stop
            early_stopping_threshold=input.trainer_args.early_stopping_threshold,  # Minimum improvement required to consider as improvement
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
    trainer.save_model(save_path / "model")
    tsp.save_pretrained(save_path / "preprocessor")
    # save preprocessor to: (save_path / tuned_model_name / "preprocessor")

    return save_path


def forecasting_tuning_to_local(
    target_dir: Path,
    model_name: str,
    input: BaseTuneInput,
) -> TuneOutput:
    """Run finetuning. TuneOutput will contain a reference to a model saved to the local file system."""
    model_path: Path = _forecasting_tuning_workflow(input, tuned_model_name=model_name, tmp_dir=target_dir)
    return TuneOutput(training_ref=model_path.as_posix())


def forecasting_tuning_to_s3(
    input: BaseTuneInput,
) -> TuneOutput:
    """Run finetuning. TuneOutput will contain a reference to a model saved to a s3-compatible object store."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # make prefix and unique tuned model name
        model_prefix = input.tune_prefix if input.tune_prefix else ""
        uid = uuid.uuid4().hex
        model_name = f"{model_prefix}{input.model_id}-{uid}"
        bucket_name = input.results_bucket
        model_path: Path = _forecasting_tuning_workflow(input, tuned_model_name=model_name, tmp_dir=tmp_dir)
        s3creds = input.s3creds
        mc = getminio(s3creds=s3creds)
        copy_dir_to_s3(mc, bucket_name, model_path, prefix=model_name)
        return TuneOutput(training_ref=f"s3a://{bucket_name}/{model_name}")
