"""Service handler for HuggingFace models"""

import copy
import importlib
import logging
import pathlib
from abc import abstractmethod
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
import torch
import transformers
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed


from tsfm_public import TimeSeriesForecastingPipeline, TimeSeriesPreprocessor, ForecastDFDataset
from tsfm_public.toolkit.time_series_preprocessor import extend_time_series
from tsfm_public.toolkit.util import select_by_index, select_by_fixed_fraction

from .inference_payloads import BaseParameters, ForecastingMetadataInput, ForecastingParameters
from .service_handler import ForecastingServiceHandler, ServiceHandler
from .tsfm_config import TSFMConfig
from .ftpayloads import TuneTypeEnum
from .filelogging_tracker import FileLoggingCallback

LOGGER = logging.getLogger(__file__)


class HuggingFaceHandler(ServiceHandler):
    """Handler for HuggingFace-like models

    Underlying assumption is that the model makes use of PretrainedConfig/PreTrainedModel
    conventions. During init we register the config using HF AutoConfig.

    Args:
        model_id (str): ID of the model
        model_path (Union[str, Path]): Full path to the model folder.
        handler_config (TSFMConfig): Configuration for the service handler

    """

    def __init__(
        self,
        model_id: str,
        model_path: Union[str, Path],
        handler_config: TSFMConfig,
    ):
        super().__init__(model_id=model_id, model_path=model_path, handler_config=handler_config)

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

        self.config = None
        self.model = None
        self.preprocessor = None

    def load_preprocessor(self, model_path: str) -> TimeSeriesPreprocessor:
        """Load TSFM preprocessor

        Args:
            model_path (str): Local path or HF Hub path to the preprocessor.
        Raises:
            Exception if there is an issue loading the preprocessor. OSError is caught to allow
            loading models when there is no preprocessor.

        Returns:
            TimeSeriesPreprocessor: the loaded preprocessor.
        """

        try:
            preprocessor = TimeSeriesPreprocessor.from_pretrained(model_path)
            LOGGER.info("Successfully loaded preprocessor")
        except OSError:
            preprocessor = None
            LOGGER.info("No preprocessor found")
        except Exception as ex:
            raise ex

        return preprocessor

    def load_hf_config(self, model_path: str, **extra_config_kwargs: Dict[str, Any]) -> PretrainedConfig:
        """Load a HuggingFace config

        Args:
            model_path (str): Local path or HF Hub path for the config.
            extra_config_kwarg: Extra keyword arguments that are passed while loading the config.

        Returns:
            PretrainedConfig: The loaded config.
        """
        # load config, separate from load model, since we may need to inspect config first
        conf = load_config(model_path, **extra_config_kwargs)

        return conf

    def load_hf_model(self, model_path: str, config: PretrainedConfig) -> PreTrainedModel:
        """Load a HuggingFace model from the HF Hub

        Uses the module_path from the handler_config if available to identify the proper class to load
        the model.

        Args:
            model_path (str): Local path or HF Hub path for the model.
            config (PretrainedConfig): A pretrained config object.

        Returns:
            PreTrainedModel: The loaded model.
        """
        model = load_model(
            model_path,
            config=config,
            module_path=self.handler_config.module_path,
        )

        LOGGER.info("Successfully loaded model")
        return model

    @abstractmethod
    def _get_config_kwargs(
        self,
        parameters: Optional[BaseParameters] = None,
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
        ...


class ForecastingHuggingFaceHandler(ForecastingServiceHandler, HuggingFaceHandler):
    """Handler for HuggingFace-like models

    Underlying assumption is that the model makes use of PretrainedConfig/PreTrainedModel
    conventions. During init we register the config using HF AutoConfig.

    Args:
        model_id (str): ID of the model
        model_path (Union[str, Path]): Full path to the model folder.
        handler_config (TSFMConfig): Configuration for the service handler

    """

    def __init__(
        self,
        model_id: str,
        model_path: Union[str, Path],
        handler_config: TSFMConfig,
    ):
        self.training_label_names = ["future_values"]

        # !!! Double check which init
        super().__init__(model_id=model_id, model_path=model_path, handler_config=handler_config)

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

    def _prepare(
        self,
        data: pd.DataFrame,
        future_data: Optional[pd.DataFrame] = None,
        schema: Optional[ForecastingMetadataInput] = None,
        parameters: Optional[ForecastingParameters] = None,
        **kwargs,
    ) -> "ForecastingHuggingFaceHandler":
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

        preprocessor = self.load_preprocessor(self.model_path)

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
        model_config = self.load_hf_config(self.model_path, **model_config_kwargs)

        model = self.load_hf_model(self.model_path, config=model_config)

        self.config = model_config
        self.model = model
        self.preprocessor = preprocessor

        return self

    def _run(
        self,
        data: pd.DataFrame,
        future_data: Optional[pd.DataFrame] = None,
        schema: Optional[ForecastingMetadataInput] = None,
        parameters: Optional[ForecastingParameters] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """_summary_

        Args:
            data (pd.DataFrame): Input historical time series data.
            future_data (Optional[pd.DataFrame], optional): Input future time series data, useful for
                passing future exogenous if known. Defaults to None.
            schema (Optional[ForecastingMetadataInput], optional): Schema information from the original inference
                request. Includes information about columns and their role. Defaults to None.
            parameters (Optional[ForecastingParameters], optional): Parameters from the original inference
                request. Defaults to None.

        Returns:
            pd.DataFrame: The forecasts produced by the model.
        """

        # warn if future data is not provided, but is needed by the model
        # Remember preprocessor.exogenous_channel_indices are the exogenous for which future data is available
        if self.preprocessor.exogenous_channel_indices and future_data is None:
            raise ValueError(
                "Future data should be provided for exogenous columns where the future is known (`control_columns` and `observable_columns`)"
            )

        if not self.preprocessor.exogenous_channel_indices and future_data is not None:
            raise ValueError("Future data was provided, but the model does not support or require future exogenous.")

        # future_data checks
        if future_data is not None:
            if schema.id_columns:
                data_lengths = future_data.groupby(schema.id_columns)[schema.id_columns].apply(len)
                fd_min_len_index = data_lengths.argmin()
                fd_min_data_length = data_lengths.iloc[fd_min_len_index]
                fd_max_data_length = data_lengths.max()
            else:
                fd_min_data_length = fd_max_data_length = len(future_data)
            LOGGER.info(
                f"Future Data length recieved {len(future_data)}, minimum series length: {fd_min_data_length}, maximum series length: {fd_max_data_length}"
            )

            # if data is too short, raise error
            prediction_length = getattr(self.config, "prediction_filter_length", None)
            has_prediction_filter = prediction_length is not None

            model_prediction_length = self.config.prediction_length

            prediction_length = prediction_length if prediction_length is not None else model_prediction_length
            if fd_min_data_length < prediction_length:
                err_str = (
                    "Future data should have time series of length that is at least the specified prediction length. "
                )
                if schema.id_columns:
                    err_str += f"Received {fd_min_data_length} time points for id {data_lengths.index[fd_min_len_index]}, but expected {prediction_length} time points."
                else:
                    err_str += (
                        f"Received {fd_min_data_length} time points, but expected {prediction_length} time points."
                    )
                raise ValueError(err_str)

            # if data exceeds prediction filter length, truncate
            if fd_max_data_length > model_prediction_length:
                LOGGER.info(f"Truncating future series lengths to {model_prediction_length}")
                future_data = select_by_index(
                    future_data, id_columns=schema.id_columns, end_index=model_prediction_length
                )

            # if provided data is greater than prediction_filter_length, but less than model_prediction_length we extend
            if has_prediction_filter and fd_min_data_length < model_prediction_length:
                LOGGER.info(f"Extending time series to match model prediction length: {model_prediction_length}")
                future_data = extend_time_series(
                    time_series=future_data,
                    freq=self.preprocessor.freq,
                    timestamp_column=schema.timestamp_column,
                    grouping_columns=schema.id_columns,
                    total_periods=model_prediction_length,
                )

        device = "cpu" if not torch.cuda.is_available() else "cuda"
        forecast_pipeline = TimeSeriesForecastingPipeline(
            model=self.model,
            explode_forecasts=True,
            feature_extractor=self.preprocessor,
            add_known_ground_truth=False,
            freq=self.preprocessor.freq,
            device=device,
        )
        forecasts = forecast_pipeline(data, future_time_series=future_data, inverse_scale_outputs=True)

        return forecasts

    def _train(
        self,
        data: pd.DataFrame,
        schema: ForecastingMetadataInput,
        parameters: ForecastingParameters,
        tuned_model_name: str, 
        tmp_dir: Path
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
            "prediction_length": self.preprocessor.prediction_length,
            "context_length": self.preprocessor.context_length,
        }

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
        parameters  = parameters
        training_tmp_dir = Path(tmp_dir)
        training_args = TrainingArguments(
            output_dir=training_tmp_dir / "output",
            overwrite_output_dir=True,
            learning_rate=parameters.trainer_args.learning_rate,
            num_train_epochs=parameters.trainer_args.num_train_epochs,
            eval_strategy="epoch",
            per_device_train_batch_size=parameters.trainer_args.per_device_train_batch_size,
            per_device_eval_batch_size=parameters.trainer_args.per_device_eval_batch_size,
            dataloader_num_workers=4, # to do: auto config this parameter
            save_strategy="epoch",
            logging_strategy="epoch",
            save_total_limit=3,
            logging_dir=training_tmp_dir / "logs",  # Make sure to specify a logging directory
            load_best_model_at_end=True,  # Load the best model when training ends
            metric_for_best_model="eval_loss",  # Metric to monitor for early stopping
            greater_is_better=False,  # For loss
            label_names=self.training_label_names,
            use_cpu=not torch.cuda.is_available(),
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


    def _calculate_data_point_counts(
        self,
        data: pd.DataFrame,
        future_data: Optional[pd.DataFrame] = None,
        output_data: Optional[pd.DataFrame] = None,
        schema: Optional[ForecastingMetadataInput] = None,
        parameters: Optional[ForecastingParameters] = None,
    ) -> Dict[str, int]:
        """Implementation for counting datapoints in input and output

        Assumes data has been truncated
        Future data may not be truncated
        """

        input_ts_columns = sum(
            [
                len(c)
                for c in [
                    schema.target_columns,
                    schema.conditional_columns,
                    schema.control_columns,
                    schema.observable_columns,
                ]
            ]
        )
        input_ts_columns = input_ts_columns if input_ts_columns != 0 else data.shape[1] - len(schema.id_columns) - 1
        input_static_columns = len(schema.static_categorical_columns)
        num_target_columns = (
            len(schema.target_columns) if schema.target_columns != [] else data.shape[1] - len(schema.id_columns) - 1
        )
        unique_ts = len(data.drop_duplicates(subset=schema.id_columns)) if schema.id_columns else 1
        has_future_data = future_data is not None

        # we don't count the static columns in the future data
        # we only count future data which falls within forecast horizon "causal assumption"
        # note that output_data.shape[0] = unique_ts * prediction_length
        future_data_points = (input_ts_columns - num_target_columns) * output_data.shape[0] if has_future_data else 0

        counts = {
            "input_data_points": input_ts_columns * data.shape[0]
            + input_static_columns * unique_ts
            + future_data_points,
            "output_data_points": output_data.shape[0] * num_target_columns,
        }
        LOGGER.info(f"Data point counts: {counts}")
        return counts


def register_config(model_type: str, model_config_name: str, module_path: str) -> None:
    """Register a configuration for a particular model architecture

    Args:
        model_type (Optional[str], optional): The type of the model, from the model implementation. Defaults to None.
        model_config_name (Optional[str], optional): The name of configuration class for the model. Defaults to None.
        module_path (Optional[str], optional): Python module path that can be used to load the
            config/model. Defaults to None.

    Raises:
        RuntimeError: Raised when the module cannot be imported from the provided module path.
    """
    # example
    # model_type: "tinytimemixer"
    # model_config_name: "TinyTimeMixerConfig"
    # module_path: "tsfm"  # place where config should be importable

    # AutoConfig.register("tinytimemixer", TinyTimeMixerConfig)
    try:
        mod = importlib.import_module(module_path)
        conf_class = getattr(mod, model_config_name, None)
    except ModuleNotFoundError as exc:  # modulenot found, key error ?
        raise RuntimeError(f"Could not load {model_config_name} from {module_path}") from exc

    if conf_class is not None:
        AutoConfig.register(model_type, conf_class)
    else:
        # issue warning?
        pass


def load_config(
    model_path: Union[str, pathlib.Path],
    model_type: Optional[str] = None,
    model_config_name: Optional[str] = None,
    module_path: Optional[str] = None,
    **extra_config_kwargs: Dict[str, Any],
) -> PretrainedConfig:
    """Load configuration

    Attempts to load the configuration, if it is not loadable, then we register it with the AutoConfig mechanism.

    Args:
        model_path (pathlib.Path): The path from which to load the config.
        model_type (Optional[str], optional): The type of the model, from the model implementation. Defaults to None.
        model_config_name (Optional[str], optional): The name of configuration class for the model. Defaults to None.
        module_path (Optional[str], optional): Python module path that can be used to load the
            config/model. Defaults to None.

    Returns:
        PretrainedConfig: The configuration object corresponding to the pretrained model.
    """
    # load config first try autoconfig, if not then we register and load
    try:
        conf = AutoConfig.from_pretrained(model_path, **extra_config_kwargs)
    except (KeyError, ValueError) as exc:  # determine error raised by autoconfig
        if model_type is None or model_config_name is None or module_path is None:
            raise ValueError("model_type, model_config_name, and module_path should be specified.") from exc

        register_config(model_type, model_config_name, module_path)
        conf = AutoConfig.from_pretrained(model_path, **extra_config_kwargs)

    return conf


def _get_model_class(config: PretrainedConfig, module_path: Optional[str] = None) -> type:
    """Helper to find model class based on config object

    First the module_path will be checked if it can be loaded in the current environment. If not
    then the transformers library will be used.

    Args:
        config (PretrainedConfig): HF configuration for the model.
        module_path (Optional[str], optional): Python module path that can be used to load the
            config/model. Defaults to None.

    Raises:
        AttributeError: Raised if the module at module_path cannot be loaded.
        AttributeError: If the architecture provided by the config cannot be loaded from
            the module.

    Returns:
        type: The class for the model.
    """
    if module_path is not None:
        try:
            mod = importlib.import_module(module_path)
        except ModuleNotFoundError as exc:
            raise AttributeError("Could not load module '{module_path}'.") from exc
    else:
        mod = transformers

    # get architecture from model config
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        try:
            model_class = getattr(mod, arch)
            return model_class
        except AttributeError as exc:
            # catch specific error import error or attribute error
            raise AttributeError(f"Could not load model class for architecture '{arch}'.") from exc


def load_model(
    model_path: Union[str, pathlib.Path],
    config: Optional[PretrainedConfig] = None,
    module_path: Optional[str] = None,
) -> PreTrainedModel:
    """Load a pretrained model.
    If module_path is provided, load the model using the provided module path.

    Args:
        model_path (Union[str, pathlib.Path]): Path to a location where the model can be loaded.
        config (Optional[PretrainedConfig], optional): HF Configuration object. Defaults to None.
        module_path (Optional[str], optional): Python module path that can be used to load the
            config/model. Defaults to None.

    Raises:
        ValueError: Raised if loading from a module_path and a configuration object is not provided.

    Returns:
        PreTrainedModel: The loaded pretrained model.
    """

    if module_path is not None and config is None:
        return ValueError("Config must be provided when loading from a custom module_path")

    if config is not None:
        model_class = _get_model_class(config, module_path=module_path)
        LOGGER.info(f"Found model class: {model_class.__name__}")
        return model_class.from_pretrained(model_path, config=config)

    return AutoModel.from_pretrained(model_path)
