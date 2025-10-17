"""Inference handler for TSFM models"""

import copy
import logging
import tempfile
import threading
from functools import cache
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
import torch
from filelock import FileLock

from tsfm_public import TimeSeriesForecastingPipeline
from tsfm_public.toolkit.conformal import PostHocProbabilisticProcessor
from tsfm_public.toolkit.time_series_preprocessor import (
    TimeSeriesPreprocessor,
    extend_time_series,
)
from tsfm_public.toolkit.tsfm_config import TSFMConfig
from tsfm_public.toolkit.util import select_by_index

from . import TSFM_ALLOW_LOAD_FROM_HF_HUB
from .inference_payloads import ForecastingMetadataInput, ForecastingParameters
from .tsfm_util import load_config, load_model, register_config


LOCAL_FILES_ONLY = not TSFM_ALLOW_LOAD_FROM_HF_HUB
LOGGER = logging.getLogger(__file__)


_THREAD_LOCK = threading.Lock()
_, _IPROCESS_LOCK_FILE = tempfile.mkstemp()


class TSFMForecastingInferenceHandler:
    def __init__(
        self,
        model_id: str,
        model_path: Union[str, Path],
        handler_config: TSFMConfig,
    ):
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

        # loosen the schema checking when using a saved preprocessor
        self.strict_schema_match = False

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

    @classmethod
    @cache
    def _cached_load_model(cls, model_path, config: str, module_path, config_class):
        with FileLock(_IPROCESS_LOCK_FILE):
            with _THREAD_LOCK:
                with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=True) as tmp:
                    tmp.write(config)
                    tmp.flush()
                    return load_model(
                        model_path=model_path,
                        config=config_class.from_json_file(tmp.name),
                        module_path=module_path,
                    )

    def prepare(
        self,
        data: pd.DataFrame,
        future_data: Optional[pd.DataFrame] = None,
        schema: Optional[ForecastingMetadataInput] = None,
        parameters: Optional[ForecastingParameters] = None,
        **kwargs,
    ) -> "TSFMForecastingInferenceHandler":
        """Implementation of prepare for HF-like models. We assume the model will make use of the TSFM
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
            preprocessor = TimeSeriesPreprocessor.from_pretrained(self.model_path, local_files_only=LOCAL_FILES_ONLY)
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
            to_check = [
                "conditional_columns",
                "control_columns",
                "observable_columns",
                "static_categorical_columns",
                "categorical_columns",
            ]

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
            # id_columns are not checked, since an artificial id may be added to do batch inference
            to_check_columns = [
                "timestamp_column",
                "target_columns",
                "conditional_columns",
                "control_columns",
                "observable_columns",
                "categorical_columns",
                "static_categorical_columns",
            ]

            to_check_params = ["freq"]
            to_check = to_check_columns.copy()
            to_check.extend(to_check_params)

            for param in to_check:
                param_val_saved = getattr(preprocessor, param)
                param_val = preprocessor_params[param]
                # if a parameter is passed we check it matches the preprocessor
                if self.strict_schema_match or not (param_val is None or param_val == [] or param_val == ""):
                    if param_val != param_val_saved:
                        raise ValueError(
                            f"Attempted to use a fine-tuned model with a different schema, please confirm you have the correct model_id and schema. Error in parameter {param}: received {param_val} but expected {param_val_saved}."
                        )
                # then we check that parameters from the saved preprocessor match what is in the data
                if param in to_check_params:
                    continue
                p = param_val_saved if isinstance(param_val_saved, list) else [param_val_saved]
                if any(c not in data.columns for c in p):
                    raise ValueError(
                        f"Attempted to use a fine-tuned model with data that does not match the saved schema, please confirm you have the correct model_id and appropriate data. Error in parameter {param}: data does not contain a column named {param_val_saved}."
                    )

        model_config_kwargs = self._get_config_kwargs(
            parameters=parameters,
            preprocessor=preprocessor,
        )
        LOGGER.info(f"model_config_kwargs: {model_config_kwargs}")
        model_config = load_config(self.model_path, **model_config_kwargs)

        model = TSFMForecastingInferenceHandler._cached_load_model(
            self.model_path,
            config=model_config.to_json_string(),
            module_path=self.handler_config.module_path,
            config_class=model_config.__class__,
        )

        self.config = model_config
        self.model = model
        self.preprocessor = preprocessor

    def run(
        self,
        data: pd.DataFrame,
        future_data: Optional[pd.DataFrame] = None,
        quantile_calibration_data: Optional[pd.DataFrame] = None,
        schema: Optional[ForecastingMetadataInput] = None,
        parameters: Optional[ForecastingParameters] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Implementation of run for TSFM models.
        Checks prediction length, data length (both past and future exogenous), configures batch size, determines
        device and uses the forecasting pipeline to generate forecasts.

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

        batch_size = (
            parameters.inference_batch_size
            if parameters.inference_batch_size
            else self.handler_config.inference_batch_size
        )
        LOGGER.info(f"Using inference batch size: {batch_size}")

        device = "cpu" if not torch.cuda.is_available() else "cuda"

        extra_pipeline_args = getattr(self.handler_config, "extra_pipeline_arguments", {})

        # if requested, add quantile predictions to our forecast
        prediction_quantiles = parameters.prediction_quantiles if parameters.prediction_quantiles else None
        if prediction_quantiles is None and quantile_calibration_data is not None:
            raise ValueError("If you provide quantile_calibration_data, you must provide prediction_quantiles")
        if prediction_quantiles is not None and quantile_calibration_data is None:
            raise ValueError("Providing prediction_quantiles requires that you also provide quantile_calibration_data")

        pp_processor: PostHocProbabilisticProcessor = None

        if prediction_quantiles:
            # quantile_calibration_data doesn't necessarily overlap with data
            # we need to generated a new forecast over a region where we have
            # ground truth in order to train PostHocProbabilisticProcessor
            # suppose below represents period spanning quantile_calibration_data
            # [t0, t1+fl] where t1-t0 is the minimum context length for a forecast
            # |
            # |
            # |__________________________|_______________________________(time)
            # t0                         t1                         t1+fl
            # we pass up to t1 to forecast up to t1+fl then we have ground truth
            # (from the original data) and predictions over [t1, t1+fl] to pass
            # to PostHocProbabilisticProcessor

            # 1st pass is generate forecasts using the calibration data to train
            # PostHocProbabilisticProcessor

            min_context_length = self.handler_config.minimum_context_length
            max_prediction_length = self.handler_config.maximum_prediction_length

            # check that each timeseries satisfies minimum length requirements
            if schema.id_columns:
                data_lengths = quantile_calibration_data.groupby(schema.id_columns)[schema.id_columns].apply(len)
                min_len_index = data_lengths.argmin()
                min_data_length = data_lengths.iloc[min_len_index]
                max_data_length = data_lengths.max()
            else:
                min_data_length = max_data_length = len(quantile_calibration_data)
            LOGGER.info(
                f"quantile_calibration_data {len(quantile_calibration_data)}, minimum series length: {min_data_length}, maximum series length: {max_data_length}"
            )

            if min_data_length < min_context_length + max_prediction_length:
                raise ValueError(
                    f"""The size of each timeseries in the given quantile_calibration_data
                    must be greater than or equal to the sum of the minimum_context_length {min_context_length}"
                    plus the model's default prediction length of {max_prediction_length}"""
                )

            forecast_pipeline = TimeSeriesForecastingPipeline(
                model=self.model,
                feature_extractor=self.preprocessor,
                freq=self.preprocessor.freq,
                device=device,
                batch_size=2000,
                inverse_scale_outputs=True,
                explode_forecasts=False,
                **extra_pipeline_args,
            )
            forecasts = forecast_pipeline(quantile_calibration_data)
            prediction_columns = [f"{c}_prediction" for c in schema.target_columns]
            pp_processor = PostHocProbabilisticProcessor(
                id_columns=schema.id_columns, window_size=100, method="conformal", quantiles=prediction_quantiles
            )

            pp_processor = pp_processor.train(
                y_cal_gt=forecasts[schema.target_columns + schema.id_columns],
                y_cal_pred=forecasts[prediction_columns + schema.id_columns],
            )  # we use forecasts twice b/c it does contain the quantile calibration data in a better format via the call to forecast_pipeline(quantile_calibration_data)

        forecast_pipeline = TimeSeriesForecastingPipeline(
            model=self.model,
            explode_forecasts=True,
            feature_extractor=self.preprocessor,
            add_known_ground_truth=False,
            freq=self.preprocessor.freq,
            device=device,
            batch_size=1000,
            probabilistic_processor=pp_processor,
            **extra_pipeline_args,
        )
        forecasts = forecast_pipeline(
            data,
            future_time_series=future_data,
            inverse_scale_outputs=True,
        )

        return forecasts

    def calculate_data_point_counts(
        self,
        data: pd.DataFrame,
        future_data: Optional[pd.DataFrame] = None,
        quantile_calibration_data: Optional[pd.DataFrame] = None,
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


class TinyTimeMixerForecastingInferenceHandler(TSFMForecastingInferenceHandler):
    """Service handler for the tiny time mixer model"""

    def _get_config_kwargs(
        self,
        parameters: Optional[ForecastingParameters] = None,
        preprocessor: Optional[TimeSeriesPreprocessor] = None,
    ) -> Dict[str, Any]:
        """Implements the _get_config_kwargs method for TTM models.

        Args:
            parameters (Optional[ForecastingParameters], optional): Inference parameters. Defaults to None.
            preprocessor (Optional[TimeSeriesPreprocessor], optional): Configured preprocessor. Defaults to None.

        Returns:
            Dict[str, Any]: Updated config arguments that get passed to during model load.
        """
        config_kwargs = {
            "num_input_channels": preprocessor.num_input_channels,
            "prediction_filter_length": parameters.prediction_length,
            "exogenous_channel_indices": preprocessor.exogenous_channel_indices,
            "prediction_channel_indices": preprocessor.prediction_channel_indices,
        }
        return config_kwargs
