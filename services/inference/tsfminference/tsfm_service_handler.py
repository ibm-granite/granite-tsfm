"""Service Handler for TSFM Models"""

import copy
import logging
from typing import Optional

import pandas as pd

from tsfm_public import TimeSeriesForecastingPipeline, TimeSeriesPreprocessor

from .inference_payloads import (
    ForecastingMetadataInput,
    ForecastingParameters,
)
from .service_handler import HuggingFaceHandler


LOGGER = logging.getLogger(__file__)


class DefaultHandler(HuggingFaceHandler):
    def _get_config_kwargs(
        self,
        parameters: Optional[ForecastingParameters] = None,
        preprocessor: Optional[TimeSeriesPreprocessor] = None,
    ):
        default_config_kwargs = {
            "num_input_channels": preprocessor.num_input_channels,
        }
        return default_config_kwargs

    def _prepare(
        self,
        schema: Optional[ForecastingMetadataInput] = None,
        parameters: Optional[ForecastingParameters] = None,
    ) -> "DefaultHandler":
        # to do: use config parameters below
        # issue: _load may need to know data length to set parameters upon model load (multst)

        preprocessor_params = copy.deepcopy(schema.model_dump())
        preprocessor_params["prediction_length"] = parameters.prediction_length

        LOGGER.info(f"Preprocessor params: {preprocessor_params}")

        preprocessor = self.load_preprocessor(self.model_id)

        if preprocessor is None:
            preprocessor = TimeSeriesPreprocessor(
                **preprocessor_params,
                scaling=False,
                encode_categorical=False,
            )
            # we don't set context length or prediction length above because it is not needed for inference

        model_config_kwargs = self._get_config_kwargs(
            parameters=parameters,
            preprocessor=preprocessor,
        )
        LOGGER.info(f"model_config_kwargs: {model_config_kwargs}")
        model_config = self.load_hf_config(self.model_id, **model_config_kwargs)

        model = self.load_hf_model(self.model_id, config=model_config)

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
    ) -> pd.DataFrame:
        # tbd, can this be moved to the HFWrapper?
        # error checking once data available
        if self.preprocessor.freq is None:
            # train to estimate freq if not available
            self.preprocessor.train(data)
            LOGGER.info(f"Data frequency determined: {self.preprocessor.freq}")

        # warn if future data is not provided, but is needed by the model
        if self.preprocessor.exogenous_channel_indices and future_data is None:
            ValueError(
                "Future data should be provided for exogenous columns where the future is known (`control_columns` and `observable_columns`)"
            )

        forecast_pipeline = TimeSeriesForecastingPipeline(
            model=self.model,
            explode_forecasts=True,
            feature_extractor=self.preprocessor,
            add_known_ground_truth=False,
            freq=self.preprocessor.freq,
        )
        forecasts = forecast_pipeline(data, future_time_series=future_data, inverse_scale_outputs=True)

        return forecasts

    def _train(
        self,
    ): ...


class TinyTimeMixerHandler(DefaultHandler):
    def _get_config_kwargs(
        self,
        parameters: Optional[ForecastingParameters] = None,
        preprocessor: Optional[TimeSeriesPreprocessor] = None,
    ):
        config_kwargs = {
            "num_input_channels": preprocessor.num_input_channels,
            "prediction_filter_length": parameters.prediction_length,
        }
        return config_kwargs
