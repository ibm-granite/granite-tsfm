# Copyright contributors to the TSFM project
#
"""Tsfminference Runtime"""

import copy
import datetime
import logging
from typing import Any, Dict, List

import pandas as pd
from fastapi import APIRouter, HTTPException
from starlette import status

from tsfm_public import TimeSeriesForecastingPipeline, TimeSeriesPreprocessor
from tsfm_public.toolkit.util import select_by_index

from . import TSFM_ALLOW_LOAD_FROM_HF_HUB, TSFM_MODEL_DIR
from .constants import API_VERSION
from .errors import error_message
from .hfutil import load_config, load_model, register_config
from .inference_payloads import ForecastingInferenceInput, PredictOutput


LOGGER = logging.getLogger(__file__)


class InferenceRuntime:
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
        self.router = APIRouter(prefix=f"/{API_VERSION}/inference", tags=["inference"])
        self.router.add_api_route(
            "/forecasting",
            self.forecast,
            methods=["POST"],
            response_model=PredictOutput,
        )
        app.include_router(self.router)

    def load(self, model_path: str, **extra_config_kwargs: Dict[str, Any]):
        try:
            preprocessor = TimeSeriesPreprocessor.from_pretrained(model_path)
            LOGGER.info("Successfully loaded preprocessor")
        except OSError:
            preprocessor = None
            LOGGER.info("No preprocessor found")
        except Exception as ex:
            return None, None, ex

        # load config and model
        conf = load_config(model_path, **extra_config_kwargs)

        model, ex = load_model(
            model_path,
            config=conf,
            module_path=self.model_to_module_map.get(conf.__class__.__name__, None),
        )
        if ex is not None:
            return None, preprocessor, ex

        LOGGER.info("Successfully loaded model")
        return model, preprocessor, None

    def forecast(self, input: ForecastingInferenceInput):
        LOGGER.info("calling forecast_common")
        answer, ex = self._forecast_common(input)

        if ex is not None:
            detail = error_message(ex)
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)

        LOGGER.info("done, returning.")
        return answer

    def _forecast_common(self, input_payload: ForecastingInferenceInput) -> PredictOutput:
        # we need some sort of model registry
        # payload = input_payload.model_dump()  # do we need?

        schema_params = input_payload.schema.model_dump()
        params = input_payload.parameters.model_dump()

        data = decode_data(input_payload.data, schema_params)
        future_data = decode_data(input_payload.future_data, schema_params)

        model_path = TSFM_MODEL_DIR / input_payload.model_id

        if not model_path.is_dir():
            LOGGER.info(f"Could not find model at path: {model_path}")
            if TSFM_ALLOW_LOAD_FROM_HF_HUB:
                model_path = input_payload.model_id
                LOGGER.info(f"Using HuggingFace Hub: {model_path}")
            else:
                raise RuntimeError(
                    f"Could not load model {input_payload.model_id} from {TSFM_MODEL_DIR.as_posix()}. If trying to load directly from the HuggingFace Hub please ensure that `TSFM_ALLOW_LOAD_FROM_HF_HUB=1`"
                )

        preprocessor_params = copy.deepcopy(schema_params)
        preprocessor_params["prediction_length"] = params["prediction_length"]

        model_config_kwargs = {
            "prediction_filter_length": preprocessor_params.get("prediction_length", None),
        }

        model, preprocessor, ex = self.load(model_path, **model_config_kwargs)

        if ex is not None:
            return None, ex

        LOGGER.info(f"Preprocessor params: {preprocessor_params}")
        # preprocess
        if preprocessor is None:
            preprocessor = TimeSeriesPreprocessor(
                **preprocessor_params,
                scaling=False,
                encode_categorical=False,
            )
            # we don't set context length or prediction length above because it is not needed for inference

            # train to estimate freq if not available
            try:
                preprocessor.train(data)
            except Exception as ex:
                return None, ex

        LOGGER.info(f"Data frequency determined: {preprocessor.freq}")

        # warn if future data is not provided, but is needed by the model
        if preprocessor.exogenous_channel_indices and future_data is None:
            return None, ValueError(
                "Future data should be provided for exogenous columns where the future is known (`control_columns` and `observable_columns`)"
            )

        context_length = model.config.context_length

        # collect and check underlying time series lengths
        if preprocessor_params["id_columns"]:
            data_lengths = data.groupby(preprocessor_params["id_columns"]).apply(len)
            min_len_index = data_lengths.argmin()
            data_length = data_lengths.iloc[min_len_index]
        else:
            data_length = len(data)

        LOGGER.info(f"Data length recieved {len(data)}, minimum series length: {data_length}")

        if data_length < context_length:
            err_str = "Data should have time series of length that is at least the required model context length. "
            if preprocessor_params["id_columns"]:
                err_str += f"Received {data_length} time points for id {data_lengths.index[min_len_index]}, but model requires {context_length} time points"
            else:
                err_str += f"Received {data_length} time points, but model requires {context_length} time points"

            return None, ValueError(err_str)

        # truncate data length when exploding
        if data_length > context_length:
            data = select_by_index(data, id_columns=preprocessor_params["id_columns"], start_index=-context_length)

        forecast_pipeline = TimeSeriesForecastingPipeline(
            model=model,
            explode_forecasts=True,
            feature_extractor=preprocessor,
            add_known_ground_truth=False,
            freq=preprocessor.freq,
        )

        try:
            forecasts = forecast_pipeline(data, future_time_series=future_data, inverse_scale_outputs=True)
        except Exception as e:
            return None, e

        return PredictOutput(
            model_id=input_payload.model_id,
            created_at=datetime.datetime.now().isoformat(),
            results=[forecasts.to_dict(orient="list")],
        ), None


def decode_data(data: Dict[str, List[Any]], schema: Dict[str, Any]) -> pd.DataFrame:
    if not data:
        return None

    df = pd.DataFrame.from_dict(data)
    if ts_col := schema["timestamp_column"]:
        df[ts_col] = pd.to_datetime(df[ts_col])
    return df
