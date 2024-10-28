# Copyright contributors to the TSFM project
#
"""Tsfminference Runtime"""

import copy
import logging
from typing import Any, Dict, List

import pandas as pd
from fastapi import APIRouter, HTTPException
from starlette import status

from tsfm_public.toolkit.util import select_by_index

from . import TSFM_ALLOW_LOAD_FROM_HF_HUB, TSFM_MODEL_DIR
from .constants import API_VERSION
from .errors import error_message
from .inference_payloads import ForecastingInferenceInput, ForecastingMetadataInput, PredictOutput
from .service_handler import ServiceHandler


LOGGER = logging.getLogger(__file__)


class InferenceRuntime:
    def __init__(self, config: Dict[str, Any] = {}):
        self.config = config
        # model_map = {}

        # if "custom_modules" in config:
        #     for custom_module in config["custom_modules"]:
        #         register_config(
        #             custom_module["model_type"],
        #             custom_module["model_config_name"],
        #             custom_module["module_path"],
        #         )
        #         LOGGER.info(f"registered {custom_module['model_type']}")

        #         model_map[custom_module["model_config_name"]] = custom_module["module_path"]

        # self.model_to_module_map = model_map

    def add_routes(self, app):
        self.router = APIRouter(prefix=f"/{API_VERSION}/inference", tags=["inference"])
        self.router.add_api_route(
            "/forecasting",
            self.forecast,
            methods=["POST"],
            response_model=PredictOutput,
        )
        app.include_router(self.router)

    # def load_preprocessor(self, model_path: str):
    #     # load preprocessor
    #     try:
    #         preprocessor = TimeSeriesPreprocessor.from_pretrained(model_path)
    #         LOGGER.info("Successfully loaded preprocessor")
    #     except OSError:
    #         preprocessor = None
    #         LOGGER.info("No preprocessor found")
    #     except Exception as ex:
    #         return None, ex

    #     return preprocessor, None

    # def load_config(self, model_path: str, **extra_config_kwargs: Dict[str, Any]):
    #     # load config, separate from load model, since we may need to inspect config first
    #     conf = load_config(model_path, **extra_config_kwargs)

    #     return conf

    # def load_model(self, model_path: str, config: PretrainedConfig):
    #     # load model
    #     model, ex = load_model(
    #         model_path,
    #         config=config,
    #         module_path=self.model_to_module_map.get(config.__class__.__name__, None),
    #     )
    #     if ex is not None:
    #         return None, ex

    #     LOGGER.info("Successfully loaded model")
    #     return model, None

    def forecast(self, input: ForecastingInferenceInput):
        LOGGER.info("calling forecast_common")
        answer, ex = self._forecast_common(input)

        if ex is not None:
            detail = error_message(ex)
            LOGGER.error(detail)
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)

        LOGGER.info("done, returning.")
        return answer

    def _forecast_common(self, input_payload: ForecastingInferenceInput) -> PredictOutput:
        model_path = TSFM_MODEL_DIR / input_payload.model_id

        if not model_path.is_dir():
            LOGGER.info(f"Could not find model at path: {model_path}")
            if TSFM_ALLOW_LOAD_FROM_HF_HUB:
                model_path = input_payload.model_id
                LOGGER.info(f"Using HuggingFace Hub: {model_path}")
            else:
                return None, RuntimeError(
                    f"Could not load model {input_payload.model_id} from {TSFM_MODEL_DIR.as_posix()}. If trying to load directly from the HuggingFace Hub please ensure that `TSFM_ALLOW_LOAD_FROM_HF_HUB=1`"
                )

        model, e = ServiceHandler.load(model_path)
        if e is not None:
            return None, e

        parameters = input_payload.parameters
        schema = input_payload.schema
        data = decode_data(input_payload.data, schema)
        future_data = decode_data(input_payload.future_data, schema)

        # collect and check underlying time series lengths
        if model.tsfm_config.get("minimum_context_length", None):
            if schema.id_columns:
                data_lengths = data.groupby(schema.id_columns).apply(len)
                min_len_index = data_lengths.argmin()
                min_data_length = data_lengths.iloc[min_len_index]
                max_data_length = data_lengths.max()
            else:
                min_data_length = max_data_length = len(data)
            LOGGER.info(f"Data length recieved {len(data)}, minimum series length: {min_data_length}")

            if min_data_length < model.tsfm_config["minimum_context_length"]:
                err_str = "Data should have time series of length that is at least the required model context length. "
                if schema.id_columns:
                    err_str += f"Received {min_data_length} time points for id {data_lengths.index[min_len_index]}, but model requires {model.tsfm_config['minimum_context_length']} time points"
                else:
                    err_str += f"Received {min_data_length} time points, but model requires {model.tsfm_config['minimum_context_length']} time points"

                return None, ValueError(err_str)

        # truncate data length
        if model.tsfm_config.get("maximum_context_length", None):
            if max_data_length > model.tsfm_config["maximum_context_length"]:
                LOGGER.info(f"Truncating series lengths to {model.tsfm_config['maximum_context_length']}")
                data = select_by_index(
                    data, id_columns=schema.id_columns, start_index=-model.tsfm_config["maximum_context_length"]
                )

        _, e = model.prepare(schema=schema, parameters=parameters)
        if e is not None:
            return None, e

        output, e = model.run(data=data, future_data=future_data, schema=schema, parameters=parameters)

        if e is not None:
            return None, e

        return output, None


def decode_data(data: Dict[str, List[Any]], schema: ForecastingMetadataInput) -> pd.DataFrame:
    if not data:
        return None

    df = pd.DataFrame.from_dict(data)
    if ts_col := schema.timestamp_column:
        df[ts_col] = pd.to_datetime(df[ts_col])

    sort_columns = copy.copy(schema.id_columns) if schema.id_columns else []

    if ts_col:
        sort_columns.append(ts_col)
    if sort_columns:
        return df.sort_values(sort_columns)

    return df
