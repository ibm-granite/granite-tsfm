# Copyright contributors to the TSFM project
#
"""Tsfminference Runtime"""

import copy
import logging
import os
from typing import Any, Dict, List

import pandas as pd
from fastapi import APIRouter, HTTPException
from prometheus_client import Histogram
from starlette import status

from tsfm_public.toolkit.util import select_by_index

from . import TSFM_ALLOW_LOAD_FROM_HF_HUB, TSFM_MODEL_DIR
from .constants import API_VERSION
from .dataframe_checks import check
from .errors import error_message
from .inference_payloads import ForecastingInferenceInput, ForecastingMetadataInput, PredictOutput
from .service_handler import ServiceHandler


LOGGER = logging.getLogger(__file__)

FORECAST_PROMETHEUS_TIME_SPENT = Histogram("forecast_time_spent", "Wall clock time histogram.")
FORECAST_PROMETHEUS_CPU_USED = Histogram("forecast_cpu_user", "CPU user time histogram.")


class InferenceRuntime:
    def __init__(self, config: Dict[str, Any] = {}):
        # to do: assess the need for config
        self.config = config

    def add_routes(self, app):
        self.router = APIRouter(prefix=f"/{API_VERSION}/inference", tags=["inference"])
        self.router.add_api_route(
            "/forecasting",
            self.forecast,
            methods=["POST"],
            response_model=PredictOutput,
        )
        app.include_router(self.router)

    def forecast(self, input: ForecastingInferenceInput):
        LOGGER.info("calling forecast_common")
        user_start_time = os.times().user
        st = datetime.datetime.now()
        answer, ex = self._forecast_common(input)
        td: datetime.timedelta = datetime.datetime.now() - st
        FORECAST_PROMETHEUS_TIME_SPENT.observe(td.total_seconds())
        FORECAST_PROMETHEUS_CPU_USED.observe(os.times().user - user_start_time)

        if ex is not None:
            detail = error_message(ex)
            LOGGER.exception(detail)
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

        handler, e = ServiceHandler.load(model_path)
        if e is not None:
            return None, e

        parameters = input_payload.parameters
        schema = input_payload.schema
        data = decode_data(input_payload.data, schema)
        future_data = decode_data(input_payload.future_data, schema)

        # collect and check underlying time series lengths
        if handler.tsfm_config.get("minimum_context_length", None):
            if schema.id_columns:
                data_lengths = data.groupby(schema.id_columns).apply(len)
                min_len_index = data_lengths.argmin()
                min_data_length = data_lengths.iloc[min_len_index]
                max_data_length = data_lengths.max()
            else:
                min_data_length = max_data_length = len(data)
            LOGGER.info(f"Data length recieved {len(data)}, minimum series length: {min_data_length}")

            if min_data_length < handler.tsfm_config["minimum_context_length"]:
                err_str = "Data should have time series of length that is at least the required model context length. "
                if schema.id_columns:
                    err_str += f"Received {min_data_length} time points for id {data_lengths.index[min_len_index]}, but model requires {handler.tsfm_config['minimum_context_length']} time points"
                else:
                    err_str += f"Received {min_data_length} time points, but model requires {handler.tsfm_config['minimum_context_length']} time points"

                return None, ValueError(err_str)

        # truncate data length
        if handler.tsfm_config.get("maximum_context_length", None):
            if max_data_length > handler.tsfm_config["maximum_context_length"]:
                LOGGER.info(f"Truncating series lengths to {handler.tsfm_config['maximum_context_length']}")
                data = select_by_index(
                    data, id_columns=schema.id_columns, start_index=-handler.tsfm_config["maximum_context_length"]
                )

        _, e = handler.prepare(schema=schema, parameters=parameters)
        if e is not None:
            return None, e

        output, e = handler.run(data=data, future_data=future_data, schema=schema, parameters=parameters)

        if e is not None:
            return None, e

        return output, None


def decode_data(data: Dict[str, List[Any]], schema: ForecastingMetadataInput) -> pd.DataFrame:
    if not data:
        return None, None

    df = pd.DataFrame.from_dict(data)
    rc, msg = check(df, schema)

    if rc != 0:
        return None, ValueError(msg)

    if ts_col := schema.timestamp_column:
        df[ts_col] = pd.to_datetime(df[ts_col])

    sort_columns = copy.copy(schema.id_columns) if schema.id_columns else []

    if ts_col:
        sort_columns.append(ts_col)
    if sort_columns:
        return df.sort_values(sort_columns), None

    return df, None
