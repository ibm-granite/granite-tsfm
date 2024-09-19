# Copyright contributors to the TSFM project
#
"""Tsfmfinetuning Runtime"""

import logging
import uuid
from typing import Any, Dict, List

import pandas as pd
from fastapi import APIRouter, HTTPException

from tsfm_public import TimeSeriesPreprocessor

from .constants import API_VERSION
from .finetuning_payloads import AsyncCallReturn, TinyTimeMixerForecastingTuneInput
from .util import load_config, load_model, register_config


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

    def finetuning(self, input: TinyTimeMixerForecastingTuneInput):
        try:
            LOGGER.info("calling forecast_common")
            answer = self._finetuning_common(input)
            LOGGER.info("done, returning.")
            return answer
        except Exception as e:
            LOGGER.exception(e)
            raise HTTPException(status_code=500, detail=repr(e))

    def _finetuning_common(self, input_payload: TinyTimeMixerForecastingTuneInput) -> AsyncCallReturn:
        # a no-op for now

        return AsyncCallReturn(job_id=uuid.uuid4().hex)


def decode_data(data: Dict[str, List[Any]], metadata: Dict[str, Any]) -> pd.DataFrame:
    if not data:
        return None

    df = pd.DataFrame.from_dict(data)
    if ts_col := metadata.timestamp_column:
        df[ts_col] = pd.to_datetime(df[ts_col])
    return df
