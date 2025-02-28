# Copyright contributors to the TSFM project
#
"""Tsfmfinetuning Runtime"""

import logging
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import pandas as pd
from fastapi import APIRouter, HTTPException
from starlette import status
from transformers import set_seed

from tsfm_public import TimeSeriesPreprocessor

from . import TSFM_ALLOW_LOAD_FROM_HF_HUB, TSFM_MODEL_DIR
from .constants import API_VERSION
from .dirutil import resolve_model_path
from .ftpayloads import (
    AsyncCallReturn,
    BaseTuneInput,
    TinyTimeMixerForecastingTuneInput,
)
from .hfutil import load_config, load_model, register_config
from .ioutils import to_pandas
from .tuning_handler import TuningHandler


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

        model, ex = load_model(
            model_path,
            config=conf,
            module_path=self.model_to_module_map.get(conf.__class__.__name__, None),
        )
        if ex is None:
            LOGGER.info("Successfully loaded model")
        else:
            LOGGER.exception(ex)
        return model, preprocessor, ex

    def finetuning(self, input: TinyTimeMixerForecastingTuneInput, tuned_model_name: str, output_dir: Path):
        answer, ex = self._finetuning_common(input, tuned_model_name=tuned_model_name, tmp_dir=output_dir)

        if ex is not None:
            import traceback

            traceback.print_exception(ex)
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=repr(ex))

        return answer

    def _finetuning_common(
        self, input_payload: BaseTuneInput, tuned_model_name: str, tmp_dir: Path
    ) -> Tuple[Path, Union[Exception, None]]:
        LOGGER.info("in _forecasting_tuning_workflow")

        # set seed, must be done before model load
        if input_payload.parameters.random_seed:
            set_seed(input_payload.parameters.random_seed)

        model_path = resolve_model_path(TSFM_MODEL_DIR, input_payload.model_id)

        if not model_path:
            LOGGER.info(f"Could not find model at path: {model_path}")
            if TSFM_ALLOW_LOAD_FROM_HF_HUB:
                model_path = input_payload.model_id
                LOGGER.info(f"Using HuggingFace Hub: {model_path}")
            else:
                return None, RuntimeError(
                    f"Could not load model {input_payload.model_id} from {TSFM_MODEL_DIR}. If trying to load directly from the HuggingFace Hub please ensure that `TSFM_ALLOW_LOAD_FROM_HF_HUB=1`"
                )

        handler, e = TuningHandler.load(model_id=input_payload.model_id, model_path=model_path)
        if e is not None:
            return None, e

        parameters = input_payload.parameters
        schema = input_payload.schema

        data: pd.DataFrame = to_pandas(uri=input_payload.data, **schema.model_dump())

        # validation_data = FinetuningRuntime._validation_data(input)

        _, e = handler.prepare(data=data, schema=schema, parameters=parameters)
        if e is not None:
            return None, e

        output, e = handler.train(
            data=data, schema=schema, parameters=parameters, tuned_model_name=tuned_model_name, tmp_dir=tmp_dir
        )
        if e is not None:
            return None, e

        return output, None
