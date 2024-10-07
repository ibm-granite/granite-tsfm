# Copyright contributors to the TSFM project
#
"""Primary entry point for inference services"""

import logging

import starlette.status as status
import yaml
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from . import (
    TSFM_CONFIG_FILE,
    TSFM_MODEL_DIR,
    TSFM_PYTHON_LOGGING_FORMAT,
    TSFM_PYTHON_LOGGING_LEVEL,
)
from .constants import API_VERSION
from .inference import InferenceRuntime


logging.basicConfig(
    format=TSFM_PYTHON_LOGGING_FORMAT,
    level=TSFM_PYTHON_LOGGING_LEVEL,
)

logging.info(f"Using TSFM_MODEL_DIR {TSFM_MODEL_DIR}")

if TSFM_CONFIG_FILE:
    with open(TSFM_CONFIG_FILE, "r") as file:
        config = yaml.safe_load(file)
else:
    config = {}


app = FastAPI(
    title="FM for Time Series API",
    version=API_VERSION,
    description="This FastAPI application provides service endpoints for performing inference tasks on TSFM HF models.",
)
ir = InferenceRuntime(config=config)
ir.add_routes(app)


@app.get("/")
def root():
    return RedirectResponse(url="/docs", status_code=status.HTTP_302_FOUND)


if __name__ == "__main__":
    # Third Party
    import uvicorn

    # Run the FastAPI application on the local host and port 7860
    # CMD["uvicorn", "app:app", "--host", "127.0.0.1", "--port", "7860", "--reload"]
    uvicorn.run(app, host="127.0.0.1", port=8000)
