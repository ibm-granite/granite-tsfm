"""Primary entry point for inference services"""

import logging

import yaml
from fastapi import FastAPI

from tsfmservices import (
    TSFM_CONFIG_FILE,
    TSFM_PYTHON_LOGGING_FORMAT,
    TSFM_PYTHON_LOGGING_LEVEL,
)
from tsfmservices.common.constants import API_VERSION
from tsfmservices.inference import InferenceRuntime


logging.basicConfig(
    format=TSFM_PYTHON_LOGGING_FORMAT,
    level=TSFM_PYTHON_LOGGING_LEVEL,
)


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
    # Endpoint at the root URL ("/") returns a welcome message with a clickable link
    message = "Welcome. Go to /docs to access the API documentation."
    return {"message": message}


if __name__ == "__main__":
    # Third Party
    import uvicorn

    # Run the FastAPI application on the local host and port 7860
    # CMD["uvicorn", "app:app", "--host", "127.0.0.1", "--port", "7860", "--reload"]
    uvicorn.run(app, host="127.0.0.1", port=8000)
