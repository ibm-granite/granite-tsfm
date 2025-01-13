import json
import logging
import os
import sys

import torch
import yaml
from fastapi import HTTPException


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# logger.addHandler(logging.StreamHandler(sys.stdout))


# clue sagemaker runtime into some things
os.environ["TSFM_ALLOW_LOAD_FROM_HF_HUB"] = "0"
os.environ["TSFM_MODEL_DIR"] = os.path.join(os.path.dirname(__file__), "granite-tsfm/services/inference/mytest-tsfm")

# service components are not git installable directly
# so hack it a bit this way
custom_path = os.path.join(os.path.dirname(__file__), "granite-tsfm/services/inference")
if custom_path not in sys.path:
    sys.path.append(custom_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# currently we're targeting only cpu devices for inference as
# the tsfm models are compact and performant for CPU-only inference
device = "cpu"


# defining model and loading weights to it.
def model_fn(model_dir):
    logger.debug(f"in model_fn with {model_dir}")
    logger.debug("doing nothing, returning {}")
    return {}


# data preprocessing
def input_fn(request_body, request_content_type):
    assert request_content_type == "application/json"
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("in input_fn")
        logger.debug(f"request_content_type {request_content_type}")
        logger.debug(f"type of request_body {type(request_body)}")
        logger.debug(f"request_body {request_body[0:100]}...")
    return request_body


# inference
def predict_fn(input_object, model):
    from json import JSONDecodeError

    from pydantic import ValidationError
    from tsfminference import TSFM_CONFIG_FILE
    from tsfminference.inference import InferenceRuntime
    from tsfminference.inference_payloads import (
        ForecastingInferenceInput,
        PredictOutput,
    )

    if os.path.exists(TSFM_CONFIG_FILE):
        with open(TSFM_CONFIG_FILE, "r") as file:
            config = yaml.safe_load(file)
    else:
        config = {}
    logger.debug("in predict_fn")
    logger.debug(f"input_object type {type(input_object)}")
    logger.debug(f"model {model}")

    try:
        input: dict = json.loads(input_object)
        logger.debug(f"input is now of type f{type(input)}")
        inference_type = input.pop("inference_type")
        if not "forecasting" == inference_type:
            return (
                json.dumps(
                    {
                        "error_code": "ValidationError",
                        "error_message": f"inference_type {inference_type} is not supported.",
                    }
                ),
                "application/json",
                400,
            )

        input: ForecastingInferenceInput = ForecastingInferenceInput(**input)

        runtime: InferenceRuntime = InferenceRuntime(config=config)
        answer: PredictOutput = runtime.forecast(input=input)
        return answer
    except HTTPException as httpex:
        error_response = {"error_code": "HTTPException", "error_message": str(httpex)}
        return json.dumps(error_response), "application/json", 400
    except ValidationError as vex:
        error_response = {"error_code": "ValidationError", "error_message": str(vex)}
        return json.dumps(error_response), "application/json", 400
    except JSONDecodeError as jde:
        error_response = {"error_code": "JSONDecodeError", "error_message": str(jde)}
        return json.dumps(error_response), "application/json", 400
    except Exception as ex:
        error_response = {"error_code": "SERVER_ERROR", "error_message": str(ex)}
        return json.dumps(error_response), "application/json", 500


# postprocess
def output_fn(predictions, content_type):
    assert content_type == "application/json"
    from tsfminference.inference_payloads import (
        PredictOutput,
    )

    if isinstance(predictions, PredictOutput):
        return predictions.model_dump_json()
    else:
        return predictions
