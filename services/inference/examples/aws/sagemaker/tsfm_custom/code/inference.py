import os
import sys
import json
import logging
import yaml
import torch

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# logger.addHandler(logging.StreamHandler(sys.stdout))


# clue sagemaker runtime into some things
os.environ["TSFM_ALLOW_LOAD_FROM_HF_HUB"] = "0"
os.environ["TSFM_MODEL_DIR"] = os.path.join(
    os.path.dirname(__file__), "granite-tsfm/services/inference/mytest-tsfm"
)

# service components are not git installable directly
# so hack it a bit this way
custom_path = os.path.join(os.path.dirname(__file__), "granite-tsfm/services/inference")
if custom_path not in sys.path:
    sys.path.append(custom_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# currently we're targeting only cpu devices for inference as
# the tsfm models are compact and performent for CPU-only inference
device = "cpu"


# defining model and loading weights to it.
def model_fn(model_dir):
    logger.debug(f"in model_fn with {model_dir}")
    logger.debug("doing nothing, returning None")
    return {}


# data preprocessing
def input_fn(request_body, request_content_type):
    assert request_content_type == "application/json"
    logger.debug("in input_fn")
    logger.debug(f"request_content_type {request_content_type}")
    logger.debug(f"type of request_body {type(request_body)}")
    logger.debug(f"request_body {request_body[0:100]}...")
    return request_body


# inference
def predict_fn(input_object, model):
    from tsfminference import TSFM_CONFIG_FILE
    from tsfminference.inference import InferenceRuntime
    from tsfminference.inference_payloads import (
        PredictOutput,
        ForecastingInferenceInput,
    )

    if os.path.exists(TSFM_CONFIG_FILE):
        with open(TSFM_CONFIG_FILE, "r") as file:
            config = yaml.safe_load(file)
    else:
        config = {}
    logger.debug("in predict_fn")
    logger.debug(f"input_object type {type(input_object)}")
    logger.debug(f"model {model}")

    input: dict = json.loads(input_object)
    logger.debug(f"input is now of type f{type(input)}")
    inference_type = input.pop("inference_type")

    if not "forecasting" == inference_type:
        raise NotImplementedError(f"infernce_type {inference_type} not supported.")

    input: ForecastingInferenceInput = ForecastingInferenceInput(**input)

    runtime: InferenceRuntime = InferenceRuntime(config=config)
    answer: PredictOutput = runtime.forecast(input=input)
    return answer


# postprocess
def output_fn(predictions, content_type):
    assert content_type == "application/json"
    return predictions.model_dump_json()
