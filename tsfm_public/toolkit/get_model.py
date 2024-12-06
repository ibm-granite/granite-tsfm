# Copyright contributors to the TSFM project
#
"""Utilities to support model loading"""

import logging
import os
from importlib import resources

import yaml

from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction


LOGGER = logging.getLogger(__file__)

SUPPORTED_LENGTHS = {
    1: {"CL": [512, 1024], "FL": [96]},
    2: {
        "CL": [512, 1024, 1536],
        "FL": [96, 192, 336, 720],
    },
    3: {
        "CL": [512, 1024, 1536],
        "FL": [96, 192, 336, 720],
    },
}


def check_ttm_model_path(model_path):
    if (
        "ibm/TTM" in model_path
        or "ibm-granite/granite-timeseries-ttm-r1" in model_path
        or "ibm-granite/granite-timeseries-ttm-v1" in model_path
        or "ibm-granite/granite-timeseries-ttm-1m" in model_path
    ):
        return 1
    elif "ibm-granite/granite-timeseries-ttm-r2" in model_path:
        return 2
    elif "ibm/ttm-research-r2" in model_path:
        return 3
    else:
        return 0


def get_model(
    model_path,
    model_name: str = "ttm",
    context_length: int = None,
    prediction_length: int = None,
    freq_prefix_tuning: bool = None,
    force_return: bool = True,
    **kwargs,
):
    """
    TTM Model card offers a suite of models with varying context_length and forecast_length combinations.
    This wrapper automatically selects the right model based on the given input context_length and prediction_length abstracting away the internal
    complexity.

    Args:
        model_path (str):
            HF model card path or local model path (Ex. ibm-granite/granite-timeseries-ttm-r1)
        model_name (*optional*, str)
            model name to use. Allowed values: ttm
        context_length (int):
            Input Context length. For ibm-granite/granite-timeseries-ttm-r1, we allow 512 and 1024.
            For ibm-granite/granite-timeseries-ttm-r2 and  ibm/ttm-research-r2, we allow 512, 1024 and 1536
        prediction_length (int):
            Forecast length to predict. For ibm-granite/granite-timeseries-ttm-r1, we can forecast upto 96.
            For ibm-granite/granite-timeseries-ttm-r2 and  ibm/ttm-research-r2, we can forecast upto 720.
            Model is trained for fixed forecast lengths (96,192,336,720) and this model add required `prediction_filter_length` to the model instance for required pruning.
            For Ex. if we need to forecast 150 timepoints given last 512 timepoints using model_path = ibm-granite/granite-timeseries-ttm-r2, then get_model will select the
            model from 512_192_r2 branch and applies prediction_filter_length = 150 to prune the forecasts from 192 to 150. prediction_filter_length also applies loss
            only to the pruned forecasts during finetuning.
        freq_prefix_tuning (*optional*, bool):
            Future use. Currently do no use this parameter.
        kwargs:
            Pass all the extra fine-tuning model parameters intended to be passed in the from_pretrained call to update model configuration.
    """
    LOGGER.info(f"Loading model from: {model_path}")

    if model_name.lower() == "ttm":
        model_path_type = check_ttm_model_path(model_path)
        prediction_filter_length = None
        ttm_model_revision = None
        if model_path_type != 0:
            if context_length is None or prediction_length is None:
                raise ValueError(
                    "Provide `context_length` and `prediction_length` when `model_path` is a hugginface model path."
                )

            # Get right TTM model
            config_dir = resources.files("tsfm_public.resources.model_paths_config")

            with open(os.path.join(config_dir, "ttm.yaml"), "r") as file:
                model_revisions = yaml.safe_load(file)

            max_supported_horizon = SUPPORTED_LENGTHS[model_path_type]["FL"][-1]
            if prediction_length > max_supported_horizon:
                if force_return:
                    selected_prediction_length = max_supported_horizon
                    LOGGER.warning(
                        f"The requested forecast horizon is greater than the maximum supported horizon ({max_supported_horizon}). Returning TTM model with horizon {max_supported_horizon} since `force_return=True`."
                    )
                else:
                    raise ValueError(f"Currently supported maximum prediction_length = {max_supported_horizon}")
            else:
                for h in SUPPORTED_LENGTHS[model_path_type]["FL"]:
                    if prediction_length <= h:
                        selected_prediction_length = h
                        break

            LOGGER.info(f"Selected TTM `prediction_length` = {selected_prediction_length}")

            if selected_prediction_length > prediction_length:
                prediction_filter_length = prediction_length
                LOGGER.warning(
                    f"Requested `prediction_length` ({prediction_length}) is not exactly equal to any of the available TTM prediction lengths. Hence, TTM will forecast using the `prediction_filter_length` argument to provide the requested prediction length. Supported context lengths (CL) and forecast/prediction lengths (FL) for Model Card: {model_path} are {SUPPORTED_LENGTHS[model_path_type]}"
                )

            # Choose closest context length
            available_context_lens = sorted(SUPPORTED_LENGTHS[model_path_type]["CL"], reverse=True)
            selected_context_length = None
            for cl in available_context_lens:
                if cl <= context_length:
                    selected_context_length = cl
                    if cl < context_length:
                        LOGGER.warning(
                            f"Selecting TTM context length ({selected_context_length}) < Requested context length ({context_length} since exact match was not found.)"
                        )
                    break
            if selected_context_length is None:
                if force_return:
                    selected_context_length = available_context_lens[-1]
                    LOGGER.warning(
                        f"Requested context length is too short. Requested = {context_length}. Available lengths for model_type = {model_path_type} are: {available_context_lens}. Returning the shortest context length model possible since `force_return=True`. Data needs to be handled properly, and it can affect the performance!"
                    )
                else:
                    raise ValueError(
                        f"Requested context length is too short. Requested = {context_length}. Available lengths for model_type = {model_path_type} are: {available_context_lens}. To return the shortest context length model possible, set `force_return=True`."
                    )

            if freq_prefix_tuning is None:
                # Default model preference (freq / nofreq)
                if model_path_type == 1 or model_path_type == 2:  # for granite use nofreq models
                    freq_prefix = "nofreq"
                elif model_path_type == 3:  # for research-use use freq models
                    freq_prefix = "freq"
                else:
                    freq_prefix = None
            else:
                raise ValueError(
                    "In the current implementation, set `freq_prefix_tuning` to None for automatic model selection accordingly."
                )
                if freq_prefix_tuning:
                    freq_prefix = "freq"
                else:
                    freq_prefix = "nofreq"

            try:
                if model_path_type == 1 or model_path_type == 2:
                    ttm_model_revision = model_revisions["ibm-granite-models"][
                        f"r{model_path_type}-{selected_context_length}-{selected_prediction_length}-{freq_prefix}"
                    ]["revision"]
                elif model_path_type == 3:
                    ttm_model_revision = model_revisions["research-use-models"][
                        f"r2-{selected_context_length}-{selected_prediction_length}-{freq_prefix}"
                    ]["revision"]
                else:
                    raise Exception(
                        "Wrong model path type calculation. Possible reason: the model card path is wrong."
                    )
            except KeyError:
                raise ValueError(
                    f"Model not found, possibly because of wrong context_length. Supported context lengths (CL) and forecast/prediction lengths (FL) for Model Card: {model_path} are {SUPPORTED_LENGTHS[model_path_type]}"
                )
        else:
            prediction_filter_length = prediction_length

        # Load model
        model = TinyTimeMixerForPrediction.from_pretrained(
            model_path,
            revision=ttm_model_revision,
            prediction_filter_length=prediction_filter_length,
            **kwargs,
        )

        LOGGER.info("Model loaded successfully!")
        LOGGER.info(
            f"[TTM] context_len = {model.config.context_length}, forecast_len = {model.config.prediction_length}"
        )
    else:
        raise ValueError("Currently supported values for `model_name` = 'ttm'.")

    return model
