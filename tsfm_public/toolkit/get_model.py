# Copyright contributors to the TSFM project
#
"""Utilities to support model loading"""

import logging
from importlib import resources

import numpy as np
import yaml

from tsfm_public.models.tinytimemixer import TinyTimeMixerConfig, TinyTimeMixerForPrediction
from tsfm_public.toolkit.time_series_preprocessor import DEFAULT_FREQUENCY_MAPPING


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

TTM_LOW_RESOLUTION_MODELS_MAX_CONTEXT = 512


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
    elif "ibm-research/ttm-research-r2" in model_path:
        return 3
    else:
        return 0


def get_random_ttm(context_length: int, prediction_length: int, size: str = "small", **kwargs):
    if size.lower().startswith("s"):
        cl_lower_bound = 4
        apl = 0
    elif size.lower().startswith("m"):
        cl_lower_bound = 16
        apl = 3
    elif size.lower().startswith("l"):
        cl_lower_bound = 32
        apl = 5
    else:
        raise ValueError("Wrong size. Should be either of these [small/medium/large].")
    if context_length < cl_lower_bound:
        raise ValueError("Context length should be at least 16 if" " `return_random_model_if_not_available=medium`.")

    cl = context_length if context_length % 2 == 0 else context_length - 1

    pl = 2
    while cl % pl == 0 and cl / pl >= 8:
        pl = pl * 2

    if size.lower().startswith("s"):
        d_model = 2 * pl
        num_layers = 3
    elif size.lower().startswith("m"):
        d_model = 16 * 2 ^ (apl - 1)
        num_layers = 3
    elif size.lower().startswith("l"):
        d_model = 16 * 2 ^ (apl - 1)
        num_layers = 5
    else:
        raise ValueError("Wrong size. Should be either of these [small/medium/large].")
    ttm_config = TinyTimeMixerConfig(
        context_length=cl,
        prediction_length=prediction_length,
        patch_length=pl,
        patch_stride=pl,
        d_model=d_model,
        num_layers=num_layers,
        decoder_num_layers=2,
        decoder_d_model=d_model,
        adaptive_patching_levels=apl,
        dropout=0.2,
        **kwargs,
    )
    model = TinyTimeMixerForPrediction(config=ttm_config)

    return model


def get_model(
    model_path,
    model_name: str = "ttm",
    context_length: int = None,
    prediction_length: int = None,
    freq_prefix_tuning: bool = None,
    resolution=None,
    prefer_l1_loss=False,
    prefer_longer_context=True,
    force_return: bool = True,
    return_random_model_if_not_available: str = "small",
    return_model_key: bool = False,
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
            For ibm-granite/granite-timeseries-ttm-r2 and  ibm-research/ttm-research-r2, we allow 512, 1024 and 1536
        prediction_length (int):
            Forecast length to predict. For ibm-granite/granite-timeseries-ttm-r1, we can forecast upto 96.
            For ibm-granite/granite-timeseries-ttm-r2 and  ibm-research/ttm-research-r2, we can forecast upto 720.
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

            # Get resolution
            R = DEFAULT_FREQUENCY_MAPPING.get(resolution, 0)

            # Get list of all TTM models
            config_dir = resources.files("tsfm_public.resources.model_paths_config")
            with open(os.path.join(config_dir, "ttm.yaml"), "r") as file:
                model_revisions = yaml.safe_load(file)

            if model_path_type == 1 or model_path_type == 2:
                available_models = model_revisions["ibm-granite-models"]
                filtered_models = {}
                if model_path_type == 1:
                    for k in available_models.keys():
                        if available_models[k]["release"].startswith("r1"):
                            filtered_models[k] = available_models[k]
                if model_path_type == 2:
                    for k in available_models.keys():
                        if available_models[k]["release"].startswith("r2"):
                            filtered_models[k] = available_models[k]
                available_models = filtered_models
            else:
                available_models = model_revisions["research-use-models"]

            # Calculate shortest TTM context length, will be needed later
            available_model_keys = list(available_models.keys())
            available_ttm_context_lengths = [available_models[m]["context_length"] for m in available_model_keys]
            shortest_ttm_context_length = min(available_ttm_context_lengths)

            # Step 1: Filter models based on resolution (R)
            if model_path_type == 1 or model_path_type == 2:
                # Only, r2.1 models are suitable for Daily or longer resolution
                if R >= 8:
                    models = [m for m in available_models.keys() if "r2.1" in available_models[m]["release"]]
                else:
                    models = list(available_models.keys())
            else:
                models = list(available_models.keys())

            # Step 2: Filter models by context length constraint
            # Choose all models which have lower context length than
            # the input available length
            if context_length < shortest_ttm_context_length and force_return:
                # Keep all models. Zero-padding must be done outside.
                selected_models_ = models
                LOGGER.warning(
                    "Requested `context_length` is shorter than the "
                    "shortest TTM available, hence, zero-padding must "
                    "be done outside."
                )
            else:
                selected_models_ = []
                lowest_context_length = np.inf
                shortest_context_models = []
                for m in models:
                    if available_models[m]["context_length"] <= context_length:
                        selected_models_.append(m)
                    if available_models[m]["context_length"] <= lowest_context_length:
                        lowest_context_length = available_models[m]["context_length"]
                        shortest_context_models.append(m)

            if len(selected_models_) == 0 and force_return:
                selected_models_ = shortest_context_models
                LOGGER.warning(
                    "Could not find a TTM with `context_length` shorter "
                    f"than the requested context length = {context_length}. "
                    f"Hence, Returning TTM with shortest context available = {lowest_context_length}."
                )
            models = selected_models_

            # Step 3: Apply L1 and FT preferences only when context_length <= 512
            if len(models) > 0:
                if prefer_longer_context:
                    reference_context = min(
                        context_length, max([available_models[m]["context_length"] for m in models])
                    )
                else:
                    reference_context = min([available_models[m]["context_length"] for m in models])
                if reference_context <= TTM_LOW_RESOLUTION_MODELS_MAX_CONTEXT:
                    # Step 3a: Filter based on L1 preference
                    if prefer_l1_loss:
                        l1_models = [m for m in models if "-l1-" in m]
                        if l1_models:
                            models = l1_models

                    # Step 3b: Filter based on frequency tuning indicator preference
                    if freq_prefix_tuning:
                        ft_models = [m for m in models if "-ft-" in m]
                        if ft_models:
                            models = ft_models

            # Step 4: Sort models by context length (descending if prefer_longer_context else ascending)
            # Step 5: Sub-sort for each context length by forecast length in ascending order
            if len(models) > 0:
                sign = -1 if prefer_longer_context else 1
                models = sorted(
                    models,
                    key=lambda m: (
                        sign * int(available_models[m]["context_length"]),
                        int(available_models[m]["prediction_length"]),
                    ),
                )

            # Step 6: Remove models whose forecast length is less than input forecast length
            # Because, this needs recursion which has to be handled outside this get_model() utility
            if len(models) > 0:
                selected_models_ = []
                highest_prediction_length = -np.inf
                highest_prediction_model = None
                for m in models:
                    if int(available_models[m]["prediction_length"]) >= prediction_length:
                        selected_models_.append(m)
                    if available_models[m]["prediction_length"] > highest_prediction_length:
                        highest_prediction_length = available_models[m]["prediction_length"]
                        highest_prediction_model = m
                if len(selected_models_) == 0 and force_return:
                    selected_models_.append(highest_prediction_model)
                    LOGGER.warning(
                        "Could not find a TTM with `prediction_length` higher "
                        f"than the request prediction length = {prediction_length}. "
                        "Hence, returning the TTM with highest prediction length {highest_prediction_length}"
                        f"satisfying the requested context length."
                    )
                models = selected_models_

            # Step 7: Do not allow unknow frequency
            if (
                freq_prefix_tuning
                and (resolution is not None)
                and (resolution not in DEFAULT_FREQUENCY_MAPPING.keys())
            ):
                models = []

            # Step 8: Return the first available model or a dummy model if none found
            if len(models) == 0:
                if (
                    return_random_model_if_not_available.lower().startswith("s")
                    or return_random_model_if_not_available.lower().startswith("m")
                    or return_random_model_if_not_available.lower().startswith("l")
                ):
                    LOGGER.info(
                        "Trying to build a TTM with random weights since no "
                        "suitable pre-trained TTM could be found. You must "
                        "train this TTM before using it for inference."
                    )
                    model = get_random_ttm(
                        context_length, prediction_length, size=return_random_model_if_not_available
                    )
                    LOGGER.info(
                        "Returning a randomly initialized TTM with context length "
                        f"= {model.config.context_length}, prediction length "
                        f"= {model.config.prediction_length}."
                    )
                    if return_model_key:
                        return model, f"TTM({return_random_model_if_not_available})"
                    else:
                        return model
                else:
                    raise ValueError(
                        "Could not find a suitable TTM for the given"
                        f"context_length = {context_length}, and"
                        f"prediction_length = {prediction_length}."
                        "Check the model card for more information."
                        "set `return_random_model_if_not_available` properly"
                        "if you want to get a randomly initialized TTM."
                    )
            else:
                model_key = models[0]

            # selected_context_length = available_models[model_key]["context_length"]
            selected_prediction_length = available_models[model_key]["prediction_length"]
            if selected_prediction_length > prediction_length:
                prediction_filter_length = prediction_length
                LOGGER.warning(
                    f"Requested `prediction_length` ({prediction_length}) is not exactly "
                    "equal to any of the available TTM prediction lengths. "
                    "Hence, TTM will forecast using the `prediction_filter_length` "
                    "argument to provide the requested prediction length. "
                    "Check the model card to know more about the supported context lengths "
                    "and forecast/prediction lengths."
                )

            if selected_prediction_length < prediction_length:
                LOGGER.warning(
                    "Selected `prediction_length` is shorter than the requested "
                    "length since no suitable model could be found. You can use "
                    " `RecursivePredictor` for forecast to the desired length."
                )

            ttm_model_revision = available_models[model_key]["revision"]

        else:
            prediction_filter_length = prediction_length

        # Load model
        model = TinyTimeMixerForPrediction.from_pretrained(
            model_path,
            revision=ttm_model_revision,
            prediction_filter_length=prediction_filter_length,
            **kwargs,
        )

        LOGGER.info(f"Model loaded successfully from {model_path}, revision = {ttm_model_revision}.")
        LOGGER.info(
            f"[TTM] context_length = {model.config.context_length}, prediction_length = {model.config.prediction_length}"
        )
    else:
        raise ValueError("Currently supported values for `model_name` = 'ttm'.")

    if return_model_key:
        return model, model_key
    else:
        return model


def get_model_deprecated(
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
            For ibm-granite/granite-timeseries-ttm-r2 and  ibm-research/ttm-research-r2, we allow 512, 1024 and 1536
        prediction_length (int):
            Forecast length to predict. For ibm-granite/granite-timeseries-ttm-r1, we can forecast upto 96.
            For ibm-granite/granite-timeseries-ttm-r2 and  ibm-research/ttm-research-r2, we can forecast upto 720.
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

            with config_dir.joinpath("ttm.yaml").open("r") as file:
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
