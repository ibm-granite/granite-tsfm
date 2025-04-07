# Copyright contributors to the TSFM project
#
"""Utilities to support model loading"""

import enum
import logging
import os
from importlib import resources
from typing import Optional, Union

import numpy as np
import yaml
from transformers import PreTrainedModel

from tsfm_public.models.tinytimemixer import TinyTimeMixerConfig, TinyTimeMixerForPrediction
from tsfm_public.toolkit.time_series_preprocessor import DEFAULT_FREQUENCY_MAPPING


LOGGER = logging.getLogger(__file__)
LOGGER.setLevel(logging.INFO)

TTM_LOW_RESOLUTION_MODELS_MAX_CONTEXT = 512


class ForceReturn(enum.Enum):
    """`Enum` for the `force_return` parameter in the `get_model` function.

    "zeropad" = Returns a pre-trained TTM that has a context length higher than the input context length, hence,
        the user must apply zero-padding to use the returned model.
    "rolling" = Returns a pre-trained TTM that has a prediction length lower than the requested prediction length,
        hence, the user must apply rolling technique to use the returned model to forecast to the desired length.
        The `RecursivePredictor` class can be utilized in this scenario.
    "random_init_small" = Returns a randomly initialized small TTM which must be trained before performing inference.
    "random_init_medium" = Returns a randomly initialized medium TTM which must be trained before performing inference.
    "random_init_large" = Returns a randomly initialized large TTM which must be trained before performing inference.

    """

    ZEROPAD = "zeropad"
    ROLLING = "rolling"
    RANDOM_INIT_SMALL = "random_init_small"
    RANDOM_INIT_MEDIUM = "random_init_medium"
    RANDOM_INIT_LARGE = "random_init_large"


class ModelSize(enum.Enum):
    """`Enum` for the `size` parameter in the `get_random_ttm` function."""

    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


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


def get_random_ttm(
    context_length: int, prediction_length: int, size: str = ModelSize.SMALL.value, **kwargs
) -> PreTrainedModel:
    """Get a TTM with random weights.

    Args:
        context_length (int): Context length or history.
        prediction_length (int): Prediction length or forecast horizon.
        size (str, optional): Size of the desired TTM (small/medium/large). Defaults to "small".

    Raises:
        ValueError: If wrong size is provided.
        ValueError: Context length should be at least 4 if `size=small`,
            or at least 16 if `size=medium`,
            or at least 32 if `size=large`.

    Returns:
        PreTrainedModel: TTM model with randomly initialized weights.
    """
    if ModelSize.SMALL.value in size.lower():
        cl_lower_bound = 4
        apl = 0
    elif ModelSize.MEDIUM.value in size.lower():
        cl_lower_bound = 16
        apl = 3
    elif ModelSize.LARGE.value in size.lower():
        cl_lower_bound = 32
        apl = 5
    else:
        raise ValueError("Wrong size. Should be either of these [small/medium/large].")
    if context_length < cl_lower_bound:
        raise ValueError(f"Context length should be at least {cl_lower_bound} if `size={size}`.")

    cl = context_length if context_length % 2 == 0 else context_length - 1

    pl = 2
    while cl % pl == 0 and cl / pl >= 8:
        pl = pl * 2

    if ModelSize.SMALL.value in size.lower():
        d_model = 2 * pl
        num_layers = 3
    elif ModelSize.MEDIUM.value in size.lower():
        d_model = 16 * 2**apl
        num_layers = 3
    elif ModelSize.LARGE.value in size.lower():
        d_model = 16 * 2**apl
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
    model_path: str,
    model_name: str = "ttm",
    context_length: Optional[int] = None,
    prediction_length: Optional[int] = None,
    freq_prefix_tuning: bool = False,
    freq: Optional[str] = None,
    prefer_l1_loss: bool = False,
    prefer_longer_context: bool = True,
    force_return: Optional[str] = None,
    return_model_key: bool = False,
    **kwargs,
) -> Union[str, PreTrainedModel]:
    """TTM Model card offers a suite of models with varying `context_length` and `prediction_length` combinations.
    This wrapper automatically selects the right model based on the given input `context_length` and
    `prediction_length` abstracting away the internal complexity.

    Args:
        model_path (str): HuggingFace model card path or local model path (Ex. ibm-granite/granite-timeseries-ttm-r2)
        model_name (str, optional): Model name to use. Current allowed values: [ttm]. Defaults to "ttm".
        context_length (int, optional): Input Context length or history. Defaults to None.
        prediction_length (int, optional): Length of the forecast horizon. Defaults to None.
        freq_prefix_tuning (bool, optional): If true, it will prefer TTM models that are trained with frequency prefix
            tuning configuration. Defaults to None.
        freq (str, optional): Resolution or frequency of the data. Defaults to None. Allowed values are as per the
            `tsfm_public.toolkit.time_series_preprocessor.DEFAULT_FREQUENCY_MAPPING`.
            See this for details: https://github.com/ibm-granite/granite-tsfm/blob/main/tsfm_public/toolkit/time_series_preprocessor.py.
        prefer_l1_loss (bool, optional): If True, it will prefer choosing models that were trained with L1 loss or
            mean absolute error loss. Defaults to False.
        prefer_longer_context (bool, optional): If True, it will prefer selecting model with longer context/history
            Defaults to True.
        force_return (str, optional): This is used to force the get_model() to return a TTM model even when the provided
            configurations don't match with the existing TTMs. It gets the closest TTM possible. Allowed values are
            ["zeropad"/"rolling"/"random_init_small"/"random_init_medium"/"random_init_large"/`None`].
            "zeropad" = Returns a pre-trained TTM that has a context length higher than the input context length, hence,
            the user must apply zero-padding to use the returned model.
            "rolling" = Returns a pre-trained TTM that has a prediction length lower than the requested prediction length,
            hence, the user must apply rolling technique to use the returned model to forecast to the desired length.
            The `RecursivePredictor` class can be utilized in this scenario.
            "random_init_small" = Returns a randomly initialized small TTM which must be trained before performing inference.
            "random_init_medium" = Returns a randomly initialized medium TTM which must be trained before performing inference.
            "random_init_large" = Returns a randomly initialized large TTM which must be trained before performing inference.
            `None` = `force_return` is disable. Raises an error if no suitable model is found.
            Defaults to None.
        return_model_key (bool, optional): If True, only the TTM model name will be returned, instead of the actual model.
            This does not downlaod the model, and only returns the name of the suitable model. Defaults to False.

    Returns:
        Union[str, PreTrainedModel]: Returns the Model, or the model name.
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

            # Get freq
            R = DEFAULT_FREQUENCY_MAPPING.get(freq, 0)

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

            # Step 1: Filter models based on freq (R)
            if model_path_type == 1 or model_path_type == 2:
                # Only, r2.1 models are suitable for Daily or longer freq
                if R >= 8:
                    models = [m for m in available_models.keys() if "r2.1" in available_models[m]["release"]]
                else:
                    models = list(available_models.keys())
            else:
                models = list(available_models.keys())

            # Step 2: Filter models by context length constraint
            # Choose all models which have lower context length than
            # the input available length
            selected_models_ = []
            if context_length < shortest_ttm_context_length:
                if force_return is None:
                    raise ValueError(
                        "Requested context length is less than the "
                        f"shortest context length for TTMs: {shortest_ttm_context_length}. "
                        "Set `force_return=zeropad` to get a TTM with longer context."
                    )
                elif force_return == ForceReturn.ZEROPAD.value:  # force_return.startswith("zero"):
                    # Keep all models. Zero-padding must be done outside.
                    selected_models_ = models
                    LOGGER.warning(
                        "Requested `context_length` is shorter than the "
                        "shortest TTM available, hence, zero-padding must "
                        "be done outside."
                    )
            else:
                lowest_context_length = np.inf
                shortest_context_models = []
                for m in models:
                    if available_models[m]["context_length"] <= context_length:
                        selected_models_.append(m)
                    if available_models[m]["context_length"] <= lowest_context_length:
                        lowest_context_length = available_models[m]["context_length"]
                        shortest_context_models.append(m)

            if len(selected_models_) == 0:
                if force_return is None:
                    raise ValueError(
                        "Could not find a TTM with `context_length` shorter "
                        f"than the requested context length = {context_length}. "
                        "Set `force_return=zeropad` to get a TTM with longer context."
                    )
                elif force_return == ForceReturn.ZEROPAD.value:  # force_return.startswith("zero"):
                    selected_models_ = shortest_context_models
                    LOGGER.warning(
                        "Could not find a TTM with `context_length` shorter "
                        f"than the requested context length = {context_length}. "
                        f"Hence, selecting TTMs with shortest context available = {lowest_context_length}."
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
                if len(selected_models_) == 0:
                    if force_return is None:
                        raise ValueError(
                            "Could not find a TTM with `prediction_length` higher "
                            f"than the requested prediction length = {prediction_length}. "
                            "Set `force_return=rolling` to get a TTM with shorted prediction "
                            "length. Rolling must be done outside."
                        )
                    elif force_return == ForceReturn.ROLLING.value:  # force_return.startswith("roll"):
                        selected_models_.append(highest_prediction_model)
                        LOGGER.warning(
                            "Could not find a TTM with `prediction_length` higher "
                            f"than the request prediction length = {prediction_length}. "
                            f"Hence, returning the TTM with highest prediction length {highest_prediction_length} "
                            f"satisfying the requested context length. Rolling must be done outside."
                        )
                models = selected_models_

            # Step 7: Do not allow unknow frequency
            if freq_prefix_tuning and (freq is not None) and (freq not in DEFAULT_FREQUENCY_MAPPING.keys()):
                LOGGER.warning(
                    f"The specified frequency ({freq}) is not in the set of "
                    f"allowed resolutions: {list(DEFAULT_FREQUENCY_MAPPING.keys())}."
                )
                models = []

            # Step 8: Return the first available model or a dummy model if none found
            if len(models) == 0:
                if force_return is None:
                    raise ValueError(
                        "No suitable pre-trained TTM was found! Set `force_return` to "
                        "random_init_small/random_init_medium/random_init_large "
                        "to get a randomly initialized TTM of size small/medium/large "
                        "respectively."
                    )
                elif force_return in [
                    ForceReturn.RANDOM_INIT_SMALL.value,
                    ForceReturn.RANDOM_INIT_MEDIUM.value,
                    ForceReturn.RANDOM_INIT_LARGE.value,
                ]:  # "sma" in force_return.lower() or "med" in force_return.lower() or "lar" in force_return.lower():
                    LOGGER.info(
                        "Trying to build a TTM with random weights since no "
                        "suitable pre-trained TTM could be found. You must "
                        "train this TTM before using it for inference."
                    )
                    model = get_random_ttm(context_length, prediction_length, size=force_return)
                    LOGGER.info(
                        "Returning a randomly initialized TTM with context length "
                        f"= {model.config.context_length}, prediction length "
                        f"= {model.config.prediction_length}."
                    )
                    if return_model_key:
                        model_key = force_return.split("_")[-1]
                        return f"TTM({model_key})"
                    else:
                        return model
                else:
                    raise ValueError(
                        "Could not find a suitable TTM for the given "
                        f"context_length = {context_length}, and "
                        f"prediction_length = {prediction_length}. "
                        "Check the model card for more information. "
                        "set `force_return` properly (see the docstrings) "
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

        if return_model_key:
            return model_key

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

    return model
