# Copyright contributors to the TSFM project
#
"""Utilities to support services"""

from pathlib import Path
from typing import Optional, Union

from transformers import PreTrainedModel

from .time_series_preprocessor import TimeSeriesPreprocessor
from .tsfm_config import TSFMConfig


# tuples of:
# "inference_handler_module_path", "inference_handler_class_name",  "tuning_handler_module_path",  "tuning_handler_class_name")

service_handler_mapping = {
    "TinyTimeMixerConfig": (
        "tsfminference.tsfm_inference_handler",
        "TinyTimeMixerForecastingInferenceHandler",
        "tsfmfinetuning.tsfm_tuning_handler",
        "TinyTimeMixerForecastingTuningHandler",
    ),
    "PatchTSTConfig": (
        "tsfminference.tsfm_inference_handler",
        "ForecastingInferenceHandler",
        "tsfmfinetuning.tsfm_tuning_handler",
        "ForecastingTuningHandler",
    ),
    "PatchTSMixerConfig": (
        "tsfminference.tsfm_inference_handler",
        "ForecastingInferenceHandler",
        "tsfmfinetuning.tsfm_tuning_handler",
        "ForecastingTuningHandler",
    ),
}


def save_deployment_package(
    save_path: Union[Path, str],
    model: PreTrainedModel,
    ts_processor: Optional[TimeSeriesPreprocessor] = None,
    **kwargs,
):
    """Convenience function for saving the deployment package needed to use the services.

    Args:
        save_path (Union[Path, str]): Path to location to save files, a folder will be created at this path.
        model (PreTrainedModel): The model for which you wish to create the deployment package.
        ts_processor (Optional[TimeSeriesProcessor], optional): The time series processor used when training or
            finetuning the model. Defaults to None.

        Currently supports TinyTimeMixer, PatchTSMixer, PatchTST. Supported models need to have an entry in
        service_handler_mapping.

    Raises:
        ValueError: Raised if model is not one of the supported types.
    """
    if isinstance(save_path, str):
        save_path = Path(save_path)

    model.save_pretrained(save_path)

    if ts_processor is not None:
        ts_processor.save_pretrained(save_path)

    # now create and save tsfm config
    params = {}

    config_class = model.config.__class__.__name__
    handler_params = service_handler_mapping.get(config_class, None)

    if handler_params is None:
        raise ValueError(f"Could not find suitable handler information for config class {config_class}")

    (
        params["inference_handler_module_path"],
        params["inference_handler_class_name"],
        params["tuning_handler_module_path"],
        params["tuning_handler_class_name"],
    ) = handler_params

    params["model_config_name"] = config_class
    params["model_class_name"] = model.__class__.__name__
    params["model_type"] = model.config.model_type
    params["module_path"] = model.__class__.__module__  # "tsfm_public", maybe filter

    params["is_finetuned"] = ts_processor is not None

    # assumes we are dealing with one of the known IBM models
    params["minimum_context_length"] = kwargs.pop("minimum_context_length", model.config.context_length)
    params["maximum_context_length"] = kwargs.pop("maximum_context_length", model.config.context_length)
    params["maximum_prediction_length"] = kwargs.pop("maximum_prediction_length", model.config.prediction_length)
    params.update(**kwargs)

    svc_config = TSFMConfig(**params)

    svc_config.save_pretrained(save_path)
