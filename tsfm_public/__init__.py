# Copyright contributors to the TSFM project
#

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

# Check the dependencies satisfy the minimal versions required.
from transformers.utils import _LazyModule

from tsfm_public.toolkit.hf_util import register_config

from .version import __version__, __version_tuple__


TSFM_PYTHON_LOGGING_LEVEL = os.getenv("TSFM_PYTHON_LOGGING_LEVEL", "INFO")

LevelNamesMapping = {
    "INFO": logging.INFO,
    "WARN": logging.WARN,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
    "DEBUG": logging.DEBUG,
    "FATAL": logging.FATAL,
}

TSFM_PYTHON_LOGGING_LEVEL = (
    logging.getLevelNamesMapping()[TSFM_PYTHON_LOGGING_LEVEL]
    if hasattr(logging, "getLevelNamesMapping")
    else LevelNamesMapping[TSFM_PYTHON_LOGGING_LEVEL]
)
TSFM_PYTHON_LOGGING_FORMAT = os.getenv(
    "TSFM_PYTHON_LOGGING_FORMAT",
    "%(levelname)s:p-%(process)d:t-%(thread)d:%(filename)s:%(funcName)s:%(message)s",
)

logging.basicConfig(
    format=TSFM_PYTHON_LOGGING_FORMAT,
    level=TSFM_PYTHON_LOGGING_LEVEL,
)

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# Base objects, independent of any specific backend
_import_structure = {
    "models": [],
    "models.tinytimemixer": ["TINYTIMEMIXER_PRETRAINED_CONFIG_ARCHIVE_MAP", "TinyTimeMixerConfig"],
    "models.tspulse": ["TINYTIMEMIXER_PRETRAINED_CONFIG_ARCHIVE_MAP", "TSPulseConfig"],
    "models.flowstate": ["FLOWSTATE_PRETRAINED_CONFIG_ARCHIVE_MAP", "FlowStateConfig"],
    "models.patchtst_fm": ["PATCHTSTFM_PRETRAINED_CONFIG_ARCHIVE_MAP", "PatchTSTFMConfig"],
    "toolkit": [
        "TimeSeriesPreprocessor",
        "TimeSeriesForecastingPipeline",
        "ClassificationDFDataset",
        "ForecastDFDataset",
        "PretrainDFDataset",
        "RegressionDFDataset",
        "get_datasets",
        "load_dataset",
        "get_model",
        "count_parameters",
        "TrackingCallback",
        "count_parameters",
        "optimal_lr_finder",
        "ClassificationDFDataset",
        "TimeSeriesImputationPipeline",
        "TimeSeriesClassificationPipeline",
        "TimeSeriesAnomalyDetectionPipeline",
        "TimeSeriesPreprocessor",
        "TimeSeriesClassificationPreprocessor",
    ],
}


# PyTorch-backed objects
_import_structure["models.tinytimemixer"].extend(
    [
        "TINYTIMEMIXER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TinyTimeMixerPreTrainedModel",
        "TinyTimeMixerModel",
        "TinyTimeMixerForMaskedPrediction",
        "TinyTimeMixerForPrediction",
    ]
)
_import_structure["models.tspulse"].extend(
    [
        "TSPULSE_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TSPulsePreTrainedModel",
        "TSPulseModel",
        "TSPulseForReconstruction",
        "TSPulseForClassification",
    ]
)
_import_structure["models.flowstate"].extend(
    [
        "FLOWSTATE_PRETRAINED_MODEL_ARCHIVE_LIST",
        "FlowStatePreTrainedModel",
        "FlowStateModel",
        "FlowStateForPrediction",
    ]
)
_import_structure["models.patchtst_fm"].extend(
    [
        "PATCHTSTFM_PRETRAINED_MODEL_ARCHIVE_LIST",
        "PatchTSTFMPretrainedModel",
        "PatchTSTFMModel",
        "PatchTSTFMForPrediction",
    ]
)

# Direct imports for type-checking
if TYPE_CHECKING:
    from .models.flowstate import (
        FLOWSTATE_PRETRAINED_CONFIG_ARCHIVE_MAP,
        FLOWSTATE_PRETRAINED_MODEL_ARCHIVE_LIST,
        FlowStateConfig,
        FlowStateForPrediction,
        FlowStateModel,
        FlowStatePreTrainedModel,
    )
    from .models.patchtst_fm import (
        PATCHTSTFM_PRETRAINED_CONFIG_ARCHIVE_MAP,
        PATCHTSTFM_PRETRAINED_MODEL_ARCHIVE_LIST,
        PatchTSTFMConfig,
        PatchTSTFMForPrediction,
        PatchTSTFMModel,
        PatchTSTFMPreTrainedModel,
    )
    from .models.tinytimemixer import (
        TINYTIMEMIXER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        TINYTIMEMIXER_PRETRAINED_MODEL_ARCHIVE_LIST,
        TinyTimeMixerConfig,
        TinyTimeMixerForMaskedPrediction,
        TinyTimeMixerForPrediction,
        TinyTimeMixerModel,
        TinyTimeMixerPreTrainedModel,
    )
    from .models.tspulse import (
        TSPULSE_PRETRAINED_CONFIG_ARCHIVE_MAP,
        TSPULSE_PRETRAINED_MODEL_ARCHIVE_LIST,
        TSPulseConfig,
        TSPulseForClassification,
        TSPulseForReconstruction,
        TSPulseModel,
        TSPulsePreTrainedModel,
    )
    from .toolkit import (
        ClassificationDFDataset,
        ForecastDFDataset,
        PretrainDFDataset,
        RegressionDFDataset,
        TimeSeriesAnomalyDetectionPipeline,
        TimeSeriesClassificationPipeline,
        TimeSeriesClassificationPreprocessor,
        TimeSeriesForecastingPipeline,
        TimeSeriesImputationPipeline,
        TimeSeriesPreprocessor,
        TrackingCallback,
        count_parameters,
        get_datasets,
        get_model,
        load_dataset,
        optimal_lr_finder,
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
        extra_objects={"__version__": __version__},
    )


# register local models now
register_config(model_type="tinytimemixer", model_config_name="TinyTimeMixerConfig", module_path="tsfm_public")
register_config(model_type="tspulse", model_config_name="TSPulseConfig", module_path="tsfm_public")
