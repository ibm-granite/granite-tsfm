# Copyright contributors to the TSFM project
#

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

# Check the dependencies satisfy the minimal versions required.
from transformers.utils import _LazyModule

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
    "toolkit": [
        "TimeSeriesPreprocessor",
        "TimeSeriesForecastingPipeline",
        "ForecastDFDataset",
        "PretrainDFDataset",
        "RegressionDFDataset",
        "get_datasets",
        "load_dataset",
        "TrackingCallback",
        "count_parameters",
    ],
}


# PyTorch-backed objects
_import_structure["models.tinytimemixer"].extend(
    [
        "TINYTIMEMIXER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TinyTimeMixerPreTrainedModel",
        "TinyTimeMixerModel",
        "TinyTimeMixerForPrediction",
    ]
)

# Direct imports for type-checking
if TYPE_CHECKING:
    from .models.tinytimemixer import (
        TINYTIMEMIXER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        TINYTIMEMIXER_PRETRAINED_MODEL_ARCHIVE_LIST,
        TinyTimeMixerConfig,
        TinyTimeMixerForPrediction,
        TinyTimeMixerModel,
        TinyTimeMixerPreTrainedModel,
    )
    from .toolkit import (
        ForecastDFDataset,
        PretrainDFDataset,
        RegressionDFDataset,
        TimeSeriesForecastingPipeline,
        TimeSeriesPreprocessor,
        TrackingCallback,
        count_parameters,
        get_datasets,
        load_dataset,
    )
else:
    # Standard
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
        extra_objects={"__version__": __version__},
    )
