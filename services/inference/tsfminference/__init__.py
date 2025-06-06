# Copyright contributors to the TSFM project
#
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional

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

TSFM_ALLOW_LOAD_FROM_HF_HUB = int(os.getenv("TSFM_ALLOW_LOAD_FROM_HF_HUB", "1")) == 1

# use TSFM_MODEL_DIR preferentially. If not set, use HF_HOME or the system tempdir if that's not set.
TSFM_MODEL_DIR: str = os.environ.get("TSFM_MODEL_DIR", os.environ.get("HF_HOME", tempfile.gettempdir()))

# basic checks
# make sure at least one of them is a valid directory
# make sure it's readable as well
_amodeldir_found = next(
    (
        adir
        for adir in (Path(p) for p in TSFM_MODEL_DIR.split(":"))
        if adir.exists() and adir.is_dir() and os.access(adir, os.R_OK)
    ),
    None,
)
if not _amodeldir_found and not TSFM_ALLOW_LOAD_FROM_HF_HUB:
    raise Exception(
        f"None of the values given in TSFM_MODEL_DIR {TSFM_MODEL_DIR} are an existing and readable directory."
    )


DO_PROMETHEUS_TRACKING = int(os.environ.get("TSFM_DO_PROMETHEUS_TRACKING", "0")) == 1

if DO_PROMETHEUS_TRACKING:
    from prometheus_client import Histogram as TSFM_HISTOGRAM
else:

    class TSFM_HISTOGRAM:
        # A no-op histogram impl
        def __init__(self, name: str, descr: str):
            """This is not a full API match for the Histogram __init__"""
            ...

        def observe(self, amount: float, exemplar: Optional[Dict[str, str]] = None) -> None: ...
