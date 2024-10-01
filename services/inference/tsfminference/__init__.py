# Copyright contributors to the TSFM project
#
import logging
import os

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

TSFM_CONFIG_FILE = os.getenv(
    "TSFM_CONFIG_FILE",
    os.path.realpath(os.path.join(os.path.dirname(__file__), "default_config.yml")),
)
