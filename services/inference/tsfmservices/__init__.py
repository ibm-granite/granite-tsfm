import logging
import os


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

# Do we have redis?

try:
    # Third Party
    import redis
    import rq

    HAVE_REDIS_BACKEND = True
except ImportError:
    HAVE_REDIS_BACKEND = False

TSFM_USE_KFTO_ASYNC_BACKEND = int(os.getenv("TSFM_USE_KFTO_ASYNC_BACKEND", "0")) == 1
USE_REDIS_ASYNC_BACKEND = HAVE_REDIS_BACKEND and not TSFM_USE_KFTO_ASYNC_BACKEND

TSFM_ALLOW_LOAD_FROM_HF_HUB = int(os.getenv("TSFM_ALLOW_LOAD_FROM_HF_HUB", "1")) == 1

TSFM_CONFIG_FILE = os.getenv(
    "TSFM_CONFIG_FILE",
    os.path.realpath(os.path.join(os.path.dirname(__file__), "config", "default_config.yml")),
)
