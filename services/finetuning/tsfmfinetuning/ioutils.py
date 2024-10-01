"""Utilities for routes"""

# Standard
import base64
import datetime
import io
import logging
import os
import re
from io import BytesIO

import numpy as np
import pandas as pd
from transformers import AutoConfig

# Third Party
# First Party
from tsfm_public import TinyTimeMixerConfig


AutoConfig.register("tinytimemixer", TinyTimeMixerConfig)

LOGGER = logging.getLogger(__file__)

# a file we leave in a directory to
# signal other threads/processes that
# it's been properly written and this
# readable from outside a process or thread
# lock (which are costly).
BREADCRUMB_FILE = ".tsfmservices"
CHECK_FOR_BREADCRUMB = os.environ.get("TSFM_MODEL_CACHE_ROOT") is None

# please don't do this
# LOGGER.setLevel(logging.INFO)


def bytes_to_b64(data: bytes) -> str:
    return base64.b64encode(io.BytesIO(data).getbuffer()).decode("utf-8")


def get_bucket_name(s3_uri):
    return re.search("://(.+?)/", s3_uri).group(1)


def get_object_name(s3_uri):
    bn = get_bucket_name(s3_uri=s3_uri)
    return s3_uri[s3_uri.find(bn) + len(bn) + 1 :]


def _to_pandas(iscsv: bool, isfeather: bool, buf: BytesIO = None, timestamp_column: str = None):
    if iscsv:
        return pd.read_csv(buf, parse_dates=[timestamp_column] if timestamp_column else False)
    if isfeather:
        return pd.read_feather(buf)

    raise Exception("should not be here!")


def _iscsv(uri):
    return uri.upper().endswith(".CSV") or uri.upper().endswith(".CSV.GZ")


def _isfeather(uri):
    return uri.upper().endswith(".FEATHER")


def _readcsv(buf: BytesIO, timestamp_column):
    try:
        return pd.read_csv(buf, parse_dates=[timestamp_column] if timestamp_column else False)
    except Exception as _:
        buf.seek(0)
        return None


def _readfeather(buf: BytesIO, timestamp_column=None):
    try:
        return pd.read_feather(buf)
    except Exception as _:
        buf.seek(0)
        return None


def to_pandas(uri: str, **kwargs) -> pd.DataFrame:
    """Will return a pandas dataframe for a given uri.

    Args:
        uri (str): A uri which describes the location of the data.
        kwargs: use this to pass arguments from your service payload
        that are used in this method


    Returns:
        pd.DataFrame: a pandas DataFrame object
    """

    received_bytes = not isinstance(uri, str)
    # some guardrails (dependeing on deployment we'll want to paramterize these)
    # at present all we're allowing is a local (file://) refernece to csv or feather
    #                               012345
    if uri[0:6].upper().startswith("S3A://"):
        raise NotImplementedError("reading s3 input is currently forbidden.")
    if received_bytes:
        raise NotImplementedError("reading byte input is currently forbidden.")
    if not received_bytes and uri[0:8].upper().startswith("HTTPS://"):
        raise NotImplementedError("reading from https sources is forbidden.")
    if not received_bytes and uri[0:8].upper().startswith("HTTP://"):
        raise NotImplementedError("reading from http sources is forbidden.")

    timestamp_column = kwargs.get("timestamp_column", None)
    # we're reading from file                              0123456
    if not received_bytes and uri[0:7].upper().startswith("FILE://"):
        return _to_pandas(iscsv=_iscsv(uri), isfeather=_isfeather(uri), buf=uri[7:], timestamp_column=timestamp_column)

    # attempt to read a binary blob in different formats
    bio = BytesIO(base64.b64decode(uri)) if not received_bytes else BytesIO(uri)
    answer = _readfeather(bio)
    if answer is None:
        answer = _readcsv(bio, timestamp_column=timestamp_column)
    if answer is None:
        raise Exception("unable to read data input")
    # we're enforcing timestamp column dtypes that you can
    # perform arithematic operations on
    if timestamp_column is not None and not isinstance(
        answer.dtypes[timestamp_column],
        (
            np.dtypes.DateTime64DType,
            np.dtypes.Float64DType,
            np.dtypes.Int64DType,
            np.dtypes.Int32DType,
            np.dtypes.Int16DType,
            np.dtypes.Float32DType,
            np.dtypes.Float16DType,
            datetime.datetime,
            pd.Timestamp,
            pd.DatetimeTZDtype,
            int,
            float,
        ),
    ):
        raise ValueError(
            f"""column '{timestamp_column}' can not be parsed to a datetime or other numeric type.
            pandas assigns it to type {answer.dtypes[timestamp_column]}."""
        )

    return answer


def getredis():
    """Obtain a redis client. You should have the following environ variables
    defined:

    REDIS_IP (defaults to localhost)
    REDIS_PORT (defaults to 6379)
    REDIS_PASSWORD (must be defined)

    Returns:
        StrictRedis: A StrictRedis implementation.
    """
    # Third Party
    from redis import StrictRedis

    redis_host = os.getenv("REDIS_IP", default="localhost")
    # Note that port 6379 can also be grabbed by occupied by ray's gcs services
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    redis_password = os.getenv("REDIS_PASSWORD")

    # Checkout here for decode_responses https://redis-py-cluster.readthedocs.io/en/stable/
    # return Redis()
    return StrictRedis(
        host=redis_host,
        port=redis_port,
        password=redis_password,
        # decode_responses=True,
    )
