"""Utilities for routes"""

# Standard
import base64
import datetime
import io
import logging
import os
import random
import re
from io import BytesIO
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from transformers import AutoConfig

# Third Party
# First Party
from tsfm_public import TinyTimeMixerConfig


AutoConfig.register("tinytimemixer", TinyTimeMixerConfig)

LOGGER = logging.getLogger(__file__)


def bytes_to_b64(data: bytes) -> str:
    return base64.b64encode(io.BytesIO(data).getbuffer()).decode("utf-8")


def get_bucket_name(s3_uri):
    return re.search("://(.+?)/", s3_uri).group(1)


def get_object_name(s3_uri):
    bn = get_bucket_name(s3_uri=s3_uri)
    return s3_uri[s3_uri.find(bn) + len(bn) + 1 :]


def _to_pandas(
    iscsv: bool, isfeather: bool, buf: Union[str, BytesIO, None] = None, timestamp_column: Union[str, None] = None
):
    if iscsv:
        if os.path.isdir(buf):
            import pyarrow.dataset as ds

            # Path to directory containing multiple CSV files
            dataset = ds.dataset(buf, format="csv")

            # Convert to a single (virtual) PyArrow Table
            return dataset.to_table().to_pandas()

        else:
            return pd.read_csv(buf, parse_dates=[timestamp_column] if timestamp_column else False)
    if isfeather:
        return pd.read_feather(buf)

    raise Exception("should not be here!")


def _split_text_file(
    source: str, target_dir: Union[str, Path], parts: int = 3, has_header: bool = True, shuffle: bool = True
):
    """Utility for spliting a text file into multiple parts. This is mostly for facilitating data prep
    for test cases involving multiple file input.

    Args:
        source (str): Source file
        target_dir (Union[str, Path]): where split files should be placed
        parts (int, optional): How many near equal parts should be created. Defaults to 3.
        has_header (bool, optional): If source has a header row that should be duplicated in each part. Defaults to True.
        shuffle (bool, optional): Whether to randomly shuffle rows (useful to test certain workflows that might assume ordered data). Defaults to True.
    """
    # split our data into three parts
    lines = open(source).readlines()
    header = lines[0] if has_header else None
    lines = lines[1 if header else 0 :]
    # shuffle rows
    if shuffle:
        random.shuffle(lines)
    splitindex = len(lines) // parts
    remainder = len(lines) % parts
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    extension = source[-4:]
    filenames = [f"{os.path.basename(source).replace(extension, '')}.part_{x}{extension}" for x in range(parts)]
    filenames = [(Path(target_dir) / fn).as_posix() for fn in filenames]

    for idx, filename in enumerate(filenames):
        with open(filename, "w") as acsvfile:
            start = idx * splitindex
            end = start + splitindex if idx < parts - 1 else start + splitindex + remainder
            if header:
                acsvfile.write(header)
            acsvfile.writelines(lines[start:end])
    return filenames


def _iscsv(uri: str):
    # we got a directory?
    # we require that it contains **only** files ending with CSV
    #                          0123456
    if uri.upper().startswith("FILE://"):
        if os.path.isdir(uri[7:]):
            contents: List[Tuple] = list(os.walk(uri[7:]))
            # we do not allow nested directories
            if len(contents) > 1:
                raise ValueError(f"{uri} must not have subdirectories.")
            files: List[str] = contents[0][2]  # type: ignore
            if not all(x.upper().endswith(".CSV") for x in files):
                raise ValueError(f"All files in {uri} must have the 'cvs' extension.")
            return True
        else:  # uri is NOT a directory
            if any([uri.upper().endswith(".CSV"), uri.upper().endswith(".CSV.GZ")]):
                return True

    # LOGGER.info("returning False")
    return False


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
        if not os.path.exists(uri[7:]):
            raise ValueError(f"{uri[7:]} does not exist.")
        return _to_pandas(iscsv=_iscsv(uri), isfeather=_isfeather(uri), buf=uri[7:], timestamp_column=timestamp_column)
    if os.path.exists(uri):
        raise NotImplementedError(f"{uri} is not a proper URI. Absolute or relative path specifies are not allowed.")

    # attempt to read a binary blob in different formats
    bio = BytesIO(base64.b64decode(uri)) if not received_bytes else BytesIO(uri)
    answer = _readfeather(bio)
    if answer is None:
        answer = _readcsv(bio, timestamp_column=timestamp_column)
    if answer is None:
        raise ValueError("Unable to read data input. Check that it exists and is in the correct format.")
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
