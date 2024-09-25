"""Utilities for routes"""

# Standard
import base64
import datetime
import io
import logging
import os
import re
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from transformers import AutoConfig

# Third Party
# First Party
from tsfm_public import TinyTimeMixerConfig

from .ftpayloads import S3aCredentials


try:
    from minio import Minio
    from minio.deleteobjects import DeleteObject

    HAVE_MINIO = True
except ImportError as _:
    HAVE_MINIO = False


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


def getminio(s3creds: S3aCredentials) -> Minio:
    """Obtain a minio client.

    Returns:
        Minio: A Minio implementation.
    """

    if not HAVE_MINIO:
        raise RuntimeError("you must have minio installed to use this method.")

    # Create minio client
    return Minio(
        endpoint=s3creds.endpoint,
        access_key=s3creds.access_key_id,
        secret_key=s3creds.secret_access_key,
    )


def copy_dir_to_s3(
    minio_client: Minio,
    bucket_name: str,
    local_folder_path: Union[Path, str],
    prefix: Optional[Union[Path, str]] = None,
):
    """Copies a local directory to s3. Note that this will not do any check
       for pre-existing objects. That is, it will always overwrite existing content
       of the same name.

    Args:
        minio_client (Minio): Minio client
        bucket_name (str): The target bucket name
        local_folder_path (Path): The local directory. It's best to use a fully qualified path for this.
        prefix Union[str, Path]: A portion of the local_folder_path that will be retained in the s3 object

    Raises:
        ValueError: If local_folder_path is not a directory or doesn't exist.
    """
    local_folder_path = local_folder_path if isinstance(local_folder_path, Path) else Path(local_folder_path)
    if not (local_folder_path.is_dir() and local_folder_path.exists()):
        raise ValueError(f"{local_folder_path} must be a directory and it must exist.")

    # I just LOVE python!
    if prefix:
        prefix = prefix if isinstance(prefix, Path) else Path(prefix)

    if not prefix:
        prefix = os.path.basename(local_folder_path)

    for root, _, files in os.walk(local_folder_path):
        for file in files:
            local_file_path = os.path.join(root, file)
            s3_object_name = os.path.join(prefix, os.path.relpath(local_file_path, local_folder_path))
            minio_client.fput_object(bucket_name, s3_object_name, local_file_path)
            LOGGER.info(f"Uploaded {local_file_path} to {s3_object_name} in bucket {bucket_name}")


def dump_model_from_s3(
    s3creds: S3aCredentials,
    bucket_name: str,
    model_id: str,
    model_prefix: str = "model",
    preprocessor_prefix: str = "preprocessor",
    target_directory: Path | str = Path(tempfile.gettempdir()),
):
    """
    This method will always dump the model from s3 to the
    desired target directory.
    """
    LOGGER.info(f"in dump_model_from_s3 with model_id {model_id} and target_directory {target_directory}")

    # create the directory holding the model
    target_directory = target_directory if isinstance(target_directory, Path) else Path(target_directory)
    model_path = target_directory / f"{model_id}/{model_prefix}"
    model_path.mkdir(parents=True, exist_ok=True)
    # create the directory holding the preprocessor
    preprocessor_path = target_directory / f"{model_id}/{preprocessor_prefix}"
    preprocessor_path.mkdir(parents=True, exist_ok=True)

    mc = getminio(s3creds)
    found = False
    # avoid ambigious prefixes for similarly named models
    theprefix = model_id if model_id.endswith("/") else model_id + "/"
    for o in mc.list_objects(bucket_name=bucket_name, prefix=theprefix, recursive=True):
        if o.is_dir:
            continue
        found = True
        with open(target_directory / o.object_name, mode="wb", buffering=0) as of:
            object = mc.get_object(bucket_name=bucket_name, object_name=o.object_name)
            of.write(object.read())  # yes, we have to copy data
    if not found:
        err = f"unable to find object {bucket_name}/{model_id}"
        LOGGER.error(err)
        raise ValueError(err)
    # place a breadcrumb file to signal others we've been here
    # DO NOT ALTER LOCATION OF THIS W/O FIRST CHECKING
    # WHERE ELSE IT'S USED!
    with open(model_path / BREADCRUMB_FILE, mode="wb", buffering=0) as _:
        ...


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
    #                               01234567
    if not received_bytes and uri[0:8].upper().startswith("HTTPS://"):
        raise NotImplementedError("data not accepted")

    timestamp_column = kwargs.get("timestamp_column", None)
    # we're reading from file       0123456
    if not received_bytes and uri[0:7].upper().startswith("FILE://"):
        return _to_pandas(_iscsv(uri), _isfeather(uri), buf=uri[7:], timestamp_column=timestamp_column)

    # we're reading from s3         012345
    if not received_bytes and uri[0:6].upper().startswith("S3A://"):
        if _iscsv(uri) or _isfeather(uri):
            bucket_name = get_bucket_name(uri)
            # object_name = os.path.basename(uri)
            object_name = uri[uri.index(bucket_name) + len(bucket_name) + 1 :]
            creds = kwargs.get("s3credentials")
            creds = (
                S3aCredentials(**creds)
                if creds
                else S3aCredentials(
                    access_key_id=kwargs.get("access_key_id"),
                    secret_access_key=kwargs.get("secret_access_key"),
                    endpoint=kwargs.get("endpoint"),
                )
            )
            mc = getminio(creds)
            object = mc.get_object(bucket_name=bucket_name, object_name=object_name)
            buf = BytesIO(object.read())
            return _to_pandas(_iscsv(uri), _isfeather(uri), buf, timestamp_column=timestamp_column)

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


def _s3_recursive_delete(mc: Minio, bucket: str, object_prefix: str):
    # clean up
    delete_object_list = [
        lambda x: DeleteObject(x.object_name),
        mc.list_objects(bucket, object_prefix, recursive=True),
    ]
    errors = mc.remove_objects(bucket, delete_object_list)
    # LOGGER.info(
    #    f"attempting to delete {mc.list_objects(bucket, object_prefix, recursive=True)}"
    # )
    for error in errors:
        LOGGER.warn("error occurred when deleting object", error)


def get_job_result_file(job_id: str, root_dir: str = tempfile.gettempdir()):
    return Path(root_dir) / f"{job_id}.result.json"


def copy_file_to_s3(source_path: str, target_bucket: str, minio_client: Minio):
    """This will flatten object names on S3 to use only the basename of the source_path.
    A file named /tmp/dir1/dir2/content.txt will be written to `target_bucket` simply as `content.txt`.
    """
    object_name = os.path.basename(source_path)
    minio_client.fput_object(bucket_name=target_bucket, object_name=object_name, file_path=source_path)


def copy_object_from_s3(
    source_bucket: str,
    source_object: str,
    minio_client: Minio,
    target_directory=tempfile.gettempdir(),
):
    """Similarly as copy_file_to_s3, this will not honor the presence of slashe (`/`) in an object name.
    It will only use the basename of the object writing it directly as
    target_directory/os.path.basename(object_name).
    """
    target_file = Path(target_directory) / os.path.basename(source_object)
    minio_client.fget_object(bucket_name=source_bucket, object_name=source_object, file_path=target_file)
