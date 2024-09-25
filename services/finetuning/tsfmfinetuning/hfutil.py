# Copyright contributors to the TSFM project
#
"""Utilities to support custom time series models"""

import importlib
import logging
import os
import pathlib
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import transformers
from transformers import AutoConfig, PretrainedConfig

import tsfm_public
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor

from .ftpayloads import S3aCredentials
from .ioutils import BREADCRUMB_FILE, CHECK_FOR_BREADCRUMB, dump_model_from_s3


LOGGER = logging.getLogger(__file__)


def register_config(model_type: str, model_config_name: str, module_path: str) -> None:
    """Register a configuration for a particular model architecture

    Args:
        model_type (Optional[str], optional): The type of the model, from the model implementation. Defaults to None.
        model_config_name (Optional[str], optional): The name of configuration class for the model. Defaults to None.
        module_path (Optional[str], optional): Python module path that can be used to load the
            config/model. Defaults to None.

    Raises:
        RuntimeError: Raised when the module cannot be imported from the provided module path.
    """
    # example
    # model_type: "tinytimemixer"
    # model_config_name: "TinyTimeMixerConfig"
    # module_path: "tsfm"  # place where config should be importable

    # AutoConfig.register("tinytimemixer", TinyTimeMixerConfig)
    try:
        mod = importlib.import_module(module_path)
        conf_class = getattr(mod, model_config_name, None)
    except ModuleNotFoundError as exc:  # modulenot found, key error ?
        raise RuntimeError(f"Could not load {model_config_name} from {module_path}") from exc

    if conf_class is not None:
        AutoConfig.register(model_type, conf_class)
    else:
        # issue warning?
        pass


def load_config(
    model_path: Union[str, pathlib.Path],
    model_type: Optional[str] = None,
    model_config_name: Optional[str] = None,
    module_path: Optional[str] = None,
) -> PretrainedConfig:
    """Load configuration

    Attempts to load the configuration, if it is not loadable, then we register it with the AutoConfig mechanism.

    Args:
        model_path (pathlib.Path): The path from which to load the config.
        model_type (Optional[str], optional): The type of the model, from the model implementation. Defaults to None.
        model_config_name (Optional[str], optional): The name of configuration class for the model. Defaults to None.
        module_path (Optional[str], optional): Python module path that can be used to load the
            config/model. Defaults to None.

    Returns:
        PretrainedConfig: The configuration object corresponding to the pretrained model.
    """
    # load config first try autoconfig, if not then we register and load

    try:
        conf = AutoConfig.from_pretrained(model_path)
    except (KeyError, ValueError) as exc:  # determine error raised by autoconfig
        if model_type is None or model_config_name is None or module_path is None:
            raise ValueError("model_type, model_config_name, and module_path should be specified.") from exc

        register_config(model_type, model_config_name, module_path)
        conf = AutoConfig.from_pretrained(model_path)

    return conf


def _hf_model_load(load_path: Path, **config_kwargs: Dict[str, Any]):
    """Load model from a given path. Currently determines model class by searchin private tsfm and huggingface.

    Args:
        load_path (Path): String representing path from which to load model.

    Raises:
        AttributeError: Raised when model class cannot be found for an architecture.
        ValueError: When the architecture cannot be determined from the model config.

    Returns:
        _type_: _description_
    """

    model_class, conf = load_model_config(load_path, model_prefix=None, **config_kwargs)
    return model_class.from_pretrained(load_path, local_files_only=True, config=conf)


def prepare_model_and_preprocessor(
    model_id: str,
    model_prefix="model",
    preprocessor_prefix: str = "preprocessor",
    bucket_name: str = None,
    s3creds: S3aCredentials = None,
) -> Path:
    LOGGER.info(f"in prepare_model_and_preprocessor with model_id {model_id}")

    # we have to deal with wanting to dump a file
    # and with one already existing in or out of our
    # temporary location (this is due to both testing and kmodel
    # requirements)
    target_location = Path(os.environ.get("TSFM_MODEL_CACHE_ROOT", Path(tempfile.gettempdir()).as_posix()))

    LOGGER.info(f"target_location {target_location}")
    mp = Path(f"{model_id}/{model_prefix}")
    LOGGER.info(f"mp: {mp}")
    s3_load_path = Path(model_id)  # for s3
    LOGGER.info(f"s3_load_path: {s3_load_path}")
    # this next block was originally intended to service test cases
    # with relative model directories but apparently
    # is getting used for more that that
    if mp.is_dir():
        LOGGER.info(f"1:returning {mp.parent} from already existing model path")
        return mp.parent
    # mimics an s3 download
    if (target_location / mp).is_dir() and (not CHECK_FOR_BREADCRUMB or (mp / BREADCRUMB_FILE).exists()):
        LOGGER.info(f"2:returning {target_location / s3_load_path} from already existing model path")
        return target_location / s3_load_path

    # if we're here we have to get it from s3
    LOGGER.info("calling dump_model_from_s3")
    dump_model_from_s3(
        s3creds=s3creds,
        bucket_name=bucket_name,
        model_id=model_id,
        preprocessor_prefix=preprocessor_prefix,
        model_prefix=model_prefix,
        target_directory=target_location,
    )

    LOGGER.info(f"exiting prepare_model_and_preprocessor returning {target_location / s3_load_path}")
    return target_location / s3_load_path


def load_preprocessor(
    model_id: str,
    preprocessor_prefix: str = "preprocessor",
    model_prefix="model",
    bucket_name: str = None,
    s3creds: S3aCredentials = None,
    **config_kwargs: Dict[str, Any],
):  # TODO minio hop needed
    load_path = prepare_model_and_preprocessor(
        model_id,
        model_prefix=model_prefix,
        preprocessor_prefix=preprocessor_prefix,
        bucket_name=bucket_name,
        s3creds=s3creds,
    )
    load_path_ = load_path / preprocessor_prefix

    # check if the path is an empty directory
    if load_path_.is_dir() and not os.listdir(load_path_):
        return None
    return TimeSeriesPreprocessor.from_pretrained(load_path_, local_files_only=True, **config_kwargs)


def load_model(
    model_id: str,
    model_prefix="model",
    preprocessor_prefix: str = "preprocessor",
    bucket_name: str = None,
    s3creds: S3aCredentials = None,
    **config_kwargs: Dict[str, Any],
):
    load_path = prepare_model_and_preprocessor(
        model_id,
        model_prefix=model_prefix,
        preprocessor_prefix=preprocessor_prefix,
        bucket_name=bucket_name,
        s3creds=s3creds,
    )

    return _hf_model_load(load_path / model_prefix, **config_kwargs)


def _get_model_class(config: PretrainedConfig, module_path: Optional[str] = None) -> type:
    """Helper to find model class based on config object

    First the module_path will be checked if it can be loaded in the current environment. If not
    then the transformers library will be used.

    Args:
        config (PretrainedConfig): HF configuration for the model.
        module_path (Optional[str], optional): Python module path that can be used to load the
            config/model. Defaults to None.

    Raises:
        AttributeError: Raised if the module at module_path cannot be loaded.
        AttributeError: If the architecture provided by the config cannot be loaded from
            the module.

    Returns:
        type: The class for the model.
    """
    if module_path is not None:
        try:
            mod = importlib.import_module(module_path)
        except ModuleNotFoundError as exc:
            raise AttributeError("Could not load module '{module_path}'.") from exc
    else:
        mod = transformers

    # get architecture from model config
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        try:
            model_class = getattr(mod, arch)
            return model_class
        except AttributeError as exc:
            # catch specific error import error or attribute error
            raise AttributeError("Could not load model class for architecture '{arch}'.") from exc


def load_model_config(
    load_path: Path, model_prefix="model", **config_kwargs: Dict[str, Any]
) -> Tuple[type, transformers.PretrainedConfig]:
    """Load model from a given path. Currently determines model class by searchin private tsfm and huggingface.

    Args:
        load_path (str): String representing path from which to load model.

    Raises:
        AttributeError: Raised when model class cannot be found for an architecture.
        ValueError: When the architecture cannot be determined from the model config.

    Returns:
        Tuple[type, transformers.PretrainedConfig]: Tuple containing the model class and the loaded configuration
    """

    load_path_ = load_path / model_prefix if model_prefix else load_path

    conf = AutoConfig.from_pretrained(load_path_, local_files_only=True, **config_kwargs)
    # For now assume we use models only that are compatible with HF Transformers
    architectures = getattr(conf, "architectures", [])

    for arch in architectures:
        try:
            model_class = getattr(transformers, arch)
        except AttributeError:
            try:
                model_class = getattr(tsfm_public, arch)
            except AttributeError:
                # raise here: could not find
                raise AttributeError("Could not find model class for architecture '{arch}'.")

        return model_class, conf

    raise ValueError(f"Could not retrieve `architectures` attribute from {type(conf)}.")

    ...
