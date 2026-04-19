"""Base serivce handler"""

import enum
import importlib
import logging
import os
from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd

from tsfm_public.toolkit.tsfm_config import TSFMConfig

from .inference_payloads import (
    BaseMetadataInput,
    BaseParameters,
)


LOGGER = logging.getLogger(__file__)

# Security: Base allowlist of trusted module prefixes for handler loading
_BASE_ALLOWED_HANDLER_MODULE_PREFIXES = (
    "tsfm_public.",
    "tsfminference.",
    "tsfmfinetuning.",
)

# Security: Allow additional trusted module prefixes via environment variable
# Format: comma-separated list of module prefixes, e.g., "mycompany.models.,custom.handlers."
# Each prefix MUST end with a dot to ensure exact module path matching
_ADDITIONAL_ALLOWED_PREFIXES = os.getenv("TSFM_ADDITIONAL_HANDLER_MODULES", "")

def _validate_and_parse_additional_prefixes(prefixes_str: str) -> tuple:
    """Parse and validate additional module prefixes from environment variable.
    
    Security: Enforces that each prefix ends with a dot to prevent overly broad matches.
    For example, "mycompany." is safe, but "mycompany" would match "mycompany_evil".
    
    Args:
        prefixes_str: Comma-separated string of module prefixes
        
    Returns:
        Tuple of validated prefixes
        
    Raises:
        ValueError: If any prefix doesn't end with a dot
    """
    if not prefixes_str.strip():
        return tuple()
    
    prefixes = []
    for prefix in prefixes_str.split(","):
        prefix = prefix.strip()
        if not prefix:
            continue
        
        if not prefix.endswith("."):
            raise ValueError(
                f"Security: Additional handler module prefix '{prefix}' must end with a dot ('.'). "
                f"This ensures exact module path matching and prevents overly broad allowlists. "
                f"Example: use 'mycompany.models.' instead of 'mycompany.models'"
            )
        
        prefixes.append(prefix)
    
    return tuple(prefixes)

_additional_prefixes = _validate_and_parse_additional_prefixes(_ADDITIONAL_ALLOWED_PREFIXES)

# Combine base and additional prefixes
ALLOWED_HANDLER_MODULE_PREFIXES = _BASE_ALLOWED_HANDLER_MODULE_PREFIXES + _additional_prefixes

if _additional_prefixes:
    LOGGER.info(
        f"Additional handler module prefixes enabled: {_additional_prefixes}. "
        f"Ensure these modules are from trusted sources."
    )

# Security: Environment variable to explicitly trust remote code
TSFM_TRUST_REMOTE_CODE = int(os.getenv("TSFM_TRUST_REMOTE_CODE", "0")) == 1


class HandlerFunction(enum.Enum):
    """`Enum` for the different functions for which we use handlers."""

    INFERENCE = "inference"
    TUNING = "tuning"


class ServiceHandler:
    """Abstraction to enable serving of various models.

    Args:
        implementation (object): An instantiated class that implements methods needed to perform the required handler
            functions.
    """

    def __init__(self, implementation: object):
        # , model_id: str, model_path: Union[str, Path], handler_config: TSFMConfig, implementation: type):
        # self.model_id = model_id
        # self.model_path = model_path
        # self.handler_config = handler_config
        # self.prepared = False
        self.implementation = implementation

    @classmethod
    def load(
        cls,
        model_id: str,
        model_path: Union[str, Path],
        handler_function: str,
    ) -> Tuple["ServiceHandler", None] | Tuple[None, Exception]:
        """Load the handler_config -- the tsfm service config for this model, returning the proper
        handler to use the model.

        model_path is expected to point to a folder containing the tsfm_config.json file. This can be a local folder
        or with a model on the HuggingFace Hub.

        Args:
            model_id (str): A string identifier for the model.
            model_path (Union[str, Path]): The full path to the model, can be a local path or a HuggingFace Hub path.
            handler_function (str): The type of handler, currently supported handlers are defined in the HandlerFunction
                enum.

        """

        handler_config_path = Path(model_path) if isinstance(model_path, str) else model_path

        try:
            config = TSFMConfig.from_pretrained(handler_config_path)
        except (FileNotFoundError, OSError):
            LOGGER.info("TSFM Config file not found.")
            config = TSFMConfig()
        try:
            handler_class = get_service_handler_class(config, handler_function)

            # could validate handler_class here

            return cls(
                implementation=handler_class(
                    model_id=model_id,
                    model_path=model_path,
                    handler_config=config,
                )
            ), None
        except Exception as e:
            return None, e

    def prepare(
        self,
        data: pd.DataFrame,
        schema: Optional[BaseMetadataInput] = None,
        parameters: Optional[BaseParameters] = None,
        **kwargs,
    ) -> Tuple["ServiceHandler", None] | Tuple[None, Exception]:
        """Prepare the wrapper by loading all the components needed to use the model.

        The actual preparation is done in the `_prepare()` method -- which should be overridden for a
        particular model implementation. This is separate from `load()` above because we may need to know
        certain details about the model (learned from the handler config) before we can effectively load
        and configure the model artifacts.

        Args:
            data (pd.DataFrame): A pandas dataframe containing historical data.
            schema (Optional[BaseMetadataInput], optional): Service request schema. Defaults to None.
            parameters (Optional[BaseParameters], optional): Service requst parameters. Defaults to None.

        Returns:
            Tuple(ServiceHandler, None) or Tuple(None, Exception): In the first case, a tuple containing
                a prepared service handler is returned. The prepared service handler contains all the
                necessary artifacts for performing subsequent inference or training tasks. In the second
                case, the tuple contains an error object.
        """
        try:
            self.implementation.prepare(data=data, schema=schema, parameters=parameters, **kwargs)
            self.prepared = True
            return self, None
        except Exception as e:
            return self, e

    @property
    def handler_config(self):
        if self.implementation is not None:
            return self.implementation.handler_config
        else:
            return None


def _validate_handler_module_path(module_path: str) -> None:
    """Validate that the handler module path is from a trusted source.
    
    Security: This function prevents arbitrary code execution by ensuring that
    only modules from allowlisted prefixes can be loaded. This protects against
    attacks where an attacker publishes a malicious HuggingFace repo with a
    crafted tsfm_config.json that attempts to load arbitrary Python modules.
    
    Args:
        module_path: The module path to validate
        
    Raises:
        ValueError: If the module path is not from an allowed prefix
    """
    if not any(module_path.startswith(prefix) for prefix in ALLOWED_HANDLER_MODULE_PREFIXES):
        raise ValueError(
            f"Security: Handler module path '{module_path}' is not allowed. "
            f"Only modules with the following prefixes are permitted: {ALLOWED_HANDLER_MODULE_PREFIXES}. "
            f"Contact your security team before enabling TSFM_TRUST_REMOTE_CODE in production."
        )


def _validate_handler_class_name(class_name: str) -> None:
    """Validate that the handler class name follows expected naming conventions.
    
    Security: This function provides defense-in-depth by ensuring that class names
    match expected handler naming patterns. This prevents attacks where an allowlisted
    module might re-export dangerous objects (like subprocess or os) at module level,
    which could be accessed via the class_name field in tsfm_config.json.
    
    Args:
        class_name: The class name to validate
        
    Raises:
        ValueError: If the class name does not match allowed patterns
    """
    # Enforce that class names match expected handler naming convention
    allowed_suffixes = ("Handler", "ServiceHandler")
    if not any(class_name.endswith(suffix) for suffix in allowed_suffixes):
        raise ValueError(
            f"Security: Handler class name '{class_name}' does not match allowed pattern. "
            f"Class names must end with one of: {allowed_suffixes}. "
            f"Contact your security team before enabling TSFM_TRUST_REMOTE_CODE in production."
        )


def get_service_handler_class(
    config: TSFMConfig, handler_function: str = HandlerFunction.INFERENCE.value
) -> "ServiceHandler":
    if handler_function == HandlerFunction.INFERENCE.value:
        handler_module_path_identifier = "inference_handler_module_path"
        handler_class_name_identifier = "inference_handler_class_name"
    elif handler_function == HandlerFunction.TUNING.value:
        handler_module_path_identifier = "tuning_handler_module_path"
        handler_class_name_identifier = "tuning_handler_class_name"
    else:
        raise ValueError(f"Unknown handler_function `{handler_function}`")

    if getattr(config, handler_module_path_identifier, None) and getattr(config, handler_class_name_identifier, None):
        module_path = getattr(config, handler_module_path_identifier)
        class_name = getattr(config, handler_class_name_identifier)
        
        # Security: Validate module path and class name before loading
        if not TSFM_TRUST_REMOTE_CODE:
            _validate_handler_module_path(module_path)
            _validate_handler_class_name(class_name)
        else:
            LOGGER.warning(
                f"TSFM_TRUST_REMOTE_CODE is enabled. Loading handler module '{module_path}' "
                f"and class '{class_name}' without validation. "
                f"This may pose a security risk if loading from untrusted sources."
            )
        
        module = importlib.import_module(module_path)
        my_class = getattr(module, class_name)

    elif handler_function == HandlerFunction.INFERENCE.value:
        # Default to forecasting task, inference
        from .tsfm_inference_handler import TSFMForecastingInferenceHandler

        my_class = TSFMForecastingInferenceHandler
    elif handler_function == HandlerFunction.TUNING.value:
        # Default to forecasting task, tuning
        from .tsfm_tuning_handler import TSFMForecastingTuningHandler

        my_class = TSFMForecastingTuningHandler

    return my_class
