"""Miscellaneous utilities for huggingface transformers"""

import importlib
import logging
import os

from transformers import AutoConfig


LOGGER = logging.getLogger(__file__)

# Security: Base allowlist of trusted module prefixes
_BASE_ALLOWED_MODULE_PREFIXES = (
    "tsfm_public.",
    "tsfminference.",
    "tsfmfinetuning.",
    "transformers.",
)

# Security: Allow additional trusted module prefixes via environment variable
# Format: comma-separated list of module prefixes, e.g., "mycompany.models.,custom.modules."
# Each prefix MUST end with a dot to ensure exact module path matching
_ADDITIONAL_ALLOWED_MODULE_PREFIXES = os.getenv("TSFM_ADDITIONAL_MODULE_PREFIXES", "")


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
        return ()

    prefixes = []
    for prefix in prefixes_str.split(","):
        prefix = prefix.strip()
        if not prefix:
            continue

        if not prefix.endswith("."):
            raise ValueError(
                f"Security: Additional module prefix '{prefix}' must end with a dot ('.'). "
                f"This ensures exact module path matching and prevents overly broad allowlists. "
                f"Example: use 'mycompany.models.' instead of 'mycompany.models'"
            )

        prefixes.append(prefix)

    return tuple(prefixes)


_additional_prefixes = _validate_and_parse_additional_prefixes(_ADDITIONAL_ALLOWED_MODULE_PREFIXES)

# Combine base and additional prefixes
ALLOWED_MODULE_PREFIXES = _BASE_ALLOWED_MODULE_PREFIXES + _additional_prefixes

if _additional_prefixes:
    LOGGER.info(
        f"Additional module prefixes enabled: {_additional_prefixes}. "
        f"Ensure these modules are from trusted sources."
    )


def _validate_module_path(module_path: str) -> None:
    """Validate that module_path is from an allowed prefix.

    Args:
        module_path (str): The module path to validate.

    Raises:
        ValueError: If module_path is not from an allowed prefix.
    """
    if not any(module_path.startswith(prefix) for prefix in ALLOWED_MODULE_PREFIXES):
        raise ValueError(
            f"Module path '{module_path}' is not allowed. "
            f"Only modules starting with {ALLOWED_MODULE_PREFIXES} are permitted for security reasons."
        )


def register_config(model_type: str, model_config_name: str, module_path: str) -> None:
    """Register a configuration for a particular model architecture

    Args:
        model_type (Optional[str], optional): The type of the model, from the model implementation. Defaults to None.
        model_config_name (Optional[str], optional): The name of configuration class for the model. Defaults to None.
        module_path (Optional[str], optional): Python module path that can be used to load the
            config/model. Defaults to None.

    Raises:
        RuntimeError: Raised when the module cannot be imported from the provided module path.
        ValueError: Raised when module_path is not from an allowed prefix.
    """
    # example
    # model_type: "tinytimemixer"
    # model_config_name: "TinyTimeMixerConfig"
    # module_path: "tsfm"  # place where config should be importable

    # Validate module_path for security
    _validate_module_path(module_path)

    try:
        mod = importlib.import_module(module_path)
        conf_class = getattr(mod, model_config_name, None)
    except ModuleNotFoundError as exc:  # modulenot found, key error ?
        raise RuntimeError(f"Could not load {model_config_name} from {module_path}") from exc

    if conf_class is not None:
        AutoConfig.register(model_type, conf_class)
    else:
        raise RuntimeError(f"Could not find config for {model_config_name}")
