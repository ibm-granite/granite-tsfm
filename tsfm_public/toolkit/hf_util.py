"""Miscellaneous utilities for huggingface transformers"""

import importlib

from transformers import AutoConfig


# Allowlist of safe module prefixes for security
ALLOWED_MODULE_PREFIXES = (
    "tsfm_public.",
    "tsfminference.",
    "tsfmfinetuning.",
    "transformers.",
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
