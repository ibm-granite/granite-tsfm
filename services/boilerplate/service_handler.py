"""Base serivce handler"""

import enum
import importlib
import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd

from .inference_payloads import (
    BaseMetadataInput,
    BaseParameters,
)
from .tsfm_config import TSFMConfig


LOGGER = logging.getLogger(__file__)


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
        module = importlib.import_module(getattr(config, handler_module_path_identifier))
        my_class = getattr(module, getattr(config, handler_class_name_identifier))

    elif handler_function == HandlerFunction.INFERENCE.value:
        # Default to forecasting task, inference
        from .tsfm_inference_handler import TSFMForecastingInferenceHandler

        my_class = TSFMForecastingInferenceHandler
    elif handler_function == HandlerFunction.TUNING.value:
        # Default to forecasting task, tuning
        from .tsfm_tuning_handler import TSFMForecastingTuningHandler

        my_class = TSFMForecastingTuningHandler

    return my_class
