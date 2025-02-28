import logging
from pathlib import Path
from typing import Tuple, Union

import pandas as pd

from .inference_payloads import (
    BaseMetadataInput,
    BaseParameters,
)
from .service_handler import HandlerFunction, ServiceHandler


LOGGER = logging.getLogger(__file__)


class TuningHandler(ServiceHandler):
    @classmethod
    def load(
        cls, model_id: str, model_path: Union[str, Path]
    ) -> Tuple["TuningHandler", None] | Tuple[None, Exception]:
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

        return super().load(model_id, model_path, handler_function=HandlerFunction.TUNING.value)

    def train(
        self,
        data: pd.DataFrame,
        schema: BaseMetadataInput,
        parameters: BaseParameters,
        tuned_model_name: str,
        tmp_dir: Path,
    ) -> Tuple[str, None] | Tuple[None, Exception]:
        """Perform a fine-tuning request"""
        if not self.prepared:
            return None, RuntimeError("Service wrapper has not yet been prepared; run `handler.prepare()` first.")

        try:
            result = self.implementation.train(
                data, schema=schema, parameters=parameters, tuned_model_name=tuned_model_name, tmp_dir=tmp_dir
            )

            # Does TuneOuput need some info about the request -- for billing purposes?
            return result, None

        except Exception as e:
            return None, e
