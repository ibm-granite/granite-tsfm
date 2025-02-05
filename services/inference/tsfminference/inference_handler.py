"""Base serivce handler"""

import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from .inference_payloads import (
    BaseMetadataInput,
    BaseParameters,
    PredictOutput,
)
from .service_handler import HandlerFunction, ServiceHandler


LOGGER = logging.getLogger(__file__)


class InferenceHandler(ServiceHandler):
    @classmethod
    def load(
        cls, model_id: str, model_path: Union[str, Path]
    ) -> Tuple["InferenceHandler", None] | Tuple[None, Exception]:
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

        return super().load(model_id, model_path, handler_function=HandlerFunction.INFERENCE.value)

    def run(
        self,
        data: pd.DataFrame,
        schema: Optional[BaseMetadataInput] = None,
        parameters: Optional[BaseParameters] = None,
        **kwargs,
    ) -> Tuple[PredictOutput, None] | Tuple[None, Exception]:
        """Perform an inference request

        Args:
            data (pd.DataFrame): A pandas dataframe containing historical data.
            schema (Optional[BaseMetadataInput], optional): Service request schema. Defaults to None.
            parameters (Optional[BaseParameters], optional): Service requst parameters. Defaults to None.

        Returns:
            Tuple[PredictOutput, None] | Tuple[None, Exception]: If successful, returns a tuple containing a PredictionOutput
                object as the first element. If unsuccessful, a tuple with an exception as the second element will be returned.
        """

        if not self.prepared:
            return None, RuntimeError("Service wrapper has not yet been prepared; run `handler.prepare()` first.")

        try:
            result = self.implementation.run(data, schema=schema, parameters=parameters, **kwargs)
            encoded_result = encode_dataframe(result)
            counts = self.implementation.calculate_data_point_counts(
                data, output_data=result, schema=schema, parameters=parameters, **kwargs
            )
            return PredictOutput(
                model_id=str(self.implementation.model_id),
                created_at=datetime.datetime.now().isoformat(),
                results=[encoded_result],
                **counts,
            ), None

        except Exception as e:
            return None, e


def encode_dataframe(result: pd.DataFrame, timestamp_column: str = None) -> Dict[str, List[Any]]:
    if timestamp_column and pd.api.types.is_datetime64_any_dtype(result[timestamp_column]):
        result[timestamp_column] = result[timestamp_column].apply(lambda x: x.isoformat())
    return result.to_dict(orient="list")
