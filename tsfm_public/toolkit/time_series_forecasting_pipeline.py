# Copyright contributors to the TSFM project
#
"""Hugging Face Pipeline for Time Series Tasks"""

from typing import Dict, Union

import pandas as pd
import torch
from transformers.pipelines.base import GenericTensor, Pipeline
from transformers.utils import (
    logging,
)

from .dataset import ForecastDFDataset


# Eventually we should support all time series models
# MODEL_FOR_TIME_SERIES_FORECASTING_MAPPING_NAMES = OrderedDict(
#     [
#         # Model for Time Series Forecasting
#         ("PatchTST", "PatchTSTForPrediction"),
#         ("TST", "TimeSeriesTransformerForPrediction"),
#     ]
# )


logger = logging.get_logger(__name__)


class TimeSeriesForecastingPipeline(Pipeline):
    """Hugging Face Pipeline for Time Series Forecasting"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.framework == "tf":
            raise ValueError(f"The {self.__class__} is only available in PyTorch.")

        # self.check_model_type(MODEL_FOR_TIME_SERIES_FORECASTING_MAPPING)

    def _sanitize_parameters(self, **kwargs):
        """Assign parameters to the different parts of the process.

        For expected parameters see the call method below.
        """
        preprocess_kwargs = {}
        postprocess_kwargs = {}
        if "id_columns" in kwargs:
            postprocess_kwargs["id_columns"] = kwargs["id_columns"]
            preprocess_kwargs["id_columns"] = kwargs["id_columns"]
        if "timestamp_column" in kwargs:
            postprocess_kwargs["timestamp_column"] = kwargs["timestamp_column"]
            preprocess_kwargs["timestamp_column"] = kwargs["timestamp_column"]
        if "input_columns" in kwargs:
            preprocess_kwargs["input_columns"] = kwargs["input_columns"]
            postprocess_kwargs["input_columns"] = kwargs["input_columns"]
        if "output_columns" in kwargs:
            preprocess_kwargs["output_columns"] = kwargs["output_columns"]
            postprocess_kwargs["output_columns"] = kwargs["output_columns"]
        elif "input_columns" in kwargs:
            preprocess_kwargs["output_columns"] = kwargs["input_columns"]
            postprocess_kwargs["output_columns"] = kwargs["input_columns"]

        return preprocess_kwargs, {}, postprocess_kwargs

    def __call__(self, time_series: Union["pandas.DataFrame", str], **kwargs):
        """Main method of the forecasting pipeline. Takes the input time series data (in tabular format) and
        produces predictions.

        Args:
            time_series (Union[&quot;pandas.DataFrame&quot;, str]): A pandas dataframe or a referce to a location
            from where a pandas datarame can be loaded.

            kwargs

            timestamp_column (str): the column containing the date / timestamp
            id_columns (List[str]): the list of columns containing ID information. If no ids are present, pass [].
            input_columns (List[str]): the columns that are used as to create the inputs to the forecasting model.
            These values are used to select data in the input dataframe.
            output_columns (List[str]): the column names that are used to label the outputs of the forecasting model.
            If omitted, it is assumed that the model will forecast values for all the input columns.


            Return:
            A new pandas dataframe containing the forecasts. Each row will contain the id, timestamp, the original
            input feature values and the output forecast for each input column. The output forecast is a list containing
            all the values over the prediction horizon.

        """

        return super().__call__(time_series, **kwargs)

    def preprocess(self, time_series, **kwargs) -> Dict[str, GenericTensor]:
        """Preprocess step
        Load the data, if not already loaded, and then generate a pytorch dataset.
        """

        if isinstance(time_series, str):
            time_series = pd.read_csv(
                time_series,
                parse_dates=[kwargs["timestamp_column"]],
            )

        # use forecasing dataset to do the preprocessing
        dataset = ForecastDFDataset(
            time_series,
            context_length=self.model.config.context_length,
            prediction_length=self.model.config.prediction_length,
            **kwargs,
        )

        # stack all the outputs
        # torch tensors are stacked, but other values are passed through as a list
        first = dataset[0]
        full_output = {}
        for k, v in first.items():
            if isinstance(v, torch.Tensor):
                full_output[k] = torch.stack(tuple(r[k] for r in dataset))
            else:
                full_output[k] = [r[k] for r in dataset]

        return full_output

    def _forward(self, model_inputs, **kwargs):
        """Forward step
        Responsible for taking pre-processed dictionary of tensors and passing it to
        the model. Aligns model parameters with the proper input parameters. Only passes
        the needed parameters from the dictionary to the model, but adds them back to the
        ouput for the next step.

        The keys in model_outputs are governed by the underlying model combined with any
        original input keys.
        """

        # Eventually we should use inspection somehow
        # inspect.signature(model_forward).parameters.keys()
        model_input_keys = {
            "past_values",
            "future_values",
        }  # todo: this should not be hardcoded
        model_inputs_only = {}
        for k in model_input_keys:
            model_inputs_only[k] = model_inputs[k]

        model_outputs = self.model(**model_inputs_only)

        # copy the other inputs
        copy_inputs = True
        for k in [
            akey
            for akey in model_inputs.keys()
            if (akey not in model_input_keys) or copy_inputs
        ]:
            model_outputs[k] = model_inputs[k]

        return model_outputs

    def postprocess(self, input, **kwargs):
        """Postprocess step
        Takes the dictionary of outputs from the previous step and converts to a more user
        readable pandas format.
        """
        out = {}

        model_output_key = (
            "prediction_outputs"
            if "prediction_outputs" in input.keys()
            else "prediction_logits"
        )

        for i, c in enumerate(kwargs["output_columns"]):
            out[f"{c}_prediction"] = input[model_output_key][:, :, i].numpy().tolist()
        for i, c in enumerate(kwargs["input_columns"]):
            out[c] = input["future_values"][:, :, i].numpy().tolist()

        if "timestamp_column" in kwargs:
            out[kwargs["timestamp_column"]] = input["timestamp"]
        for i, c in enumerate(kwargs["id_columns"]):
            out[c] = [elem[i] for elem in input["id"]]
        out = pd.DataFrame(out)

        # reorder columns
        cols = out.columns.to_list()
        cols_ordered = []
        if "timestamp_column" in kwargs:
            cols_ordered.append(kwargs["timestamp_column"])
        if "id_columns" in kwargs:
            cols_ordered.extend(kwargs["id_columns"])
        cols_ordered.extend([c for c in cols if c not in cols_ordered])

        out = out[cols_ordered]
        return out
