# Copyright contributors to the TSFM project
#
"""Hugging Face Pipeline for Time Series Tasks"""

import datetime
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import torch
from transformers.pipelines.base import (
    GenericTensor,
    Pipeline,
    build_pipeline_init_args,
)
from transformers.utils import add_end_docstrings, logging

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


@add_end_docstrings(
    build_pipeline_init_args(
        has_tokenizer=False, has_feature_extractor=True, has_image_processor=False
    )
)
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

        context_length = kwargs.get("context_length", self.model.config.context_length)
        prediction_length = kwargs.get(
            "prediction_length", self.model.config.prediction_length
        )

        preprocess_kwargs = {}
        postprocess_kwargs = {}
        # id_columns: List[str] = [],
        # timestamp_column: Optional[str] = None,
        # target_columns: List[str] = [],
        # observable_columns: List[str] = [],
        # control_columns: List[str] = [],
        # conditional_columns: List[str] = [],
        # static_categorical_columns: List[str] = [],

        preprocess_params = [
            "id_columns",
            "timestamp_column",
            "target_columns",
            "observable_columns",
            "control_columns",
            "conditional_columns",
            "static_categorical_columns",
            "prediction_length",
            "context_length",
            "future_time_series",
        ]
        postprocess_params = [
            "id_columns",
            "timestamp_column",
            "target_columns",
            "observable_columns",
            "control_columns",
            "conditional_columns",
            "static_categorical_columns",
            "prediction_length",
            "context_length",
        ]

        for c in preprocess_params:
            if c in kwargs:
                preprocess_kwargs[c] = kwargs[c]

        for c in postprocess_params:
            if c in kwargs:
                postprocess_kwargs[c] = kwargs[c]

        # if "id_columns" in kwargs:
        #     preprocess_kwargs["id_columns"] = kwargs["id_columns"]
        #     postprocess_kwargs["id_columns"] = kwargs["id_columns"]
        # if "timestamp_column" in kwargs:
        #     preprocess_kwargs["timestamp_column"] = kwargs["timestamp_column"]
        #     postprocess_kwargs["timestamp_column"] = kwargs["timestamp_column"]
        # if "input_columns" in kwargs:
        #     preprocess_kwargs["input_columns"] = kwargs["input_columns"]
        #     postprocess_kwargs["input_columns"] = kwargs["input_columns"]
        # if "output_columns" in kwargs:
        #     preprocess_kwargs["output_columns"] = kwargs["output_columns"]
        #     postprocess_kwargs["output_columns"] = kwargs["output_columns"]
        # elif "input_columns" in kwargs:
        #     preprocess_kwargs["output_columns"] = kwargs["input_columns"]
        #     postprocess_kwargs["output_columns"] = kwargs["input_columns"]

        return preprocess_kwargs, {}, postprocess_kwargs

    def __call__(
        self,
        time_series: Union["pandas.DataFrame", str],
        **kwargs,
    ):
        """Main method of the forecasting pipeline. Takes the input time series data (in tabular format) and
        produces predictions.

        Args:
            time_series (Union[&quot;pandas.DataFrame&quot;, str]): A pandas dataframe or a referce to a location
            from where a pandas datarame can be loaded containing the time series on which to perform inference.

            future_time_series (Union[&quot;pandas.DataFrame&quot;, str]): A pandas dataframe or a referce to a location
            from where a pandas datarame can be loaded containing future values, i.e., exogenous or supporting features
            which are known in advance.

            To do: describe batch vs. single and the need for future_time_series


            kwargs

            future_time_series: Optional[Union["pandas.DataFrame", str]] = None,
            prediction_length
            context_length

            timestamp_column (str): the column containing the date / timestamp
            id_columns (List[str]): the list of columns containing ID information. If no ids are present, pass [].

            "target_columns",
            "observable_columns",
            "control_columns",
            "conditional_columns",
            "static_categorical_columns",


            # OLD
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

    def preprocess(
        self, time_series, **kwargs
    ) -> Dict[str, Union[GenericTensor, List[Any]]]:
        """Preprocess step
        Load the data, if not already loaded, and then generate a pytorch dataset.
        """

        if isinstance(time_series, str):
            time_series = pd.read_csv(
                time_series,
                parse_dates=[kwargs["timestamp_column"]],
            )

        future_time_series = kwargs.pop("future_time_series", None)
        if future_time_series is not None:
            if isinstance(future_time_series, str):
                future_time_series = pd.read_csv(
                    future_time_series,
                    parse_dates=[kwargs["timestamp_column"]],
                )

            # stack the time series
            for c in future_time_series.columns:
                if c not in time_series.columns:
                    raise ValueError(
                        f"Future time series input contains an unknown column {c}"
                    )

            time_series = pd.concat((time_series, future_time_series), axis=0)

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
            "static_categorical_values",
            "freq_token",
        }  # todo: this should not be hardcoded
        model_inputs_only = {}
        for k in model_input_keys:
            if k in model_inputs:
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

        # name the predictions of target columns
        # outputs should only have size equal to target columns
        for i, c in enumerate(kwargs["target_columns"]):
            out[f"{c}_prediction"] = input[model_output_key][:, :, i].numpy().tolist()
        # provide the ground truth values for the targets
        # when future is unknown, we will have augmented the provided dataframe with NaN values to cover the future
        for i, c in enumerate(kwargs["target_columns"]):
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


def augment_time_series(
    time_series: pd.DataFrame,
    start_timestamp,
    column_name: str,
    grouping_columns: List[str],
    periods: int = 1,
    delta: datetime.timedelta = datetime.timedelta(days=1),
):

    def augment_one_series(group: pd.Series):
        return pd.concat(
            (
                group,
                pd.DataFrame(
                    {
                        column_name: pd.date_range(
                            start_timestamp + delta,
                            freq=delta,
                            periods=periods,
                        )
                    }
                ),
            ),
            axis=0,
        )

    new_time_series = time_series.groupby(group_keys=grouping_columns).apply(
        augment_one_series
    )

    return new_time_series
