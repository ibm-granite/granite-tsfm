# Copyright contributors to the TSFM project
#
"""Hugging Face Pipeline for Time Series Tasks"""

from typing import Any, Dict, List, Union

import pandas as pd
from transformers import PreTrainedModel
from transformers.pipelines.base import (
    GenericTensor,
    build_pipeline_init_args,
)
from transformers.utils import add_end_docstrings, logging

from .dataset import ClassificationDFDataset
from .time_series_forecasting_pipeline import TimeSeriesPipeline


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
    build_pipeline_init_args(has_tokenizer=False, has_feature_extractor=True, has_image_processor=False)
)
class TimeSeriesClassificationPipeline(TimeSeriesPipeline):
    """
    Time Series Classification using HF time series forecasting models. This pipeline consumes a `pandas.DataFrame`
    containing the time series data and produces a new `pandas.DataFrame` containing the classification results.

    """

    def __init__(
        self,
        model: Union["PreTrainedModel"],
        *args,
        explode_forecasts: bool = False,
        inverse_scale_outputs: bool = True,
        add_known_ground_truth: bool = True,
        **kwargs,
    ):
        # autopopulate from feature extractor and model
        if "feature_extractor" in kwargs:
            for p in [
                "id_columns",
                "timestamp_column",
                "input_columns",
                "label_column",
                # "observable_columns",
                # "control_columns",
                # "conditional_columns",
                # "categorical_columns",
                "static_categorical_columns",
                # "freq",
            ]:
                if p not in kwargs:
                    kwargs[p] = getattr(kwargs["feature_extractor"], p)

        if "context_length" not in kwargs:
            kwargs["context_length"] = model.config.context_length

        # check if we need to use the frequency token, get token if needed
        # use_frequency_token = getattr(model.config, "resolution_prefix_tuning", False)

        # if use_frequency_token and "feature_extractor" in kwargs:
        #     kwargs["frequency_token"] = kwargs["feature_extractor"].get_frequency_token(kwargs["freq"])
        # else:
        #     kwargs["frequency_token"] = None

        super().__init__(model, *args, **kwargs)

        self.__context_memory = {}

        if self.framework == "tf":
            raise ValueError(f"The {self.__class__} is only available in PyTorch.")

        # self.check_model_type(MODEL_FOR_TIME_SERIES_FORECASTING_MAPPING)

    def _sanitize_parameters(
        self,
        **kwargs,
    ):
        """Assigns parameters to the different steps of the process. If context_length and prediction_length
        are not provided they are taken from the model config.

        For expected parameters see the call method below.
        """

        preprocess_kwargs = {}
        postprocess_kwargs = {}

        preprocess_params = [
            "prediction_length",
            "context_length",
            # "frequency_token",
            "id_columns",
            "timestamp_column",
            "input_columns",
            "label_column",
            "static_categorical_columns",
        ]
        postprocess_params = [
            "prediction_length",
            "context_length",
            "id_columns",
            "timestamp_column",
            "input_columns",
            "label_column",
            "static_categorical_columns",
            # "freq",
        ]

        for c in preprocess_params:
            if c in kwargs:
                preprocess_kwargs[c] = kwargs[c]

        for c in postprocess_params:
            if c in kwargs:
                postprocess_kwargs[c] = kwargs[c]

        # same logic as HF Pipeline
        batch_size = kwargs.get("batch_size", self._batch_size)
        if batch_size is None:
            if self._batch_size is None:
                batch_size = 1
            else:
                batch_size = self._batch_size

        num_workers = kwargs.get("num_workers", self._num_workers)
        if num_workers is None:
            if self._num_workers is None:
                num_workers = 0
            else:
                num_workers = self._num_workers

        forward_kwargs = {"batch_size": batch_size, "num_workers": num_workers}

        return preprocess_kwargs, forward_kwargs, postprocess_kwargs

    def __call__(
        self,
        time_series: Union["pd.DataFrame", str],
        **kwargs,
    ):
        """Main method of the forecasting pipeline. Takes the input time series data (in tabular format) and
        produces predictions.

        Args:
            time_series (Union[&quot;pandas.DataFrame&quot;, str]): A pandas dataframe containing the time series on
                which to perform inference.

        Keyword arguments:
            future_time_series (Union[&quot;pandas.DataFrame&quot;, str]): A pandas dataframe containing future values,
                i.e., exogenous or supporting features which are known in advance.

            feature_extractor (TimeSeriesPreprocessor): A time series preprpocessor object that specifies how the time
                series should be prepared. If this is provided, any of the other options below will be automatically
                populated from this instance.

            timestamp_column (str): The name of the column containing the timestamp of the time series.

            id_columns (List[str]): List of column names which identify different time series in a multi-time series input.

            input_columns (List[str]): List of column names which identify the target channels in the input, these are the
                columns that will be forecasted.

            label_column (str): Column containing time series label

            static_categorical_columns (List[str]): List of column names which identify categorical-valued channels in the input
                which are fixed over time.

            freq (str): A freqency indicator for the given `timestamp_column`. See
                https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#period-aliases for a description of the
                allowed values. If not provided, we will attempt to infer it from the data. If not provided, frequency will be
                inferred from `timestamp_column`.

            prediction_length (int): The length of the desired forecast. Currently, this value must not exceed the maximum value
                suported by the model. If not specified, the maximum value supported by the model is used.

            context_length (int): Specifies the length of the context windows extracted from the historical data for feeding into
                the model.

            explode_forecasts (bool): If true, forecasts are returned one value per row of the pandas dataframe. If false, the
                forecast over the prediction length will be contained as a list in a single row of the pandas dataframe.

            inverse_scale_outputs (bool): If true and a valid feature extractor is provided, the outputs will be inverse scaled.

            add_known_ground_truth (bool): If True add columns containing the ground truth data. Prediction columns will have a
                suffix of "_prediction". Default True. If false, on columns containing predictions are produced, no suffix is
                added.

        Return (pandas dataframe):
            A new pandas dataframe containing the forecasts. Each row will contain the id, timestamp, the original
            input feature values and the output forecast for each input column. The output forecast is a list containing
            all the values over the prediction horizon.

        """

        return super().__call__(time_series, **kwargs)

    def preprocess(self, time_series, **kwargs) -> Dict[str, Union[GenericTensor, List[Any]]]:
        """Preprocess step
        Load the data, if not already loaded, and then generate a pytorch dataset.
        """

        timestamp_column = kwargs.get("timestamp_column")
        # id_columns = kwargs.get("id_columns")
        # input_columns = kwargs.get("input_columns")
        # label_column = kwargs.get("label_column")
        # context_length = kwargs.get("context_length")

        # use the feature extractor here

        if isinstance(time_series, str):
            time_series = pd.read_csv(
                time_series,
                parse_dates=[timestamp_column],
            )

        if self.feature_extractor:
            time_series_prep = self.feature_extractor.preprocess(time_series)
        else:
            time_series_prep = time_series

        # add full_series option here
        kwargs["full_series"] = True

        # prepare classification dataset
        dataset = ClassificationDFDataset(
            time_series_prep,
            **kwargs,
        )

        self.__context_memory["data"] = time_series

        return {"dataset": dataset}

    def _forward(self, model_inputs, **kwargs):
        """Forward step
        Responsible for taking pre-processed dictionary of tensors and passing it to
        the model. Aligns model parameters with the proper input parameters. Only passes
        the needed parameters from the dictionary to the model, but adds them back to the
        ouput for the next step.

        The keys in model_outputs are governed by the underlying model combined with any
        original input keys.
        """

        model_outputs = self.model(**model_inputs)

        return model_outputs

    def postprocess(self, input, **kwargs):
        """Postprocess step
        Takes the dictionary of outputs from the previous step and converts to a more user
        readable pandas format.

        If the explode forecasts option is True, then individual forecasts are expanded as multiple
        rows in the dataframe. This should only be used when producing a single forecast (i.e., unexploded
        result is one row per ID).
        """
        out = {}

        model_output_key = "prediction_outputs"  #  if "prediction_outputs" in input.keys() else "prediction_logits"

        # name the predictions of target columns
        # outputs should only have size equal to target columns

        if kwargs.get("timestamp_column", None):
            out[kwargs["timestamp_column"]] = input["timestamp"]
        for i, c in enumerate(kwargs["id_columns"]):
            out[c] = [elem[i] for elem in input["id"]]

        # input series
        for c in kwargs["input_columns"]:
            input_data = self.__context_memory["data"]
            out[c] = input_data[c]

        # ground truth
        column = kwargs["label_column"]
        out[column] = input["target_values"].numpy().tolist()

        # predictions
        column = f"{kwargs['label_column']}_prediction"
        out[column] = input[model_output_key].numpy().argmax(axis=1).tolist()

        out = pd.DataFrame(out)

        # inverse transform labels
        if self.feature_extractor is not None:
            out = self.feature_extractor.inverse_transform_labels(out)
            out = self.feature_extractor.inverse_transform_labels(out, suffix="_prediction")

        self.__context_memory = {}
        return out
