# Copyright contributors to the TSFM project
#
"""Hugging Face Pipeline for Time Series Tasks"""

import inspect
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers.data.data_collator import default_data_collator
from transformers.pipelines.base import (
    GenericTensor,
    Pipeline,
    build_pipeline_init_args,
)
from transformers.trainer_utils import RemoveColumnsCollator
from transformers.utils import add_end_docstrings, logging

from .dataset import ForecastDFDataset
from .time_series_preprocessor import create_timestamps, extend_time_series


# Eventually we should support all time series models
# MODEL_FOR_TIME_SERIES_FORECASTING_MAPPING_NAMES = OrderedDict(
#     [
#         # Model for Time Series Forecasting
#         ("PatchTST", "PatchTSTForPrediction"),
#         ("TST", "TimeSeriesTransformerForPrediction"),
#     ]
# )


logger = logging.get_logger(__name__)


class TimeSeriesPipeline(Pipeline):
    def run_single(self, inputs, preprocess_params, forward_params, postprocess_params):
        """Replaces base `run_single` method which does batching during inference. This is needed to support
        large inference requests.

        Args:
            inputs (_type_): _description_
            preprocess_params (_type_): _description_
            forward_params (_type_): _description_
            postprocess_params (_type_): _description_

        Returns:
            _type_: _description_
        """
        # our preprocess returns a dataset
        dataset = self.preprocess(inputs, **preprocess_params)

        batch_size = forward_params["batch_size"]
        signature = inspect.signature(self.model.forward)
        signature_columns = list(signature.parameters.keys())

        # if len(dataset) < batch_size:
        # build a dataloader
        # collate_fn = no_collate_fn if batch_size == 1 else pad_collate_fn(self.tokenizer, feature_extractor)

        remove_columns_collator = RemoveColumnsCollator(
            data_collator=default_data_collator,
            signature_columns=signature_columns,
            logger=None,
            description=None,
            model_name=self.model.__class__.__name__,
        )
        dataloader = DataLoader(
            dataset, num_workers=1, batch_size=batch_size, collate_fn=remove_columns_collator, shuffle=False
        )

        # iterate over dataloader
        it = iter(dataloader)
        accumulator = []
        model_output_key = None
        while (batch := next(it, None)) is not None:
            item = self.forward(batch, **forward_params)
            if not model_output_key:
                model_output_key = "prediction_outputs" if "prediction_outputs" in item.keys() else "prediction_logits"
            accumulator.append(item[model_output_key])

        # collect all ouputs needed for post processing
        first = dataset[0]
        model_outputs = {}
        for k, v in first.items():
            if isinstance(v, torch.Tensor):
                model_outputs[k] = torch.stack(tuple(r[k] for r in dataset))
            else:
                model_outputs[k] = [r[k] for r in dataset]

        # without shuffling in the dataloader above, we assume that order is preserved
        # otherwise we need to incorporate sequence id somewhere and do a proper join
        model_outputs["prediction_outputs"] = torch.cat(accumulator, axis=0)

        # call postprocess
        outputs = self.postprocess(model_outputs, **postprocess_params)

        return outputs


@add_end_docstrings(
    build_pipeline_init_args(has_tokenizer=False, has_feature_extractor=True, has_image_processor=False)
)
class TimeSeriesForecastingPipeline(TimeSeriesPipeline):
    """Hugging Face Pipeline for Time Series Forecasting

    feature_extractor (TimeSeriesPreprocessor): A time series preprpocessor object that specifies how the time
            series should be prepared. If this is provided, and of the other options below will be automatically
            populated from this instance.
    """

    def __init__(
        self,
        *args,
        freq: Optional[str] = None,
        explode_forecasts: bool = False,
        inverse_scale_outputs: bool = True,
        **kwargs,
    ):
        kwargs["freq"] = freq
        kwargs["explode_forecasts"] = explode_forecasts
        kwargs["inverse_scale_outputs"] = inverse_scale_outputs
        super().__init__(*args, **kwargs)

        if self.framework == "tf":
            raise ValueError(f"The {self.__class__} is only available in PyTorch.")

        # self.check_model_type(MODEL_FOR_TIME_SERIES_FORECASTING_MAPPING)

    def _sanitize_parameters(
        self,
        **kwargs,
    ):
        """Assign parameters to the different parts of the process.

        For expected parameters see the call method below.
        """

        context_length = kwargs.get("context_length", self.model.config.context_length)
        prediction_length = kwargs.get("prediction_length", self.model.config.prediction_length)

        preprocess_kwargs = {
            "prediction_length": prediction_length,
            "context_length": context_length,
        }
        postprocess_kwargs = {
            "prediction_length": prediction_length,
            "context_length": context_length,
        }

        preprocess_params = [
            "id_columns",
            "timestamp_column",
            "target_columns",
            "observable_columns",
            "control_columns",
            "conditional_columns",
            "static_categorical_columns",
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
            "freq",
            "explode_forecasts",
            "inverse_scale_outputs",
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

        forward_kwargs = {"batch_size": batch_size}

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
                series should be prepared. If this is provided, and of the other options below will be automatically
                populated from this instance.

            timestamp_column (str): The name of the column containing the timestamp of the time series.

            id_columns (List[str]): List of column names which identify different time series in a multi-time series input.

            target_columns (List[str]): List of column names which identify the target channels in the input, these are the
                columns that will be forecasted.

            observable_columns (List[str]): List of column names which identify the observable channels in the input.
                Observable channels are channels which we have knowledge about in the past and future. For example, weather
                conditions such as temperature or precipitation may be known or estimated in the future, but cannot be
                changed.

            control_columns (List[str]): List of column names which identify the control channels in the input. Control
                channels are similar to observable channels, except that future values may be controlled. For example, discount
                percentage of a particular product is known and controllable in the future.

            conditional_columns (List[str]): List of column names which identify the conditional channels in the input.
                Conditional channels are channels which we know in the past, but do not know in the future.

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

        prediction_length = kwargs.get("prediction_length")
        timestamp_column = kwargs.get("timestamp_column")
        id_columns = kwargs.get("id_columns")
        # context_length = kwargs.get("context_length")

        if isinstance(time_series, str):
            time_series = pd.read_csv(
                time_series,
                parse_dates=[timestamp_column],
            )

        future_time_series = kwargs.pop("future_time_series", None)

        if future_time_series is not None:
            if isinstance(future_time_series, str):
                future_time_series = pd.read_csv(
                    future_time_series,
                    parse_dates=[timestamp_column],
                )
            elif isinstance(future_time_series, pd.DataFrame):
                # do we need to check the timestamp column?
                pass
            else:
                raise ValueError(f"`future_time_series` of type {type(future_time_series)} is not supported.")

            # stack the time series
            for c in future_time_series.columns:
                if c not in time_series.columns:
                    raise ValueError(f"Future time series input contains an unknown column {c}.")

            time_series = pd.concat((time_series, future_time_series), axis=0)
        else:
            # no additional exogenous data provided, extend with empty periods
            time_series = extend_time_series(
                time_series=time_series,
                timestamp_column=timestamp_column,
                grouping_columns=id_columns,
                periods=prediction_length,
            )

        # use forecasing dataset to do the preprocessing
        dataset = ForecastDFDataset(
            time_series,
            **kwargs,
        )

        # # stack all the outputs
        # # torch tensors are stacked, but other values are passed through as a list
        # first = dataset[0]
        # full_output = {}
        # for k, v in first.items():
        #     if isinstance(v, torch.Tensor):
        #         full_output[k] = torch.stack(tuple(r[k] for r in dataset))
        #     else:
        #         full_output[k] = [r[k] for r in dataset]

        # return full_output
        return dataset

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
        # model_input_keys = {
        #     "past_values",
        #     "static_categorical_values",
        #     "freq_token",
        # }  # todo: this should not be hardcoded

        # signature = inspect.signature(self.model.forward)
        # model_input_keys = list(signature.parameters.keys())

        # model_inputs_only = {}
        # for k in model_input_keys:
        #     if k in model_inputs:
        #         model_inputs_only[k] = model_inputs[k]

        # model_outputs = self.model(**model_inputs_only)

        # # copy the other inputs
        # copy_inputs = True
        # for k in [akey for akey in model_inputs.keys() if (akey not in model_input_keys) or copy_inputs]:
        #     model_outputs[k] = model_inputs[k]

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
        prediction_columns = []
        for i, c in enumerate(kwargs["target_columns"]):
            prediction_columns.append(f"{c}_prediction")
            out[prediction_columns[-1]] = input[model_output_key][:, :, i].numpy().tolist()
        # provide the ground truth values for the targets
        # when future is unknown, we will have augmented the provided dataframe with NaN values to cover the future
        for i, c in enumerate(kwargs["target_columns"]):
            out[c] = input["future_values"][:, :, i].numpy().tolist()

        if "timestamp_column" in kwargs:
            out[kwargs["timestamp_column"]] = input["timestamp"]
        for i, c in enumerate(kwargs["id_columns"]):
            out[c] = [elem[i] for elem in input["id"]]
        out = pd.DataFrame(out)

        if kwargs["explode_forecasts"]:
            # we made only one forecast per time series, explode results
            # explode == expand the lists in the dataframe
            out_explode = []
            for _, row in out.iterrows():
                l = len(row[prediction_columns[0]])
                tmp = {}
                if "timestamp_column" in kwargs:
                    tmp[kwargs["timestamp_column"]] = create_timestamps(
                        row[kwargs["timestamp_column"]], freq=kwargs["freq"], periods=l
                    )  # expand timestamps
                if "id_columns" in kwargs:
                    for c in kwargs["id_columns"]:
                        tmp[c] = row[c]
                for p in prediction_columns:
                    tmp[p] = row[p]

                out_explode.append(pd.DataFrame(tmp))

            out = pd.concat(out_explode)

        # reorder columns
        cols = out.columns.to_list()
        cols_ordered = []
        if "timestamp_column" in kwargs:
            cols_ordered.append(kwargs["timestamp_column"])
        if "id_columns" in kwargs:
            cols_ordered.extend(kwargs["id_columns"])
        cols_ordered.extend([c for c in cols if c not in cols_ordered])

        out = out[cols_ordered]

        # inverse scale if we have a feature extractor
        if self.feature_extractor is not None and kwargs["inverse_scale_outputs"]:
            out = self.feature_extractor.inverse_scale_targets(out, suffix="_prediction")

        return out
