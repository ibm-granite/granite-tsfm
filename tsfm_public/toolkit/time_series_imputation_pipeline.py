# Copyright contributors to the TSFM project
#
"""Hugging Face Pipeline for Time Series Imputation"""

from collections import OrderedDict
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import torch
from transformers import PreTrainedModel
from transformers.pipelines.base import (
    GenericTensor,
    build_pipeline_init_args,
)
from transformers.utils import add_end_docstrings, logging

from .dataset import ForecastDFDataset
from .time_series_forecasting_pipeline import TimeSeriesPipeline


logger = logging.get_logger(__name__)

# Eventually we should support all time series models
MODEL_FOR_TIME_SERIES_IMPUTATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Time Series Imputation
        ("TSPulse", "TSPulseForReconstruction"),
    ]
)


@add_end_docstrings(
    build_pipeline_init_args(has_tokenizer=False, has_feature_extractor=True, has_image_processor=False)
)
class TimeSeriesImputationPipeline(TimeSeriesPipeline):
    """
    Time Series Forecasting using HF time series forecasting models. This pipeline consumes a `pandas.DataFrame`
    containing the time series data and produces a new `pandas.DataFrame` containing the forecasts.

    """

    def __init__(
        self,
        model: Union["PreTrainedModel"],
        *args,
        inverse_scale_outputs: bool = True,
        add_known_ground_truth: bool = True,
        **kwargs,
    ):
        kwargs["inverse_scale_outputs"] = inverse_scale_outputs
        kwargs["add_known_ground_truth"] = add_known_ground_truth

        # autopopulate from feature extractor and model
        if "feature_extractor" in kwargs:
            for p in [
                "id_columns",
                "timestamp_column",
                "target_columns",
                # "observable_columns",
                # "control_columns",
                # "conditional_columns",
                # "categorical_columns",
                # "static_categorical_columns",
            ]:
                if p not in kwargs:
                    kwargs[p] = getattr(kwargs["feature_extractor"], p)

        if "context_length" not in kwargs:
            kwargs["context_length"] = model.config.context_length

        super().__init__(model, *args, **kwargs)

        self.__context_memory = {}
        # these control the shared run_single method
        self._model_output_key = "reconstruction_outputs"
        self._copy_dataset_keys = False

        if self.framework == "tf":
            raise ValueError(f"The {self.__class__} is only available in PyTorch.")

        self.check_model_type(MODEL_FOR_TIME_SERIES_IMPUTATION_MAPPING_NAMES)

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
            "context_length",
            "id_columns",
            "timestamp_column",
            "target_columns",
            # "observable_columns",
            # "control_columns",
            # "conditional_columns",
            # "categorical_columns",
            # "static_categorical_columns",
            "impute_method",
        ]
        postprocess_params = [
            "prediction_length",
            "context_length",
            "id_columns",
            "timestamp_column",
            "target_columns",
            # "observable_columns",
            # "control_columns",
            # "conditional_columns",
            # "categorical_columns",
            # "static_categorical_columns",
            "inverse_scale_outputs",
            "add_known_ground_truth",
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
        """Main method of the imputation pipeline. Takes the input time series data (in tabular format) and
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

            context_length (int): Specifies the length of the context windows extracted from the historical data for feeding into
                the model.

            inverse_scale_outputs (bool): If true and a valid feature extractor is provided, the outputs will be inverse scaled.

            add_known_ground_truth (bool): If True add columns containing the ground truth data (possibly containing missing NaN values) to the imputed columns. Imputed columns will have a
                suffix of "_imputed". These columns have original non-missing values from the ground truth and imputed (reconstructed)
                values from the model at the missing positions in the ground truth. Default True. If False, only columns containing
                imputed values in place of missing NaN values and original non-missing values intact are produced, no suffix is added.

        Return (pandas dataframe):
            A new pandas dataframe containing the imputed series. Each row will contain the id, timestamp, the original
            input feature values and the output forecast for each input column. The output forecast is a list containing
            all the values over the prediction horizon.

        """

        return super().__call__(time_series, **kwargs)

    def preprocess(self, time_series, **kwargs) -> Dict[str, Union[GenericTensor, List[Any]]]:
        """Preprocess step
        Load the data, if not already loaded, and then generate a pytorch dataset.
        """

        timestamp_column = kwargs.get("timestamp_column")

        if isinstance(time_series, str):
            time_series = pd.read_csv(
                time_series,
                parse_dates=[timestamp_column],
            )

        if self.feature_extractor:
            time_series_prep = self.feature_extractor.preprocess(time_series)

        # use forecasting dataset to do the preprocessing
        dataset = ForecastDFDataset(
            time_series_prep,
            prediction_length=0,
            **kwargs,
        )

        # save our series to make life easier
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

        out = self.__context_memory["data"].copy()  # original dataframe
        # input is a list of tensors: bs x context_length x features
        input = torch.cat(input, axis=0).detach().cpu().numpy()

        # stitching logic
        n_batches = input.shape[0]
        n_obs = input.shape[1]
        total_length = n_batches + n_obs - 1
        data_dim = (total_length, input.shape[2]) if input.ndim == 3 else total_length
        counters = np.zeros(data_dim)
        predictions = np.zeros(data_dim)
        for i in range(n_batches):
            predictions[i : (i + n_obs)] += input[i]
            counters[i : (i + n_obs)] += 1
        reconstructed_out = predictions / np.maximum(counters, 1)  # this output is all reconstructions from the model

        reconstructed_df = pd.DataFrame(reconstructed_out, columns=kwargs["target_columns"])

        # inverse scale if we have a feature extractor
        if self.feature_extractor is not None and kwargs["inverse_scale_outputs"]:
            reconstructed_df = self.feature_extractor.inverse_scale_targets(reconstructed_df)

        # need to select original values for non-missing points and use the reconstructed values only for missing points
        reconstructed_df.index = out.index
        imputed_output = out.where(~out.isna(), reconstructed_df)

        dtype_map = {col: out[col].dtype for col in out.columns}
        imputed_output = imputed_output.astype(
            dtype_map
        )  # setting the type of imputed columns to same as the input data columns

        if kwargs["add_known_ground_truth"]:
            imputed_output = imputed_output[kwargs["target_columns"]]
            imputed_output = imputed_output.add_suffix("_imputed")
            out = pd.concat([out, imputed_output], axis=1)
        else:
            out = imputed_output

        self.__context_memory = {}
        return out
