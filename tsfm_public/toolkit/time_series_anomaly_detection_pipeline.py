# Copyright contributors to the TSFM project
#
"""Hugging Face Pipeline for Time Series Anomaly Detection"""

import inspect
from collections import defaultdict
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch import nn as nn
from torch.utils.data import DataLoader
from transformers.data.data_collator import default_data_collator
from transformers.modeling_utils import PreTrainedModel
from transformers.pipelines.base import GenericTensor, build_pipeline_init_args
from transformers.trainer_utils import RemoveColumnsCollator
from transformers.utils.doc import add_end_docstrings
from transformers.utils.generic import ModelOutput

from tsfm_public.models.tinytimemixer.modeling_tinytimemixer import TinyTimeMixerForPrediction
from tsfm_public.models.tinytimemixer.utils.ad_helpers import TinyTimeMixerADUtility
from tsfm_public.models.tspulse.modeling_tspulse import TSPulseForReconstruction
from tsfm_public.models.tspulse.utils.ad_helpers import TSPulseADUtility
from tsfm_public.toolkit.conformal import PostHocProbabilisticProcessor

from .ad_helpers import AnomalyScoreMethods
from .dataset import ForecastDFDataset
from .time_series_forecasting_pipeline import TimeSeriesPipeline


class AggregationFunction(Enum):
    """Enum type aggregation functions used when combining anomaly scores."""

    MIN = "min"
    MAX = "max"
    MEAN = "mean"


def score_smoothing(
    x: np.ndarray,
    smoothing_window_size: int,
) -> np.ndarray:
    """Utility function for moving average smoothing of N-dimensional dataset.
    Smoothing is applied along axis=0.

    Args:
        x   (np.ndarray): numpy array of arbitrary dimension
        smoothing_window_size (int): parameter specifies moving window size used for smoothing
    """
    if smoothing_window_size < 2:
        return x
    elif x.ndim == 1:
        return np.convolve(x, np.ones(smoothing_window_size) / smoothing_window_size, mode="same")
    else:
        return np.array([score_smoothing(x[..., i], smoothing_window_size) for i in range(x.shape[-1])]).T


@add_end_docstrings(
    build_pipeline_init_args(has_tokenizer=False, has_feature_extractor=True, has_image_processor=False)
)
class TimeSeriesAnomalyDetectionPipeline(TimeSeriesPipeline):
    """Time Series Anomaly Detection using HF time series models. This pipeline consumes a `pandas.DataFrame`
    containing the time series data and produces a new `pandas.DataFrame` with anomaly scores.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        *args,
        prediction_mode: Optional[List[str]] = None,
        aggr_function: str = AggregationFunction.MAX.value,
        aggregation_length: int = 32,
        smoothing_length: int = 8,
        probabilistic_processor: Optional[PostHocProbabilisticProcessor] = None,
        **kwargs,
    ):
        """Huggingface pipeline for time series anomaly detection using time series foundation models.

        Args:
            model (PreTrainedModel): time series foundation model instance
            prediction_mode (list, optional): specify list of appropriate modes for anomaly scoring. Defaults to [AnomalyPredictionModes.PREDICTIVE.value].
            aggr_function (str, optional): aggregation function for merging scores using different mode, supported values are (max/min/mean). Defaults to "max".
            aggregation_length (int, optional): parameter required for imputation or window based scoring. Defaults to 32.
            smoothing_length (int, optional): window size for post processing of the generated scores. Defaults to 8.
            probabilistic_processor (PostHocProbabilisticProcessor, optional): if prediction mode is "probabilistic", use the probabilistic processor to determine the p-values associated with the forecasts from the underlying model. Defaults to None.
            expand_score (bool, optional): if true report anomaly score for each target column separately. Defaults to False.
            report_mode (bool, optional): if true reports which prediction mode is detects higher anomaly score for each observation. Defaults to False.
            predictive_score_smoothing (bool, optional): if true smoothing is applied to the forecast score. Defaults to False.
            least_significant_scale (float, optional): value between (0, 1). Model scores are function of data variance, this factor specifies a relative score threshold to data variance for marking anomaly. Defaults to 0.01.
            least_significant_score (float, optional): minimum score for marking anomaly. Defaults to 0.2.

        Raises:
            ValueError: unsupported model
            ValueError: invalid prediction_mode
            ValueError: no pytorch support
        """
        if prediction_mode is None:
            prediction_mode = [AnomalyScoreMethods.PREDICTIVE.value]

        if isinstance(prediction_mode, str):
            prediction_mode = [prediction_mode]

        model_processor = None
        if isinstance(model, TSPulseForReconstruction):
            model_processor = TSPulseADUtility(
                model, mode=prediction_mode, aggregation_length=aggregation_length, **kwargs
            )
        elif isinstance(model, TinyTimeMixerForPrediction):
            model_processor = TinyTimeMixerADUtility(
                model=model, mode=prediction_mode, probabilistic_processor=probabilistic_processor, **kwargs
            )
        else:
            raise ValueError(f"Error: does not support {self.model.__class__} object!")

        kwargs["context_length"] = model.config.context_length
        kwargs["prediction_length"] = model.config.prediction_length

        if "batch_size" not in kwargs:
            kwargs["batch_size"] = 128

        kwargs["aggregation_length"] = aggregation_length
        kwargs["smoothing_length"] = smoothing_length

        known_mode = model_processor.is_valid_mode(prediction_mode)
        if not known_mode:
            raise ValueError(f"Error: incompatible operation mode {prediction_mode}!")

        # check if we need to use the frequency token, get token if needed
        use_frequency_token = getattr(model.config, "resolution_prefix_tuning", False)

        # needed for TTM support
        if use_frequency_token and "feature_extractor" in kwargs:
            freq = getattr(kwargs["feature_extractor"], "freq", None)
            kwargs["frequency_token"] = (
                kwargs["feature_extractor"].get_frequency_token(freq) if freq is not None else None
            )
        else:
            kwargs["frequency_token"] = None

        kwargs["mode"] = prediction_mode

        self._model_processor = model_processor
        # possibly save the posthoc_processor too?

        super().__init__(model, *args, **kwargs)

        self.__context_memory = {}
        if aggr_function.lower() == AggregationFunction.MIN.value:
            aggr_function_ = np.min
            select_function_ = np.argmin
        elif aggr_function.lower() == AggregationFunction.MEAN.value:
            aggr_function_ = np.mean
            select_function_ = None
        elif aggr_function.lower() == AggregationFunction.MAX.value:
            aggr_function_ = np.max
            select_function_ = np.argmax
        else:
            raise ValueError(
                f"Unsupported aggregation function provided {aggr_function}, expected one of {[c.value for c in AggregationFunction]}"
            )
        self.aggr_function = aggr_function_
        self.select_function = select_function_

        if self.framework == "tf":
            raise ValueError(f"The {self.__class__} is only available in PyTorch.")

    @property
    def model_type(self) -> str:
        """Returns corresponding short name of the associated model class.

        Raises:
            ValueError: unsupported model type.

        Returns:
            str: short name of the model associated with the pipeline
        """
        if isinstance(self.model, TSPulseForReconstruction):
            return "tspulse"
        elif isinstance(self.model, TinyTimeMixerForPrediction):
            return "ttm"
        raise ValueError(f"Error: unsupported model type {self.model.__class__}!")

    def _sanitize_parameters(self, **kwargs):
        """Assigns parameters to the different steps of the process. If context_length and prediction_length
        are not provided they are taken from the model config.

        For expected parameters see the call method below.
        """
        preprocess_kwargs = {}
        postprocess_kwargs = {}

        preprocess_params = [
            "context_length",
            "prediction_length",
            "timestamp_column",
            "target_columns",
            "frequency_token",
        ]
        postprocess_params = [
            "timestamp_column",
            "target_columns",
            "aggregation_length",
            "smoothing_length",
            "expand_score",
            "report_mode",
            "predictive_score_smoothing",
        ]

        for c in preprocess_params:
            if c in kwargs:
                preprocess_kwargs[c] = kwargs[c]

        for c in postprocess_params:
            if c in kwargs:
                postprocess_kwargs[c] = kwargs[c]

        if self.model_type == "tspulse":
            preprocess_kwargs["prediction_length"] = 1  # should not override model setting when TTM

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

        forward_kwargs = {
            "batch_size": batch_size,
            "num_workers": num_workers,
        }

        for p in ["mode", "aggregation_length", "expand_score"]:
            if p in kwargs:
                forward_kwargs[p] = kwargs[p]

        return preprocess_kwargs, forward_kwargs, postprocess_kwargs

    def preprocess(self, input_, **kwargs) -> Dict[str, Union[GenericTensor, List[Any]]]:
        """Preprocess step
        Load the data, if not already loaded, and then generate a pytorch dataset.
        """
        # use the feature extractor here
        if isinstance(input_, str):
            timestamp_column: str = kwargs.get("timestamp_column", "")
            if not isinstance(timestamp_column, str) or (timestamp_column == ""):
                raise ValueError("Error: timestamp column must be specified!")

            input_ = pd.read_csv(
                input_,
                parse_dates=[timestamp_column],
            )

        # For any forecasting based methods
        if self.feature_extractor:
            input_prep = self.feature_extractor.preprocess(input_)
        else:
            input_prep = input_

        # possibly calibrate posthoc_processor if calibration data is passed (not enabled yet)

        # Fixing stride to 1
        kwargs["stride"] = 1

        # maintaining processed data reference
        # CHECK if there is a preprocessor should we pass the preprocessed data here instead?
        processed_input_ = self._model_processor.preprocess(input_prep, **kwargs)

        # use forecasting dataset to do the preprocessing
        dataset = ForecastDFDataset(
            processed_input_,
            **kwargs,
        )
        self.__context_memory["data"] = input_
        self.__context_memory["reference"] = processed_input_
        return {"dataset": dataset}

    def run_single(self, inputs, preprocess_params, forward_params, postprocess_params) -> pd.DataFrame:
        """Replaces base `run_single` method which does batching during inference. This is needed to support
        large inference requests.

        Args:
            inputs (str | pd.DataFrame): data input
            preprocess_params (dict): required parameters for data preprocessing
            forward_params (dict): required parameters for model evaluation
            postprocess_params (dict): required parameters for output post processing
        Returns:
            pd.DataFrame: pipeline output
        """
        dataset = self.preprocess(inputs, **preprocess_params)["dataset"]
        signature = inspect.signature(self.model.forward)
        signature_columns = list(signature.parameters.keys())

        remove_columns_collator = RemoveColumnsCollator(
            data_collator=default_data_collator,
            signature_columns=signature_columns,
            logger=None,
            description=None,
            model_name=None,
        )

        batch_size = forward_params.get("batch_size")
        num_workers = forward_params.get("num_workers")
        dataloader = DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=remove_columns_collator, shuffle=False
        )

        it = iter(dataloader)
        accumulator = defaultdict(list)

        with torch.no_grad():  # check if really needed
            while (batch := next(it, None)) is not None:
                scores = self.forward(batch, **forward_params)
                for key in scores:
                    accumulator[key].append(scores[key])

        # call postprocess
        outputs = self.postprocess(ModelOutput(accumulator), **postprocess_params)

        return outputs

    def _forward(self, input_tensors, **kwargs):
        """Forward step
        Responsible for taking pre-processed dictionary of tensors and passing it to
        the model. Aligns model parameters with the proper input parameters. Only passes
        the needed parameters from the dictionary to the model, but adds them back to the
        ouput for the next step.

        The keys in model_outputs are governed by the underlying model combined with any
        original input keys.
        """
        return self._model_processor.compute_score(input_tensors, **kwargs)

    def postprocess(self, model_outputs, **postprocess_parameters):
        """Overrides the postprocess of the base class. Applies post-processing logic on the model outputs.

        Args:
            model_outputs (dict): dictionary containing model outputs.

        Raises:
            RuntimeError: Returned if there is an inconsistency in the target columns and the resulting scores.

        Returns:
            pd.DataFrame: pandas dataframe with anomaly score attached
        """
        result = self.__context_memory["data"].copy()
        expand_score = postprocess_parameters.get("expand_score", False)
        smoothing_window_size = postprocess_parameters.get("smoothing_length", 1)
        target_columns = postprocess_parameters.get("target_columns")

        report_mode = postprocess_parameters.get("report_mode", False)
        predictive_score_smoothing = postprocess_parameters.get("predictive_score_smoothing", False)
        if not isinstance(smoothing_window_size, int):
            try:
                smoothing_window_size = int(smoothing_window_size)
            except ValueError:
                smoothing_window_size = 1

        # adjust scoring and smooth
        extra_kwargs = {}
        if "reference" in self.__context_memory:
            data = self.__context_memory["reference"]
            if len(target_columns) > 0:
                data = data[target_columns]
            extra_kwargs["reference"] = data.values

        model_outputs_ = {}
        for k in model_outputs:
            score = model_outputs[k]
            score = self._model_processor.adjust_boundary(k, score, **extra_kwargs)
            if not predictive_score_smoothing and (
                k == AnomalyScoreMethods.PREDICTIVE.value
            ):  # Skip Smoothing For 1 Lookahead forecast
                model_outputs_[k] = score
            elif k == AnomalyScoreMethods.PROBABILISTIC.value:
                model_outputs_[k] = score_smoothing(
                    score, smoothing_window_size=1
                )  # no smoothing of p-value scores across time
            else:
                model_outputs_[k] = score_smoothing(score, smoothing_window_size=smoothing_window_size)

        # aggregate scores and expand
        score = np.stack([score_ for _, score_ in model_outputs_.items()], axis=0)
        mode_selected = None
        if report_mode and (self.select_function is not None):
            keys = [key for key, _ in model_outputs_.items()]
            sel_index = self.select_function(score, axis=0)
            mode_selected = np.asarray([keys[z] for z in sel_index.ravel()]).reshape(sel_index.shape)

        score = self.aggr_function(score, axis=0)

        expand_score = (len(target_columns) > 1) and expand_score
        model_outputs = {}
        if expand_score:
            if len(target_columns) != score.shape[-1]:
                raise RuntimeError(f"Error: inconsistent state, with target columns {target_columns}")
            for i, col_name in enumerate(target_columns):
                model_outputs[f"{col_name}_anomaly_score"] = score[..., i]

            if mode_selected is not None:
                for i, col_name in enumerate(target_columns):
                    model_outputs[f"{col_name}_selected_mode"] = mode_selected[..., i]
        else:
            model_outputs.update(anomaly_score=score.ravel())
            if mode_selected is not None:
                model_outputs.update(selected_mode=mode_selected.ravel())

        # populate dataframe
        for k in model_outputs:
            result[k] = model_outputs[k]
        self.__context_memory = {}
        return result
