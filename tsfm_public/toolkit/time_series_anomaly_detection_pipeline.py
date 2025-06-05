import inspect
from collections import OrderedDict, defaultdict
from enum import Enum
from typing import Any, Dict, List, Union

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

from .dataset import ForecastDFDataset
from .time_series_forecasting_pipeline import TimeSeriesPipeline


class AnomalyPredictionModes(Enum):
    """Enum type for time series foundation model based anomaly detection modes."""

    MEAN_DEVIATION = "meandev"
    PREDICTIVE = "forecast"
    TIME_IMPUTATION = "time"
    FREQUENCY_IMPUTATION = "fft"
    TIME_AND_FREQUENCY_IMPUTATION = "time+fft"
    PREDICTIVE_WITH_TIME_IMPUTATION = "forecast+time"
    PREDICTIVE_WITH_FREQUENCY_IMPUTATION = "forecast+fft"
    PREDICTIVE_WITH_IMPUTATION = "forecast+time+fft"


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
        prediction_mode: str = AnomalyPredictionModes.PREDICTIVE.value,
        aggr_function: str = "max",
        aggr_win_size: int = 32,
        smoothing_window_size: int = 8,
        **kwargs,
    ):
        """Huggingface pipeline for time series anomaly detection using time series foundation models.

        Args:
            model (PreTrainedModel): time series foundation model instance
            prediction_mode (str, optional): specify appropriate mode for anomaly scoring. Defaults to AnomalyPredictionModes.PREDICTIVE.value.
            aggr_function (str, optional): aggregation function for merging scores using different mode, supported values are (max/min/mean). Defaults to "max".
            aggr_win_size (int, optional): parameter required for imputation or window based scoring. Defaults to 32.
            smoothing_window_size (int, optional): window size for post processing of the generated scores. Defaults to 8.

        Raises:
            ValueError: unsupported model
            ValueError: invalid prediction_mode
            ValueError: no pytorch support
        """
        model_processor = None
        if isinstance(model, TSPulseForReconstruction):
            model_processor = TSPulseADUtility(model, mode=prediction_mode, aggr_win_size=aggr_win_size, **kwargs)
        elif isinstance(model, TinyTimeMixerForPrediction):
            model_processor = TinyTimeMixerADUtility(model=model, mode=prediction_mode, **kwargs)
        else:
            raise ValueError(f"Error: does not support {self.model.__class__} object!")

        kwargs["context_length"] = model.config.context_length
        kwargs["prediction_length"] = model.config.prediction_length

        if "batch_size" not in kwargs:
            kwargs["batch_size"] = 128

        kwargs["aggr_win_size"] = aggr_win_size
        kwargs["smoothing_window_size"] = smoothing_window_size

        known_mode = model_processor.is_valid_mode(prediction_mode)
        if not known_mode:
            raise ValueError(f"Error: incompatible operation mode {prediction_mode}!")

        # *** TTM
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
        # *** END TTM

        kwargs["prediction_mode"] = prediction_mode
        self._prediction_mode = prediction_mode
        self._model_processor = model_processor
        self._aggr_win_size = aggr_win_size
        self._smoothing_window_size = smoothing_window_size

        super().__init__(model, *args, **kwargs)

        self.__context_memory = {}
        if aggr_function.lower() == "min":
            aggr_function_ = np.min
            select_function_ = np.argmin
        elif aggr_function.lower() == "mean":
            aggr_function_ = np.mean
            select_function_ = None
        else:
            aggr_function_ = np.max
            select_function_ = np.argmax
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
            "frequency_token",  # TTM Specific
        ]
        postprocess_params = [
            "timestamp_column",
            "target_columns",
        ]

        for c in preprocess_params:
            if c in kwargs:
                preprocess_kwargs[c] = kwargs[c]

        for c in postprocess_params:
            if c in kwargs:
                postprocess_kwargs[c] = kwargs[c]

        if self.model_type == "tspulse":
            preprocess_kwargs["prediction_length"] = 1  # should not override model setting when TTM

        mode = kwargs.get("prediction_mode", self._prediction_mode)
        aggr_win_size = kwargs.get("aggr_win_size", self._aggr_win_size)
        expand_score = kwargs.get("expand_score", False)
        report_mode = kwargs.get("report_mode", False)

        smoothing_window_size = kwargs.get("smoothing_window_size", self._smoothing_window_size)

        postprocess_kwargs.update(
            smoothing_window_size=smoothing_window_size, expand_score=expand_score, report_mode=report_mode
        )

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
            "mode": mode,
            "aggr_win_size": aggr_win_size,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "expand_score": expand_score,
        }

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
            input_ = self.feature_extractor.preprocess(input_)

        # Fixing stride to 1
        kwargs["stride"] = 1

        # use forecasting dataset to do the preprocessing
        dataset = ForecastDFDataset(
            input_,
            **kwargs,
        )
        target_columns = kwargs.get("target_columns", [])
        self.__context_memory["data"] = input_
        self.__context_memory["target_columns"] = target_columns
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
        aggr_win_size = forward_params.get("aggr_win_size")
        dataloader = DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=remove_columns_collator, shuffle=False
        )

        it = iter(dataloader)
        accumulator = defaultdict(list)

        with torch.no_grad():
            while (batch := next(it, None)) is not None:
                scores = self.forward(batch, **forward_params)
                for key in scores:
                    accumulator[key].append(scores[key])

        aggr_win_size = forward_params.get("aggr_win_size")
        accumulator_ = OrderedDict()

        extra_kwargs = {}
        if "data" in self.__context_memory:
            data = self.__context_memory["data"]
            target_columns = self.__context_memory.get("target_columns", [])
            if len(target_columns) > 0:
                data = data[target_columns]
            extra_kwargs["reference"] = data.values

        for k in accumulator:
            # score = torch.cat(accumulator[k], axis=0).detach().cpu().numpy()
            score = accumulator[k]
            score = self._model_processor.adjust_boundary(k, score, aggr_win_size=aggr_win_size, **extra_kwargs)
            accumulator_[k] = score

        # call postprocess
        outputs = self.postprocess(ModelOutput(accumulator_), **postprocess_params)

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
            RuntimeError: __description__

        Returns:
            pd.DataFrame: pandas dataframe with anomaly score attached
        """
        result = self.__context_memory["data"].copy()
        expand_score = postprocess_parameters.get("expand_score", False)
        smoothing_window_size = postprocess_parameters.get("smoothing_window_size", 1)
        report_mode = postprocess_parameters.get("report_mode", False)
        if not isinstance(smoothing_window_size, int):
            try:
                smoothing_window_size = int(smoothing_window_size)
            except ValueError:
                smoothing_window_size = 1

        model_outputs_ = OrderedDict()
        for k in model_outputs:
            model_outputs_[k] = score_smoothing(model_outputs[k], smoothing_window_size=smoothing_window_size)

        score = np.stack([score_ for _, score_ in model_outputs_.items()], axis=0)
        mode_selected = None
        if report_mode and (self.select_function is not None):
            keys = [key for key, _ in model_outputs_.items()]
            sel_index = self.select_function(score, axis=0)
            mode_selected = np.asarray([keys[z] for z in sel_index.ravel()]).reshape(sel_index.shape)

        score = self.aggr_function(score, axis=0)
        target_columns = self.__context_memory["target_columns"]
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

        for k in model_outputs:
            result[k] = model_outputs[k]
        self.__context_memory = {}
        return result
