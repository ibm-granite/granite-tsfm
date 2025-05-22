import inspect
from collections import defaultdict
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler as MinMaxScaler_
from torch import nn as nn
from torch.utils.data import DataLoader

# First Party
from transformers import PreTrainedModel
from transformers.data.data_collator import default_data_collator
from transformers.pipelines.base import GenericTensor, build_pipeline_init_args
from transformers.trainer_utils import RemoveColumnsCollator
from transformers.utils import add_end_docstrings

from tsfm_public.models.tspulse.modeling_tspulse import TSPulseForReconstruction
from tsfm_public.models.tspulse.utils.ad_helpers import boundary_adjusted_scores, compute_tspulse_score

from .dataset import ForecastDFDataset

# Third Party
from .time_series_forecasting_pipeline import TimeSeriesPipeline


@add_end_docstrings(
    build_pipeline_init_args(has_tokenizer=False, has_feature_extractor=True, has_image_processor=False)
)
class TimeSeriesAnomalyDetectionPipeline(TimeSeriesPipeline):
    def __init__(
        self,
        model: Union[PreTrainedModel],
        *args,
        prediction_mode: str = "forecast",
        aggr_function: str = "max",
        **kwargs,
    ):
        kwargs["context_length"] = model.config.context_length
        kwargs["prediction_length"] = model.config.prediction_length

        if "batch_size" not in kwargs:
            kwargs["batch_size"] = 128

        kwargs["aggr_win_size"] = kwargs.get("aggr_win_size", 32)
        kwargs["smoothing_window_size"] = kwargs.get("smoothing_window_size", 1)

        model_type = None
        if isinstance(model, TSPulseForReconstruction):
            model_type = "tspulse"
        # elif isinstance(model, TinyTimeMixerForPrediction):
        #    model_type = 'ttm'
        else:
            raise ValueError(f"Error: does not support {self.model.__class__} object!")

        if (model_type == "tspulse") and (prediction_mode is None):
            prediction_mode = "time"
        else:
            prediction_mode = "time"

        known_mode = False
        for mode_str in ["time", "fft", "forecast"]:
            if mode_str in prediction_mode:
                known_mode = True

        if not known_mode:
            raise ValueError(
                f"Error: unknown operation mode {prediction_mode}, atleast (forecast/time/fft) must be specified! "
            )

        kwargs["prediction_mode"] = prediction_mode

        super().__init__(model, *args, **kwargs)
        self.__context_memory = {}
        if aggr_function.lower() == "min":
            aggr_function = np.min
        elif aggr_function.lower() == "mean":
            aggr_function = np.mean
        else:
            aggr_function = np.max
        self.aggr_function = aggr_function

        if self.framework == "tf":
            raise ValueError(f"The {self.__class__} is only available in PyTorch.")

    @property
    def model_type(self):
        if isinstance(self.model, TSPulseForReconstruction):
            return "tspulse"
        # elif isinstance(self.model, TinyTimeMixerForPrediction):
        #    return 'ttm'
        raise ValueError(f"Error: unsupported model type {self.model.__class__}!")

    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        postprocess_kwargs = {}

        preprocess_params = [
            "context_length",
            "prediction_length",
            "timestamp_column",
            "target_columns",
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
            preprocess_kwargs["prediction_length"] = 1

        mode = kwargs.get("prediction_mode", "time" if self.model_type == "tspulse" else "forecast")
        aggr_win_size = kwargs.get("aggr_win_size", 32)
        postprocess_kwargs["smoothing_window_size"] = kwargs.get("smoothing_window_size", 1)

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
        }

        return preprocess_kwargs, forward_kwargs, postprocess_kwargs

    def preprocess(self, time_series, **kwargs) -> Dict[str, Union[GenericTensor, List[Any]]]:
        """Preprocess step
        Load the data, if not already loaded, and then generate a pytorch dataset.
        """
        # use the feature extractor here
        if isinstance(time_series, str):
            timestamp_column = kwargs.get("timestamp_column")

            time_series = pd.read_csv(
                time_series,
                parse_dates=[timestamp_column],
            )

        # use forecasting dataset to do the preprocessing
        dataset = ForecastDFDataset(
            time_series,
            **kwargs,
        )
        self.__context_memory["data"] = time_series
        return dataset

    def run_single(self, inputs, preprocess_params, forward_params, postprocess_params):
        dataset = self.preprocess(inputs, **preprocess_params)
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

        device = self.model.device

        while (batch := next(it, None)) is not None:
            batch_x = batch["past_values"]
            batch_y = batch["future_values"]
            # Move to device
            batch_x = batch_x.to(device).float()
            batch_y = batch_y.to(device).float()
            model_input = {"past_values": batch_x, "future_values": batch_y}
            scores = self.forward(model_input, **forward_params)
            for key in scores:
                accumulator[key].append(scores[key])

        context_length = self.model.config.context_length
        aggr_win_size = forward_params.get("aggr_win_size")
        for k in accumulator:
            score = np.concatenate(accumulator[k], axis=0)
            if self.model_type == "tspulse":
                score = boundary_adjusted_scores(k, score, context_length, aggr_win_size)
            accumulator[k] = score

        score = self.aggr_function(
            np.vstack(
                [MinMaxScaler_().fit_transform(score_.reshape(-1, 1)).ravel() for _, score_ in accumulator.items()]
            ),
            axis=0,
        )

        model_outputs = defaultdict(list)
        # without shuffling in the dataloader above, we assume that order is preserved
        # otherwise we need to incorporate sequence id somewhere and do a proper join
        model_outputs["prediction_outputs"] = score

        # call postprocess
        outputs = self.postprocess(model_outputs, **postprocess_params)

        return outputs

    def _forward(self, model_inputs, **kwargs):
        """Forward step
        Responsible for taking pre-processed dictionary of tensors and passing it to
        the model. Aligns model parameters with the proper input parameters. Only passes
        the needed parameters from the dictionary to the model, but adds them back to the
        ouput for the next step.

        The keys in model_outputs are governed by the underlying model combined with any
        original input keys.
        """
        if self.model_type == "tspulse":
            model_outputs = compute_tspulse_score(self.model, model_inputs, **kwargs)
        return model_outputs

    def postprocess(self, model_outputs, **postprocess_parameters):
        result = self.__context_memory["data"].copy()
        smoothing_window_size = postprocess_parameters.get("smoothing_window_size")
        if smoothing_window_size > 1:
            for k in model_outputs:
                model_outputs[k] = np.convolve(
                    model_outputs[k], np.ones(smoothing_window_size) / smoothing_window_size, mode="same"
                )

        for k in model_outputs:
            result[k] = model_outputs[k]
        self.__context_memory = {}
        return result
