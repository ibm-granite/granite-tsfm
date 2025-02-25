# Copyright contributors to the TSFM project
#
import bisect
from typing import Union

import numpy as np
import torch
from gluonts.dataset.split import InputDataset, LabelDataset, TrainingDataset
from gluonts.itertools import batcher
from gluonts.transform.feature import LastValueImputation
from torch.utils.data import Dataset
from tqdm import tqdm

from tsfm_public.toolkit.dataset import _torch
from tsfm_public.toolkit.time_series_preprocessor import DEFAULT_FREQUENCY_MAPPING


# TTM Constants:
TTM_MAX_FORECAST_HORIZON = 720
TTM_MIN_FORECAST_HORIZON = 16

RESOLUTION_MAP = {
    "oov": "oov",
    "OOV": "oov",
    "min": "min",
    "1min": "min",
    "T": "min",
    "1T": "min",
    "2min": "2min",
    "2T": "2min",
    "5min": "5min",
    "5T": "5min",
    "10min": "10min",
    "10T": "10min",
    "15min": "15min",
    "15T": "15min",
    "30min": "30min",
    "30T": "30min",
    "h": "h",
    "1h": "h",
    "H": "h",
    "1H": "h",
    "d": "d",
    "1d": "d",
    "D": "d",
    "1D": "d",
    "w": "W",
    "1w": "W",
    "W": "W",
    "1W": "W",
    "W-FRI": "W",
    "W-TUE": "W",
    "W-MON": "W",
    "W-WED": "W",
    "W-THU": "W",
    "W-SAT": "W",
    "W-SUN": "W",
    "M": "W",
    "1M": "W",
    "Q-DEC": "W",
    "A-DEC": "A",
}


def get_freq_mapping():
    freq_token_mapping = {}

    freq_token_mapping["oov"] = torch.Tensor([DEFAULT_FREQUENCY_MAPPING["oov"]])
    freq_token_mapping["T"] = torch.Tensor([DEFAULT_FREQUENCY_MAPPING["min"]])
    freq_token_mapping["2T"] = torch.Tensor([DEFAULT_FREQUENCY_MAPPING["2min"]])
    freq_token_mapping["5T"] = torch.Tensor([DEFAULT_FREQUENCY_MAPPING["5min"]])
    freq_token_mapping["10T"] = torch.Tensor([DEFAULT_FREQUENCY_MAPPING["10min"]])
    freq_token_mapping["15T"] = torch.Tensor([DEFAULT_FREQUENCY_MAPPING["15min"]])
    freq_token_mapping["30T"] = torch.Tensor([DEFAULT_FREQUENCY_MAPPING["30min"]])
    freq_token_mapping["H"] = torch.Tensor([DEFAULT_FREQUENCY_MAPPING["h"]])
    freq_token_mapping["D"] = torch.Tensor([DEFAULT_FREQUENCY_MAPPING["d"]])
    freq_token_mapping["W"] = torch.Tensor([DEFAULT_FREQUENCY_MAPPING["W"]])
    freq_token_mapping["W-FRI"] = torch.Tensor([DEFAULT_FREQUENCY_MAPPING["W"]])
    freq_token_mapping["W-TUE"] = torch.Tensor([DEFAULT_FREQUENCY_MAPPING["W"]])
    freq_token_mapping["W-MON"] = torch.Tensor([DEFAULT_FREQUENCY_MAPPING["W"]])
    freq_token_mapping["W-WED"] = torch.Tensor([DEFAULT_FREQUENCY_MAPPING["W"]])
    freq_token_mapping["W-THU"] = torch.Tensor([DEFAULT_FREQUENCY_MAPPING["W"]])
    freq_token_mapping["W-SAT"] = torch.Tensor([DEFAULT_FREQUENCY_MAPPING["W"]])
    freq_token_mapping["W-SUN"] = torch.Tensor([DEFAULT_FREQUENCY_MAPPING["W"]])
    freq_token_mapping["M"] = torch.Tensor([DEFAULT_FREQUENCY_MAPPING["W"]])
    freq_token_mapping["A-DEC"] = torch.Tensor([DEFAULT_FREQUENCY_MAPPING["W"]])
    freq_token_mapping["Q-DEC"] = torch.Tensor([DEFAULT_FREQUENCY_MAPPING["W"]])

    # Seconds are currently mapped to OOV
    freq_token_mapping["10S"] = torch.Tensor([DEFAULT_FREQUENCY_MAPPING["oov"]])

    return freq_token_mapping


def impute_series(target):
    if np.isnan(target).any():
        target = target.copy()
        if len(target.shape) == 2:
            for i in range(target.shape[0]):
                target[i, ...] = LastValueImputation()(target[i, ...])
        elif len(target.shape) == 1:
            target = LastValueImputation()(target)
        else:
            raise Exception("Only 1D and 2D arrays are accepted by the impute_series() function.")
    return target


class StandardScalingGluonTSDataset:
    """
    TTM works best on standard scaled data, especially if fewshot
    finetuning is being performed.
    We can utilize the entire available context to do that.
    This is a global sclaing operation done independently on
    each channel.
    """

    def __init__(self) -> None:
        self.mean = {}
        self.std = {}

    def fit(self, train_data: Union[TrainingDataset, InputDataset]):
        """Calculate the statistics on the historical train data.

        Args:
            train_data (Union[TrainingDataset, InputDataset]): Iterator with
                each series of shape [num_channels, context_length] for multivariate
                and [context_length] for univariate.
        """
        for batch in tqdm(batcher(train_data, batch_size=1)):
            if batch[0]["target"].ndim == 1:
                batch[0]["target"] = batch[0]["target"].reshape(1, -1)  # [1, context_length]
            self.mean[batch[0]["item_id"]] = np.mean(impute_series(batch[0]["target"]), axis=1).reshape(-1, 1)
            std = np.std(impute_series(batch[0]["target"]), axis=1).reshape(-1, 1)
            for i in range(std.shape[0]):
                if std[i] == 0:
                    std[i] = 1
            self.std[batch[0]["item_id"]] = std

    def transform(self, data: Union[TrainingDataset, InputDataset]):
        """Apply scaler using calculated statistics.

        Args:
            data (Union[TrainingDataset, InputDataset]): Iterator with
                each series of shape [num_channels, context_length] for multivariate
                and [context_length] for univariate.

        Returns:
            Iternator: With each series transformed.
        """
        assert len(self.mean) > 0
        assert len(self.std) > 0
        out = list(data)
        for i, _ in tqdm(enumerate(out)):
            item_id = out[i]["item_id"]
            out[i]["target"] = (impute_series(out[i]["target"]) - self.mean[item_id]) / (self.std[item_id])
        return iter(out)

    def inverse_transform(
        self, data: np.ndarray, series_ids: list, prediction_channel_indices: list = []
    ) -> np.ndarray:
        """Inverse transform, and bring data to original scale.

        Args:
            data (np.ndarray): Forecast output of shape [batch, context_length, num_channels]
            series_ids (list): List of item ids.
            prediction_channel_indices (list, optional): List of target channel indices. Defaults to [].

        Raises:
            Exception: If NaN is found in the forecast.

        Returns:
            np.ndarray: Of shape [batch, context_length, num_channels].
        """
        out = np.zeros(data.shape)

        for i in tqdm(range((data.shape[0]))):
            if len(prediction_channel_indices) > 0:
                out[i, ...] = (
                    data[i, ...] * (self.std[series_ids[i]][prediction_channel_indices].T)
                    + self.mean[series_ids[i]][prediction_channel_indices].T
                )
            else:
                out[i, ...] = data[i, ...] * (self.std[series_ids[i]].T) + self.mean[series_ids[i]].T
            if np.isnan(out[i, ...]).any():
                raise Exception("NaN found in forecast!")
        return out


class TorchDatasetFromGluonTSTrainingDataset(Dataset):
    def __init__(
        self,
        gluon_dataset: TrainingDataset,
        context_length: int,
        prediction_length: int,
        last_window_only=False,
        gen_more_samples_for_short_series: bool = True,
        force_short_context: bool = False,
        min_context_mult: int = 4,
        fewshot_fraction: float = 1.0,
        fewshot_location: str = "rand",  # end/start
        use_mask: bool = False,
        send_freq: bool = True,
        freq: str = None,
    ):
        """Wrapper to create pytorch `Dataset` from GluonTS dataset.

        Args:
            gluon_dataset (TrainingDataset): Training dataset.
            context_length (int): Context length.
            prediction_length (int): Prediction length.
            last_window_only (bool, optional): If True, only last window will be take from each series.
                Defaults to False.
            gen_more_samples_for_short_series (bool, optional): If True, it will pad zeros and try to
                generate more samples for short series. Defaults to True.
            force_short_context (bool, optional): If True, it will force to mine short context.
                Defaults to False (recommended).
            min_context_mult (int, optional): Minimum context length multiplier (of prediction length)
                to encourage more sample generation for short series. Defaults to 4.
            fewshot_fraction (float, optional): Fewshot fraction. Defaults to 1.0.
            fewshot_location (str, optional): Fewshot location. Defaults to "rand" (recommended).
            send_freq (bool, optional): Send frequency in getitem. Defaults to True.
            freq (str, optional): Frequency of the dataset. Defaults to None.
        """

        # assert context_length > prediction_length, f'sequence lenght {context_length} has to be strictly greater than forecast length {prediction_length}'
        self.context_length = context_length
        self.prediction_length = min(prediction_length, TTM_MAX_FORECAST_HORIZON)
        self.X = list(gluon_dataset)
        self.last_window_only = last_window_only
        self.stride = 1  # TODO: support other strides
        min_context_needed_mult = min_context_mult  # 4*H needed to forecast
        self.force_short_context = force_short_context
        self.use_mask = use_mask
        self.send_freq = send_freq
        self.freq = freq
        self.series_ids = []

        # force short context
        if self.force_short_context:
            self.actual_seq_len = context_length
            self.context_length = min_context_needed_mult * self.prediction_length
            gen_more_samples_for_short_series = False

        # handle univariate series, and nans
        for i, _ in enumerate(self.X):
            if self.X[i]["target"].ndim == 1:
                self.X[i]["target"] = self.X[i]["target"].reshape(1, -1)

            # Nan imputation
            self.X[i]["target"] = impute_series(self.X[i]["target"])

            # Fewshot: for fewshot_location `start` or `end` truncate each series
            if fewshot_fraction < 1.0 and fewshot_location in ["start", "end"]:
                len_ = self.X[i]["target"].shape[1]
                fewshot_len_ = int(np.floor(len_ * fewshot_fraction))
                if fewshot_len_ >= self.prediction_length * min_context_needed_mult:
                    if fewshot_location == "end":
                        self.X[i]["target"] = self.X[i]["target"][:, -fewshot_len_:]
                    elif fewshot_location == "start":
                        self.X[i]["target"] = self.X[i]["target"][:, :fewshot_len_]

            if self.X[i]["target"].shape[1] < self.context_length + self.prediction_length:
                # This means only 1 sample can be created from this series
                # even after zero-padding. We try to create more when
                # `gen_more_samples_for_short_series=True`
                if (
                    gen_more_samples_for_short_series
                    and self.X[i]["target"].shape[1] >= (min_context_needed_mult + 1) * self.prediction_length
                ):
                    # make sure at least a context of min_context_needed_mult*H is possible
                    # pad more zeros to create more training samples
                    pad = np.zeros(
                        (
                            self.X[i]["target"].shape[0],
                            self.context_length - min_context_needed_mult * self.prediction_length,
                        )
                    )
                else:
                    # pad it to Seq_len + Forecast_len to create 1 window
                    pad = np.zeros(
                        (
                            self.X[i]["target"].shape[0],
                            self.context_length + self.prediction_length - self.X[i]["target"].shape[1],
                        )
                    )

                # prepend
                self.X[i]["target"] = np.concatenate((pad, self.X[i]["target"]), axis=1)
                # print(self.X[i]["target"].shape)

            # series id
            self.series_ids.append(self.X[i]["item_id"])

        # get shape
        if not self.last_window_only:
            self.cumulative_sizes = self.cumsum(self.X)

    def cumsum(self, list_data):
        """
        list_data: list of numpy array of shape [channels x len]
        """
        list_len, sum_ = [], 0
        for i, elm in enumerate(list_data):
            data = elm["target"]
            len_ = data.shape[1] - self.context_length - self.prediction_length + 1
            list_len.append(len_ + sum_)
            sum_ += len_
        return list_len

    def __len__(self):
        if self.last_window_only:
            return len(self.X)  # = num of series
        else:
            return self.cumulative_sizes[-1] // self.stride

    def __getitem__(self, idx):
        if self.last_window_only:
            seq_x = self.X[idx]["target"][:, -(self.context_length + self.prediction_length) : -self.prediction_length]
            seq_y = self.X[idx]["target"][:, -(self.prediction_length) :]
            item_id = self.series_ids[idx]
        else:
            idx = idx * self.stride
            if idx < 0:
                if -idx > len(self):
                    raise ValueError("absolute value of index should not exceed dataset length")
                idx = len(self) + idx
            series_idx = bisect.bisect_right(self.cumulative_sizes, idx)
            if series_idx == 0:
                time_id = idx
            else:
                time_id = idx - self.cumulative_sizes[series_idx - 1]
            seq_x = self.X[series_idx]["target"][:, time_id : time_id + self.context_length]
            seq_y = self.X[series_idx]["target"][
                :, time_id + self.context_length : time_id + self.context_length + self.prediction_length
            ]
            item_id = self.series_ids[series_idx]

        if self.force_short_context and seq_x.shape[1] < self.actual_seq_len:
            pad = np.zeros((seq_x.shape[0], self.actual_seq_len - seq_x.shape[1]))
            seq_x = np.concatenate((pad, seq_x), axis=1)
            past_observed_mask = np.ones(seq_x.shape).astype(bool)
            past_observed_mask[:, : pad.shape[1]] = False
        else:
            # Create a boolean mask where non-zero values are True
            nonzero_mask = seq_x != 0
            # Find the first non-zero column index
            col_indices = np.where(nonzero_mask.any(axis=0))[0]
            if len(col_indices) == 0:
                past_observed_mask = np.zeros(seq_x.shape).astype(bool)
            else:
                past_observed_mask = np.ones(seq_x.shape).astype(bool)
                past_observed_mask[:, : col_indices[0]] = False

        # return torch.from_numpy(seq_x.astype(np.float)).float()
        seq_x, seq_y = _torch(seq_x, seq_y)

        return_output = {
            "past_values": seq_x.T,
            "future_values": seq_y.T,
            "item_id": item_id,
        }
        if self.use_mask:
            return_output["past_observed_mask"]: past_observed_mask.T

        if self.send_freq:
            freq_map = get_freq_mapping()
            return_output["freq_token"] = freq_map[self.freq]

        return return_output


class TorchDatasetFromGluonTSTestDataset(Dataset):
    def __init__(
        self,
        gluon_test_input: InputDataset,
        gluon_test_label: LabelDataset,
        context_length: int,
        prediction_length: int,
        force_short_context: bool = False,
        min_context_mult: int = 4,
    ):
        """Wrapper to create pytorch `Dataset` from GluonTS dataset.

        Args:
            gluon_test_input (InputDataset): GluonTS input dataset.
            gluon_test_label (LabelDataset): GluonTS label dataset.
            context_length (int): Context length.
            prediction_length (int): Forecast horizon.
            force_short_context (bool, optional): If True, it will force to mine short context.
                Defaults to False (recommended).
            min_context_mult (int, optional): Minimum context length multiplier (of prediction length)
                to encourage more sample generation for short series. Defaults to 4.
        """
        # assert context_length > prediction_length, f'sequence lenght {context_length} has to be strictly greater than forecast length {prediction_length}'
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.X = list(gluon_test_input)
        self.Y = list(gluon_test_label)
        self.min_context_needed_mult = min_context_mult
        self.force_short_context = force_short_context
        if self.force_short_context:
            self.actual_seq_len = context_length
            self.context_length = self.min_context_needed_mult * self.prediction_length

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        seq_x = self.X[idx]["target"]
        seq_y = self.Y[idx]["target"]

        if len(seq_x.shape) == 1:
            seq_x = seq_x.reshape(1, -1)
            seq_y = seq_y.reshape(1, -1)

        if seq_x.shape[1] < self.context_length:
            pad = np.zeros((seq_x.shape[0], self.context_length - seq_x.shape[1]))
            seq_x = np.concatenate((pad, seq_x), axis=1)

        seq_x, seq_y = _torch(seq_x[:, -self.context_length :], seq_y[:, : self.prediction_length])

        if self.force_short_context and seq_x.shape[1] < self.actual_seq_len:
            pad = np.zeros((seq_x.shape[0], self.actual_seq_len - seq_x.shape[1]))
            seq_x = np.concatenate((pad, seq_x), axis=1)
            past_observed_mask = np.ones(seq_x.shape).astype(bool)
            past_observed_mask[:, : pad.shape[1]] = False
        else:
            # Create a boolean mask where non-zero values are True
            nonzero_mask = seq_x != 0
            # Find the first non-zero column index
            col_indices = np.where(nonzero_mask.any(axis=0))[0]
            if len(col_indices) == 0:
                past_observed_mask = np.zeros(seq_x.shape).astype(bool)
            else:
                past_observed_mask = np.ones(seq_x.shape).astype(bool)
                past_observed_mask[:, : col_indices[0]] = False

        return_output = {
            "past_values": seq_x.T,
            "future_values": seq_y.T,
        }
        if self.use_mask:
            return_output["past_observed_mask"]: past_observed_mask.T

        return return_output


class ForecastDataset(Dataset):
    def __init__(self, forecast_samples, series_ids, insample_errors, point_forecasts, quantiles):
        self.forecast_samples = forecast_samples
        self.series_ids = series_ids
        self.insample_errors = insample_errors
        self.point_forecasts = point_forecasts
        self.quantiles = quantiles

    def __len__(self):
        return self.forecast_samples.shape[0]

    def __getitem__(self, idx):
        forecast_sample = self.forecast_samples[idx]
        series_id = self.series_ids[idx]
        insample_error = self.insample_errors[series_id]
        point_forecast = self.point_forecasts[idx]
        return forecast_sample, insample_error, point_forecast
