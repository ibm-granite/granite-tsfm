import bisect
from typing import Union

import numpy as np
from gluonts.dataset.split import InputDataset, LabelDataset, TrainingDataset
from gluonts.itertools import batcher
from gluonts.transform.feature import LastValueImputation
from torch.utils.data import Dataset
from tqdm import tqdm

from tsfm_public.toolkit.dataset import _torch


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
        self.mean = []
        self.std = []

    def fit(self, train_data: Union[TrainingDataset, InputDataset]):
        """Calculate the statistics on the historical train data.

        Args:
            train_data (Union[TrainingDataset, InputDataset]): Iterator with
                each series of shape [num_channels, seq_len] for multivariate
                and [seq_len] for univariate.
        """
        for batch in tqdm(batcher(train_data, batch_size=1)):
            if batch[0]["target"].ndim == 1:
                batch[0]["target"] = batch[0]["target"].reshape(1, -1)  # [1, seq_len]
            self.mean.append(np.mean(impute_series(batch[0]["target"]), axis=1).reshape(-1, 1))
            std = np.std(impute_series(batch[0]["target"]), axis=1).reshape(-1, 1)
            for i in range(std.shape[0]):
                if std[i] == 0:
                    std[i] = 1
            self.std.append(std)

    def transform(self, data: Union[TrainingDataset, InputDataset]):
        """Apply scaler using calculated statistics.

        Args:
            data (Union[TrainingDataset, InputDataset]): Iterator with
                each series of shape [num_channels, seq_len] for multivariate
                and [seq_len] for univariate.

        Returns:
            Iternator: With each series transformed.
        """
        assert len(self.mean) > 0
        assert len(self.std) > 0

        out = list(data)
        for i, _ in tqdm(enumerate(out)):
            out[i]["target"] = (impute_series(out[i]["target"]) - self.mean[i]) / (self.std[i])
        return iter(out)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform, and bring data to original scale.

        Args:
            data (np.ndarray): Forecast output of shape [batch, seq_len, num_channels]

        Raises:
            Exception: If NaN is found in the forecast.

        Returns:
            np.ndarray: Of shape [batch, seq_len, num_channels].
        """
        out = np.zeros(data.shape)
        for i in tqdm(range((data.shape[0]))):
            out[i, ...] = data[i, ...] * (self.std[i].T) + self.mean[i].T
            if np.isnan(out[i, ...]).any():
                raise Exception("NaN found in forecast!")
        return out


class TorchDatasetFromGluonTSTrainingDataset(Dataset):
    def __init__(
        self,
        gluon_dataset: TrainingDataset,
        seq_len: int,
        forecast_len: int,
        last_window_only=False,
    ):
        """Wrapper to create pytorch `Dataset` from GluonTS dataset.

        Args:
            gluon_dataset (TrainingDataset): GluonTS dataset.
            seq_len (int): Context length.
            forecast_len (int): Forecast horizon.
            last_window_only (bool, optional): If True, only last window will be processed. Defaults to False.
        """
        # assert seq_len > forecast_len, f'sequence lenght {seq_len} has to be strictly greater than forecast length {forecast_len}'
        self.seq_len = seq_len
        self.forecast_len = forecast_len
        self.X = list(gluon_dataset)
        self.last_window_only = last_window_only
        self.stride = 1  # TODO: support other strides

        # handle univariate series, and nans
        for i, _ in enumerate(self.X):
            if len(self.X[i]["target"].shape) == 1:
                self.X[i]["target"] = self.X[i]["target"].reshape(1, -1)

            # Nan imputation
            self.X[i]["target"] = impute_series(self.X[i]["target"])

            # pad zeros if needed
            if self.X[i]["target"].shape[1] < self.seq_len + self.forecast_len:
                pad = np.zeros(
                    (
                        self.X[i]["target"].shape[0],
                        self.seq_len + self.forecast_len - self.X[i]["target"].shape[1] + 1,
                    )
                )
                self.X[i]["target"] = np.concatenate((pad, self.X[i]["target"]), axis=1)

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
            len_ = data.shape[1] - self.seq_len - self.forecast_len + 1
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
            seq_x = self.X[idx]["target"][:, -(self.seq_len + self.forecast_len) : -self.forecast_len]
            seq_y = self.X[idx]["target"][:, -(self.forecast_len) :]
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
            seq_x = self.X[series_idx]["target"][:, time_id : time_id + self.seq_len]
            seq_y = self.X[series_idx]["target"][
                :, time_id + self.seq_len : time_id + self.seq_len + self.forecast_len
            ]

        # return torch.from_numpy(seq_x.astype(np.float)).float()
        seq_x, seq_y = _torch(seq_x, seq_y)

        return_output = {
            "past_values": seq_x.T,
            "future_values": seq_y.T,
        }

        return return_output


class TorchDatasetFromGluonTSTestDataset(Dataset):
    def __init__(
        self,
        gluon_test_input: InputDataset,
        gluon_test_label: LabelDataset,
        seq_len: int,
        forecast_len: int,
    ):
        """Wrapper to create pytorch `Dataset` from GluonTS dataset.

        Args:
            gluon_dataset (TrainingDataset): GluonTS dataset.
            seq_len (int): Context length.
            forecast_len (int): Forecast horizon.
            last_window_only (bool, optional): If True, only last window will be processed. Defaults to False.
        """
        # assert seq_len > forecast_len, f'sequence lenght {seq_len} has to be strictly greater than forecast length {forecast_len}'
        self.seq_len = seq_len
        self.forecast_len = forecast_len
        self.X = list(gluon_test_input)
        self.Y = list(gluon_test_label)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        seq_x = self.X[idx]["target"]
        seq_y = self.Y[idx]["target"]

        if len(seq_x.shape) == 1:
            seq_x = seq_x.reshape(1, -1)
            seq_y = seq_y.reshape(1, -1)

        if seq_x.shape[1] < self.seq_len:
            pad = np.zeros((seq_x.shape[0], self.seq_len - seq_x.shape[1]))
            seq_x = np.concatenate((pad, seq_x), axis=1)

        seq_x, seq_y = _torch(seq_x[:, -self.seq_len :], seq_y[:, : self.forecast_len])

        return_output = {
            "past_values": seq_x.T,
            "future_values": seq_y.T,
        }

        return return_output
