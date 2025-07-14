# Copyright contributors to the TSFM project
#

import torch
import torch.utils.data
import numpy as np

epsilon = 1e-8


class TSPulseReconstructionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data,
        window_size,
        aggr_window_size=None,
        label=None,
        stride=1,
        normalize=True,
        return_dict=False,
        channel_last=True,
    ):
        # label is only used for plotting
        super().__init__()
        self.window_size = window_size

        self.stride = stride
        self.data = self._normalize_data(data) if normalize else data

        if self.data.shape[0] < window_size:
            pad_len = window_size - self.data.shape[0]
            pad = np.zeros((pad_len, self.data.shape[1]))
            self.data = np.concatenate((self.data, pad), axis=0)

        self.label = label
        self.return_dict = return_dict
        self.channel_last = channel_last

        self.univariate = self.data.shape[1] == 1
        self.sample_num = max((self.data.shape[0] - window_size) // stride + 1, 0)

        if self.label is not None:
            self.samples, self.gen_labels = self._generate_samples()
        else:
            self.samples = self._generate_samples()
        self.input_mask = np.ones(
            (self.window_size, data.shape[1]), dtype=np.float32
        )  # Fixed input mask
        if aggr_window_size is not None:
            self.input_mask[:aggr_window_size, :] = 0

        if not self.channel_last:
            # For MOMENT
            # breakpoint()
            self.samples = self.samples.permute(0, 2, 1)  # batch, channel, window_size
            self.input_mask = self.input_mask[:, 0]  # window_size

    def _normalize_data(self, data, epsilon=1e-8):
        mean, std = np.mean(data, axis=0), np.std(data, axis=0)
        std = np.where(std == 0, epsilon, std)  # Avoid division by zero
        return (data - mean) / std

    def _generate_samples(self):
        data = torch.tensor(self.data, dtype=torch.float32)
        indices = np.arange(0, self.sample_num * self.stride, self.stride)

        if self.univariate:
            X = torch.stack([data[i : i + self.window_size] for i in indices])
        else:
            X = torch.stack([data[i : i + self.window_size, :] for i in indices])

        if self.label is not None:
            self.label = torch.tensor(self.label)
            Y = torch.stack([self.label[i : i + self.window_size] for i in indices])
            return X, Y

        return X

    def __len__(self):
        return self.sample_num

    def __getitem__(self, index):
        if self.label is not None:
            if self.return_dict:
                return {
                    "past_values": self.samples[index],
                    "anomaly_labels": self.gen_labels[index],
                    "past_observed_mask": self.input_mask,
                }
            else:
                return self.samples[index], self.input_mask, self.gen_labels[index]
        else:
            if self.return_dict:
                return {
                    "past_values": self.samples[index],
                    "past_observed_mask": self.input_mask,
                }
            else:
                return self.samples[index], self.input_mask



# The following dataset is used by TSPulse
# It generates data for reconstruction as well as
# forecasting task.
class TSPulseForecastDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data,
        window_size,
        aggr_window_size=None,
        forecast_window_size=None,
        label=None,
        stride=1,
        normalize=True,
        channel_last=True,
    ):
        # label is only used for plotting
        super().__init__()
        self.window_size = window_size
        self.forecast_window_size = forecast_window_size

        self.stride = stride
        self.data = self._normalize_data(data) if normalize else data

        if (
            forecast_window_size is not None
            and self.data.shape[0] < window_size + forecast_window_size
        ):
            pad_len = window_size + forecast_window_size - self.data.shape[0]
            pad = np.zeros((pad_len, self.data.shape[1]))
            self.data = np.concatenate((pad, self.data), axis=0)
        elif self.data.shape[0] < window_size:
            pad_len = window_size - self.data.shape[0]
            pad = np.zeros((pad_len, self.data.shape[1]))
            self.data = np.concatenate((self.data, pad), axis=0)

        self.label = label
        self.channel_last = channel_last

        self.univariate = self.data.shape[1] == 1
        self.sample_num = max((self.data.shape[0] - window_size) // stride + 1, 0)

        if self.label is not None:
            if self.forecast_window_size is not None:
                self.samples, self.gen_labels, self.forecast_labels = self._generate_samples()
            else:
                self.samples, self.gen_labels = self._generate_samples()
        else:
            if self.forecast_window_size is not None:
                self.samples, self.forecast_labels = self._generate_samples()
            else:
                self.samples = self._generate_samples()

        self.input_mask = np.ones(
            (self.window_size, data.shape[1]), dtype=np.float32
        )  # Fixed input mask
        if aggr_window_size is not None:
            self.input_mask[:aggr_window_size, :] = 0

        if not self.channel_last:
            # For MOMENT
            self.samples = self.samples.permute(0, 2, 1)  # batch, channel, window_size
            self.input_mask = self.input_mask[:, 0]  # window_size

    def _normalize_data(self, data, epsilon=1e-8):
        mean, std = np.mean(data, axis=0), np.std(data, axis=0)
        std = np.where(std == 0, epsilon, std)  # Avoid division by zero
        return (data - mean) / std

    def _generate_samples(self):
        data = torch.tensor(self.data, dtype=torch.float32)
        indices = np.arange(0, self.sample_num * self.stride, self.stride)

        if self.univariate:
            if self.forecast_window_size is not None:
                X = []
                F = []
                for i in indices:
                    X.append(data[i : i + self.window_size])
                    f = data[
                        i + self.window_size : i
                        + self.window_size
                        + self.forecast_window_size
                    ]
                    if len(f) < self.forecast_window_size:
                        pad_len = self.forecast_window_size - len(f)
                        f = torch.cat(
                            (
                                f,
                                torch.full(
                                    (pad_len, f.shape[-1]),
                                    float("nan"),
                                ),
                            )
                        )
                    F.append(f)
                X, F = torch.stack(X), torch.stack(F)
            else:
                X = torch.stack([data[i : i + self.window_size] for i in indices])
        else:
            if self.forecast_window_size is not None:
                X = []
                F = []
                for i in indices:
                    X.append(data[i : i + self.window_size, :])
                    f = data[
                        i + self.window_size : i
                        + self.window_size
                        + self.forecast_window_size,
                        :,
                    ]
                    if len(f) < self.forecast_window_size:
                        pad_len = self.forecast_window_size - len(f)
                        f = torch.cat(
                            (
                                f,
                                torch.full(
                                    (pad_len, f.shape[-1]),
                                    float("nan"),
                                ),
                            )
                        )
                    F.append(f)
                X, F = torch.stack(X), torch.stack(F)
            else:
                X = torch.stack([data[i : i + self.window_size, :] for i in indices])

        if self.label is not None:
            self.label = torch.tensor(self.label)
            Y = torch.stack([self.label[i : i + self.window_size] for i in indices])
            if self.forecast_window_size is not None:
                return X, Y, F
            else:
                return X, Y
        if self.forecast_window_size is not None:
            return X, F

        return X

    def __len__(self):
        return self.sample_num

    def __getitem__(self, index):
        ret = {
            "past_values": self.samples[index],
            "past_observed_mask": self.input_mask,
        }
        if self.forecast_window_size is not None:
            ret["future_values"] = self.forecast_labels[index]
        if self.label is not None:
            ret["anomaly_labels"] = self.gen_labels[index]

        return ret
