# Copyright contributors to the TSFM project
#
from typing import Union

import numpy as np
import torch
from gluonts.model import Forecast
from scipy import interpolate

from tsfm_public.models.flowstate.utils.utils import get_fixed_factor


class FlowState_Gift_Wrapper:
    def __init__(
        self, model, prediction_length, batch_size=64, n_ch=1, f="h", device="cpu", domain=None, no_daily=False
    ):
        self.model = model
        cfg = self.model.config

        cfg = cfg.to_dict()

        self.pred_dist = cfg["decoder_patch_len"]

        self.model.eval()
        self.no_daily = no_daily
        self.device = device
        self.prediction_length = prediction_length
        self.f = f
        self.pretrain_context = cfg["context_length"]
        if cfg["prediction_type"] != "quantile":
            raise ValueError("Only quantile loss is supported.")
        self.quantiles = cfg["quantiles"]

        self.replace_nan = False
        self.n_ch = n_ch
        self.batch_size = batch_size
        self.domain = domain
        torch.cuda.empty_cache()

    def fill_nan(self, seq, min_len=10):
        # dealing with nans in sequence
        # no nan
        if not np.isnan(seq).any():
            return seq
        # only nan
        if not (~np.isnan(seq)).any():
            return np.zeros_like(seq)
        # remove nan at beginning
        first_ix = np.isnan(seq).argmin()
        seq = seq[first_ix:]
        if not self.replace_nan:
            return seq
        # fill nan at the end
        last_ix = np.flip(np.isnan(seq), axis=0).argmin()
        if last_ix != 0:
            seq[-last_ix:] = seq[-(last_ix + 1)]
        # ensure min length
        if len(seq) < min_len:
            seq = np.concatenate([np.ones(min_len - len(seq)) * seq[0], seq])
        # interpolate nan values
        inds = np.arange(seq.shape[0])
        good = np.where(np.isfinite(seq))
        f = interpolate.interp1d(inds[good], seq[good], bounds_error=False)
        nanfree = np.where(np.isfinite(seq), seq, f(inds))
        return nanfree

    def get_batched(self, test_data):
        max_context = int(self.pretrain_context / get_fixed_factor(self.f, self.domain))
        datalist = []
        for idx, seq in enumerate(test_data):
            if seq["target"].ndim == 1:
                seq = self.fill_nan(seq["target"])
                inp = torch.from_numpy(seq).view(seq.shape[-1], -1, 1).float()
            else:
                seq = np.stack([self.fill_nan(seqi) for seqi in seq["target"]])  # ch, len
                inp = torch.from_numpy(seq).transpose(0, 1).float().view(seq.shape[-1], -1, 1)
            context_len = min(max_context, inp.shape[0])
            inp = inp[-context_len:]
            inp = torch.from_numpy(self.fill_nan(inp.numpy()))  # could be nan again in the beginning (special case)
            datalist.append((inp.shape[0], inp, idx))
        # sort by context
        datalist = sorted(datalist, key=lambda el: el[0])
        # cluster same length sequences in batches of max size self.batch_size
        batched = []
        indeces = [dl[2] for dl in datalist]
        prev_cl = -1
        for cl, seq, _ in datalist:
            if torch.isnan(seq).sum() > 0:
                seq = torch.tensor(self.fill_nan(seq.numpy()))
            if cl != prev_cl or batched[-1].shape[1] + self.n_ch > self.batch_size:
                batched.append(seq)
            else:
                batched[-1] = torch.cat((batched[-1], seq), dim=1)
            prev_cl = cl

        return batched, indeces

    def forecasts_from_batch(self, pred):
        if pred.shape[0] % self.n_ch != 0:
            raise Exception("This should not happen. Predictions for different channels should be in the same batch.")
        pred = [Gift_Forecast(indi_pred, self.quantiles) for indi_pred in torch.split(pred, self.n_ch, dim=0)]
        return pred

    def set_freq(self):
        self.scale_factor = get_fixed_factor(self.f, self.domain)
        if self.no_daily:
            self.scale_factor /= 7  # seasonality correction for datasets without daily, but only weekly season

    def predict(self, test_data):
        self.model.eval()
        preds = []
        batched_data, indeces = self.get_batched(test_data)
        self.set_freq()

        for idx, batch in enumerate(batched_data):
            pred = self.model(
                past_values=batch.to(self.device),
                scale_factor=self.scale_factor,
                prediction_length=self.prediction_length,
                batch_first=False,
            ).prediction_outputs
            pred = pred.squeeze(-1).transpose(-1, -2)  # pred has shape: batch, forecast_len, quantiles
            preds.extend(self.forecasts_from_batch(pred[:, : self.prediction_length].cpu()))
        # reorder back to original order
        preds_ = len(preds) * [None]
        for i, i_ in enumerate(indeces):
            preds_[i_] = preds[i]
        return preds_


class Gift_Forecast(Forecast):
    def __init__(self, seq, quantiles):
        super().__init__()
        self.quantiles = quantiles
        self.seq = seq.transpose(0, 1).detach().numpy()  # seq_len, n_channels, n_quantiles

    @property
    def mean(self):
        return self.seq[..., self.quantiles.index(0.5)].squeeze()

    def quantile(self, q: Union[float, str]) -> np.ndarray:
        """
        Compute a quantile from the predicted distribution.

        Parameters
        ----------
        q
            Quantile to compute.

        Returns
        -------
        numpy.ndarray
            Value of the quantile across the prediction range.
        """
        q = float(q)
        if q in self.quantiles:
            return self.seq[..., self.quantiles.index(q)].squeeze()
        elif len(self.quantiles) == 1:
            return self.mean
        elif q < self.quantiles[0]:
            return self.seq[..., 0].squeeze()
        elif q > self.quantiles[0]:
            return self.seq[..., -1].squeeze()
        raise NotImplementedError("Return closest quantile or weighted average of larger and smaller quantile.")
