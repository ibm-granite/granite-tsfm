import logging
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from gluonts.itertools import batcher
from gluonts.model import Forecast
from gluonts.model.forecast import QuantileForecast
from scipy import interpolate
from tqdm.auto import tqdm


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PatchTSTFMEvalPredictor:
    def __init__(
        self,
        model,
        prediction_length,
        dataset_name,
        quantile_levels=None,
        use_fill_nan=False,  # backward compatibility with r1
    ):
        self.model = model
        self.model.eval()
        logging.info(self.model.model_summary())

        self.device = self.model.device
        self.prediction_length = prediction_length
        self.dataset_name = dataset_name
        cur_path = Path(__file__).parent.resolve()
        self.dataset_properties = pd.read_csv(os.path.join(cur_path, "GIFT_EVAL_META.csv"), index_col="dataset")
        self.freq = self.dataset_properties.loc[self.dataset_name, "freq"]
        self.quantile_levels = quantile_levels
        self.use_fill_nan = use_fill_nan

        logging.info(f"{'=' * 10} Dataset Info {'=' * 10}")
        logging.info(f"Dataset name: {self.dataset_name}")
        logging.info(f"Frequency: {self.freq}")
        logging.info(f"Device {self.device}")
        logging.info("=" * 35)

    def preprocess(self, raw):
        target = []
        for entry in raw:
            t = entry["target"]
            if self.use_fill_nan:
                t = self.fill_nan(t)
            if any(np.isnan(t)):
                if all(np.isnan(t)):
                    t = np.zeros_like(t)
                else:
                    t = np.nan_to_num(t, np.nanmean(t))

            target.append(torch.from_numpy(t).float().to(self.device))
        return target

    def fill_nan(self, seq, min_len=65):
        # pad when shorter than min_len
        if len(seq) < min_len:
            seq = np.concatenate([np.ones(min_len - len(seq)) * seq[0], seq])

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

        if len(seq) < min_len:
            seq = np.concatenate([np.ones(min_len - len(seq)) * seq[0], seq])

        # fill nan at the end
        last_ix = np.flip(np.isnan(seq), axis=0).argmin()
        if last_ix != 0:
            seq[-last_ix:] = seq[-(last_ix + 1)]

        # interpolate inf values
        inds = np.arange(seq.shape[0])
        good = np.where(np.isfinite(seq))
        f = interpolate.interp1d(inds[good], seq[good], bounds_error=False)
        nanfree = np.where(np.isfinite(seq), seq, f(inds))
        return nanfree

    @torch.no_grad()
    def predict(self, test_data_input, batch_size=2048, *args, **kwargs) -> List[Forecast]:
        input_ndim = next(iter(test_data_input))["target"].ndim
        while True:
            try:
                forecast_outputs = []
                sample_count = 0
                for raw in tqdm(batcher(test_data_input, batch_size=batch_size)):
                    sample_count += len(raw)
                    if sample_count < len(forecast_outputs):
                        continue
                    target = self.preprocess(raw)

                    with torch.inference_mode():
                        model_outputs = self.model(
                            past_values=target,
                            prediction_length=self.prediction_length,
                            quantile_levels=self.quantile_levels,
                        )
                        torch.cuda.synchronize()
                    pred_quantiles = [
                        (x.squeeze(-1) if input_ndim == 1 else x).cpu().numpy() for x in model_outputs.quantile_outputs
                    ]
                    forecast_outputs.extend(pred_quantiles)
                break
            except torch.cuda.OutOfMemoryError:
                logging.warning(f"OutOfMemoryError at batch_size {batch_size}, reducing to {batch_size // 2}")
                batch_size //= 2

        forecasts = []
        for item, ts in zip(forecast_outputs, test_data_input):
            forecast_start_date = ts["start"] + len(ts["target"])
            forecasts.append(
                QuantileForecast(
                    forecast_arrays=item,
                    forecast_keys=list(map(str, self.quantile_levels)),
                    start_date=forecast_start_date,
                )
            )
        return forecasts
