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
        logging.info(f"{'=' * 10} Dataset Info {'=' * 10}")
        logging.info(f"Dataset name: {self.dataset_name}")
        logging.info(f"Frequency: {self.freq}")
        logging.info(f"Device {self.device}")
        logging.info("=" * 35)

    def preprocess(self, raw):
        target = []
        for entry in raw:
            t = entry["target"]
            if any(np.isnan(t)):
                if all(np.isnan(t)):
                    t = np.zeros_like(t)
                else:
                    t = np.nan_to_num(t, np.nanmean(t))

            target.append(torch.from_numpy(t).float())
        return target

    @torch.no_grad()
    def predict(self, test_data_input, batch_size=2048, *args, **kwargs) -> List[Forecast]:
        while True:
            try:
                forecast_outputs = []
                sample_count = 0
                for raw in tqdm(batcher(test_data_input, batch_size=batch_size)):
                    sample_count += len(raw)
                    if sample_count < len(forecast_outputs):
                        continue
                    target = self.preprocess(raw)

                    # with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16):
                    model_outputs = self.model(
                        inputs=target,
                        prediction_length=self.prediction_length,
                        quantile_levels=self.quantile_levels,
                    )
                    pred_quantiles = [x.cpu().numpy() for x in model_outputs.quantile_predictions]
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
