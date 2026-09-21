"""Implementations of Forecaster class using pre-trained models directly as for gift-eval submission"""

import logging
import os
from typing import Optional

from tsfm_public.toolkit.forecasters import Forecaster, ForecastResult

import numpy as np
import torch
from gluonts.transform.feature import LastValueImputation
from scipy import interpolate

from tsfm_public import PatchTSTFMConfig, PatchTSTFMForPrediction
from tsfm_public import TinyTimeMixerForPrediction
from tsfm_public import FlowStateForPrediction
from tsfm_public.toolkit.get_model import (
    TTM_LOW_RESOLUTION_MODELS_MAX_CONTEXT,
    get_model,
)
from tsfm_public.toolkit.time_series_preprocessor import DEFAULT_FREQUENCY_MAPPING
# TTM_LOW_RESOLUTION_MODELS_MAX_CONTEXT and get_model are used by

class PatchTSTFMGiftModelForecaster(Forecaster):

    _PATCHTST_MODELS = {
        "patchtst-fm-r1":           {"model_checkpoint": "ibm-research/patchtst-fm-r1",                          "max_context_length": 8192},
        "granite-patchtst-fm-r1":   {"model_checkpoint": "ibm-granite/granite-timeseries-patchtst-fm-r1",        "max_context_length": 8192},
        "granite-patchtst-fm-r2":   {"model_checkpoint": "ibm-granite/granite-timeseries-patchtst-fm-r2",        "max_context_length": 8192}
    }

    def __init__(self, model_version: str = "patchtst-fm-r1", device: str = "cuda", use_fill_nan = False):

        self.device = device
        self.uses_variable_length_input = model_version == "granite-patchtst-fm-r2"

        model_checkpoint = self._PATCHTST_MODELS[model_version]['model_checkpoint']
        token = os.environ.get("HF_TOKEN")
        config = PatchTSTFMConfig.from_pretrained(model_checkpoint, token=token)
        self.model = PatchTSTFMForPrediction.from_pretrained(
            model_checkpoint, config=config, token=token,
        ).to(device)

        # self.model = PatchTSTFMForPrediction.from_pretrained(model_checkpoint, device_map=device)
        self.model.eval()
        self.max_context_length = self.model.config.context_length

        self.quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.ix_median = 4
        self.use_fill_nan = use_fill_nan


    def _fill_nan(self, seq, min_len=65):
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

    def __call__(self, 
                 data: list[list[int | float]], 
                 prediction_length: list[int], 
                 enforce_only_positive = True
    ) ->dict:
        """
        Args:
            data: list[list[int|float]], a list of time-series sequences (each sequence is a list of numbers)
            prediction_length: list[int], a list of prediction lengths for each time-series sequence
        Returns:
            output: list[dict], a list of dictionaries, each containing the low, median, and high values of the forecast
                - median: list[float], the median values of the forecast
                - quantile_levels: list[str], the quantile levels
                - quantile_{i}: list[float], the quantile values for each quantile level
        """
        assert (
            len(data) == len(prediction_length) and len(data) > 0
        ), "data and prediction_length must have the same length and not empty"
        
        self.model.eval()
        output = []
        
        # Process each series independently
        for i, (series, pred_len) in enumerate(zip(data, prediction_length)):

            all_non_negative = np.nanmin(series) >= 0

            # NaN treatment Following: https://github.com/ibm-granite/granite-tsfm/blob/412573aa1e618311be4e69b0a8c61f4eb847f538/notebooks/hfdemo/patchtst_fm/patchtst_fm_predictor.py#L44
            series = np.array(series)
            if self.use_fill_nan:
                # print('FILLING NANS!!')
                series = self._fill_nan(series, min_len=65)

            # Limit to max context length
            max_ctx_series = min(len(series), self.max_context_length)
            series = series[-max_ctx_series:]
            if any(np.isnan(series)):
                if all(np.isnan(series)):
                    series = np.zeros_like(series)
                else:
                    series = np.nan_to_num(series, np.nanmean(series))
            # print(series.shape)
            series = torch.from_numpy(series).float().to(self.device)
            if self.uses_variable_length_input:
                past_values = [series]
            else:
                past_values = series.unsqueeze(0) if series.ndim == 1 else series
            

            # Call forecast_with_patchtst for this series
            try:
                with torch.no_grad():
                    model_outputs = self.model(
                            past_values=past_values,
                            prediction_length=pred_len,
                            quantile_levels=self.quantile_levels,
                        )
                
            except Exception as e:
                raise RuntimeError(f"PatchTST-fm forecasting failed for series {i}: {str(e)}")
            
            
            if self.uses_variable_length_input:
                pred_quantiles = model_outputs.quantile_outputs[0].detach().cpu().numpy()
            else:
                pred_quantiles = model_outputs.quantile_outputs.detach().cpu().numpy()[0]
            # print(pred_quantiles.shape)

            # Build output dictionary
            # predicted shape: (n_samples, prediction_length, n_targets)
            # Extract median forecast for the last sample, first (and only) target
            forecast_i = {
                "median": [pred_quantiles[self.ix_median,t].squeeze() for t in range(pred_len)],
                "quantile_levels": [str(q) for q in self.quantile_levels],
            }
            if enforce_only_positive & all_non_negative:
                forecast_i['median'] = [max(x, 0) for x in forecast_i['median']]
            
            # Extract quantiles
            for q_idx in range(len(self.quantile_levels)):
                # predicted_quantiles shape: (n_samples, prediction_length, n_targets, n_quantiles)
                forecast_i[f"quantile_{q_idx}"] = [
                    pred_quantiles[q_idx,t].squeeze()
                    for t in range(pred_len)
                ]

                if enforce_only_positive & all_non_negative:
                    forecast_i[f"quantile_{q_idx}"] = [max(x, 0) for x in forecast_i[f"quantile_{q_idx}"]]
            output.append(forecast_i)
        
        return output


    def forecast_for_ensemble(self,
                 data: list[list[int | float]],
                 prediction_length: list[int],
                 enforce_only_positive = True
    ) -> np.ndarray:
        """Returns forecast array of shape (n_samples, prediction_length, n_targets, n_quantiles).

        n_targets is 1 since each series is univariate.
        n_samples matches the number of series in data.
        """
        forecast = self.__call__(data=data,
                                 prediction_length=prediction_length,
                                 enforce_only_positive=enforce_only_positive)

        n_quantiles = len(self.quantile_levels)
        series_arrays = []
        for forecast_i, pred_len in zip(forecast, prediction_length):
            # Stack quantiles: (pred_len, n_quantiles)
            quantile_matrix = np.stack(
                [forecast_i[f"quantile_{q_idx}"] for q_idx in range(n_quantiles)],
                axis=-1,
            )
            # Add n_targets=1 and n_samples=1 dims -> (1, pred_len, 1, n_quantiles)
            series_arrays.append(quantile_matrix[np.newaxis, :, np.newaxis, :])

        # Concatenate across series -> (n_samples, pred_len, n_targets=1, n_quantiles)
        return np.concatenate(series_arrays, axis=0)
         





def _impute_ttm_series(target):
    target = np.asarray(target, dtype=float)
    if np.isnan(target).any():
        target = LastValueImputation()(target.copy())
    return target


class _TTMContextScaler:
    """Match the per-item rolling normalization used by the TTM leaderboard."""

    def __init__(self, data):
        self.mean = {}
        self.std = {}
        for entry in data:
            target = np.asarray(entry["target"], dtype=float)
            if target.ndim == 1:
                target = target.reshape(1, -1)
            target = _impute_ttm_series(target)
            item_id = entry["item_id"]
            self.mean[item_id] = target.mean(axis=1).reshape(-1, 1)
            std = target.std(axis=1).reshape(-1, 1)
            std[std == 0] = 1
            self.std[item_id] = std

    def transform(self, target, item_id):
        target = _impute_ttm_series(target)
        target_for_stats = target.reshape(1, -1) if target.ndim == 1 else target
        self.mean[item_id] = target_for_stats.mean(axis=1).reshape(-1, 1)
        std = target_for_stats.std(axis=1).reshape(-1, 1)
        std[std == 0] = 1
        self.std[item_id] = std
        return (target - self.mean[item_id].squeeze()) / self.std[item_id].squeeze()

    def inverse_transform(self, forecast, item_id):
        return forecast * self.std[item_id].T + self.mean[item_id].T


class TinyTimeMixerPreTrainedGiftModelForecaster(Forecaster):

    # TTM model selection constants
    _TTM_MAX_FORECAST_HORIZON = 720
    _TTM_MIN_FORECAST_HORIZON = 16
    _RESOLUTION_MAP = {
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
        "M": "oov",
        "1M": "oov",
        "Q-DEC": "oov",
        "A-DEC": "oov",
        "A": "oov",
    }
    _TTM_MODELS = {
        "ttm-r3-pt": {"model_checkpoint": "ibm-research/ttm-r3"},
        "granite-ttm-r3": {"model_checkpoint": "ibm-granite/granite-timeseries-ttm-r3"},
    }

    def _get_gift_ttm_model(
        self,
        model_path: str,
        context_length: int,
        prediction_length: int,
        freq: str,
        term: str,
        return_model_key: bool = False,
        use_lite: bool = False,
        **kwargs,
    ):
        """Select the most suitable TTM model variant based on context length,
        prediction length, frequency and term.

        Args:
            model_path: HuggingFace model card or local path.
            context_length: Context length of the dataset.
            prediction_length: Forecast horizon.
            freq: Frequency string of the dataset (e.g. "h", "D", "W").
            term: Forecast term, one of "short", "medium", "long".
            return_model_key: If True, return the selected model key alongside the model.
            use_lite: If True, prefer lite model variants.
            **kwargs: Additional arguments forwarded to get_model.

        Returns:
            Loaded TTM model (or (model, key) tuple if return_model_key=True).
        """
        prefer_l1_loss = False
        prefer_longer_context = True
        freq_prefix_tuning = False
        force_return = "zeropad"

        if term == "short" and (
            str(freq).startswith("W")
            or str(freq).startswith("M")
            or str(freq).startswith("Q")
            or str(freq).startswith("A")
        ):
            prefer_l1_loss = True
            prefer_longer_context = False
            freq_prefix_tuning = True

        if term == "short" and str(freq).startswith("D"):
            prefer_l1_loss = True
            freq_prefix_tuning = True
            if context_length < 2 * TTM_LOW_RESOLUTION_MODELS_MAX_CONTEXT:
                prefer_longer_context = False
            else:
                prefer_longer_context = True

        if term == "short" and str(freq).startswith("A"):
            force_return = "random_init_small"

        if prediction_length > self._TTM_MAX_FORECAST_HORIZON:
            force_return = "rolling"

        return get_model(
            model_path=model_path,
            context_length=context_length,
            prediction_length=prediction_length,
            freq_prefix_tuning=freq_prefix_tuning,
            freq=self._RESOLUTION_MAP.get(freq, "oov"),
            prefer_l1_loss=prefer_l1_loss,
            prefer_longer_context=prefer_longer_context,
            force_return=force_return,
            return_model_key=return_model_key,
            use_lite=use_lite,
            **kwargs,
        )

    uses_series_ids = True

    def __init__(self, model_version: str = "ttm-r3-pt", device="cuda", use_get_gift_model=False, context_length = None, prediction_length = None, freq='oov', term='None', scaling_data=None):

        self.quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.ix_median = 4
        model_checkpoint = self._TTM_MODELS[model_version]['model_checkpoint']

        if use_get_gift_model:
            self.model = self._get_gift_ttm_model(
                model_path=model_checkpoint,
                context_length=context_length,
                prediction_length=prediction_length,
                freq=freq,
                term=term,
                return_model_key=False,
                use_lite=False,
            )
        else:
            self.model = TinyTimeMixerForPrediction.from_pretrained(model_checkpoint)

        self.model = self.model.to(device)
        self.model.eval()
        self.max_context_length = self.model.config.context_length
        self.device = device
        self.freq = freq
        self.term = term
        self.scaler = _TTMContextScaler(scaling_data) if scaling_data is not None else None

    def _frequency_token(self):
        if self.freq in {"M", "1M", "Q-DEC", "A-DEC", "A"}:
            return 11
        if self.freq == "10S":
            return 10
        normalized_freq = self._RESOLUTION_MAP.get(self.freq, "oov")
        return DEFAULT_FREQUENCY_MAPPING.get(normalized_freq, DEFAULT_FREQUENCY_MAPPING["oov"])

    def __call__(self, 
                 data: list[list[int | float]], 
                 prediction_length: list[int], 
                 enforce_only_positive = False,
                 series_ids = None,
    ):
        """
        Args:
            data: list[list[int|float]], a list of time-series sequences (each sequence is a list of numbers)
            prediction_length: list[int], a list of prediction lengths for each time-series sequence
        Returns:
            output: list[dict], a list of dictionaries, each containing the low, median, and high values of the forecast
                - median: list[float], the median values of the forecast
                - quantile_levels: list[str], the quantile levels
                - quantile_{i}: list[float], the quantile values for each quantile level
        """
        assert (
            len(data) == len(prediction_length) and len(data) > 0
        ), "data and prediction_length must have the same length and not empty"
        if self.scaler is not None and (series_ids is None or len(series_ids) != len(data)):
            raise ValueError("series_ids must identify every series when TTM scaling is enabled")
        
        self.model.eval()
        output = []
        
        # Process each series independently
        for i, (series, pred_len) in enumerate(zip(data, prediction_length)):

            all_non_negative = np.nanmin(series) >= 0

            series = np.asarray(series, dtype=float)
            if self.scaler is not None:
                series = self.scaler.transform(series, series_ids[i])
            else:
                series = _impute_ttm_series(series)

            series = series[-self.max_context_length:]
            padding_length = self.max_context_length - len(series)
            past_values = np.pad(series, (padding_length, 0))
            observed_mask = np.pad(
                np.ones(len(series), dtype=bool),
                (padding_length, 0),
                constant_values=False,
            )
            model_inputs = {
                "past_values": torch.from_numpy(past_values).float().reshape(1, -1, 1).to(self.device),
                "past_observed_mask": torch.from_numpy(observed_mask).reshape(1, -1, 1).to(self.device),
            }
            if getattr(self.model.config, "resolution_prefix_tuning", False):
                model_inputs["freq_token"] = torch.tensor(
                    [self._frequency_token()], device=self.device
                )
            
            #
            try:
                with torch.no_grad():
                    model_outputs = self.model(**model_inputs)
                
            except Exception as e:
                raise RuntimeError(f"TTM forecasting failed for series {i}: {str(e)}")
            
            
            pred_quantiles = model_outputs.quantile_outputs.detach().cpu().numpy()
            if self.scaler is not None:
                pred_quantiles = self.scaler.inverse_transform(pred_quantiles, series_ids[i])
            # print(pred_quantiles.shape)

            # Build output dictionary
            # predicted shape: (n_samples, prediction_length, n_targets)
            # Extract median forecast for the last sample, first (and only) target
            forecast_i = {
                "median": [pred_quantiles[:,self.ix_median,t].squeeze() for t in range(pred_len)],
                "quantile_levels": [str(q) for q in self.quantile_levels],
            }
            if enforce_only_positive & all_non_negative:
                forecast_i['median'] = [max(x, 0) for x in forecast_i['median']]
            
            # Extract quantiles
            for q_idx in range(len(self.quantile_levels)):
                # predicted_quantiles shape: (n_samples, prediction_length, n_targets, n_quantiles)
                forecast_i[f"quantile_{q_idx}"] = [
                    pred_quantiles[:,q_idx,t].squeeze()
                    for t in range(pred_len)
                ]

                if enforce_only_positive & all_non_negative:
                    forecast_i[f"quantile_{q_idx}"] = [max(x, 0) for x in forecast_i[f"quantile_{q_idx}"]]
            
            
            output.append(forecast_i)
        
        return output


    def forecast_for_ensemble(self,
                              data: list[list[int | float]],
                              prediction_length: list[int],
                              enforce_only_positive = False,
                              series_ids = None,
    ) -> np.ndarray:
        """Returns forecast array of shape (n_samples, prediction_length, n_targets, n_quantiles).

        n_targets is 1 since each series is univariate.
        n_samples matches the number of series in data.
        """
        forecast = self.__call__(data=data,
                                 prediction_length=prediction_length,
                                 enforce_only_positive=enforce_only_positive,
                                 series_ids=series_ids)

        n_quantiles = len(self.quantile_levels)
        series_arrays = []
        for forecast_i, pred_len in zip(forecast, prediction_length):
            # Stack quantiles: (pred_len, n_quantiles)
            quantile_matrix = np.stack(
                [forecast_i[f"quantile_{q_idx}"] for q_idx in range(n_quantiles)],
                axis=-1,
            )
            # Add n_targets=1 and n_samples=1 dims -> (1, pred_len, 1, n_quantiles)
            series_arrays.append(quantile_matrix[np.newaxis, :, np.newaxis, :])

        # Concatenate across series -> (n_samples, pred_len, n_targets=1, n_quantiles)
        return np.concatenate(series_arrays, axis=0)


class FlowstateGiftModelForecaster(Forecaster):
    """Forecaster wrapping FlowState for GIFT-eval style batch inference."""

    _BASE_SEASON = 24.0
    _FLOWSTATE_MODELS = {
        "granite-flowstate-r1":   {"model_checkpoint": "ibm-granite/granite-timeseries-flowstate-r1", "max_context_length": 2048},
        "flowstate-r1":           {"model_checkpoint": "ibm-research/flowstate", "max_context_length": 2048},
        "flowstate-r1.1":         {"model_checkpoint": "ibm-research/flowstate", "max_context_length": 4096, "revision": "r1.1"},
        "granite-flowstate-r1.1": {"model_checkpoint": "ibm-granite/granite-timeseries-flowstate-r1", "max_context_length": 4096, "revision": "r1.1"},
    }

    def __init__(
        self,
        model_version: str = "flowstate-r1.1",
        device: str = "cuda",
        freq: str = None,
        domain: str = None,
        no_daily: bool = False,
        replace_nan: bool = False,
    ):
        """Load FlowState model and store inference defaults.

        Args:
            model_version: Key in _FLOWSTATE_MODELS. Default: "flowstate-r1.1".
            device: Inference device. Default: "cuda".
            freq: Default frequency string used for scale factor computation.
                  If None, scale_factor defaults to 1.0 in __call__.
            domain: Default domain string (affects weekly seasonality in get_fixed_factor).
            no_daily: Default no_daily flag for get_fixed_factor.
            replace_nan: If True, interpolate interior NaNs in _fill_nan. Default: False.
        """
        cfg = self._FLOWSTATE_MODELS[model_version]
        revision = cfg.get("revision", "main")

        self.model = FlowStateForPrediction.from_pretrained(
            cfg["model_checkpoint"],
            batch_first=True,
            revision=revision,
        ).to(device)
        self.model.eval()

        self.device = device
        self.max_context_length = self.model.config.context_length
        self.quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.ix_median = 4
        self.freq = freq
        self.domain = domain
        self.no_daily = no_daily
        self._replace_nan = replace_nan

    def _fill_nan(self, seq: np.ndarray, min_len: int = 10) -> np.ndarray:
        """Handle NaN values in a 1D time series array.

        Removes leading NaNs. If self._replace_nan is True, also fills trailing
        NaNs and interpolates interior NaNs using scipy.

        Args:
            seq: 1D numpy array of float values.
            min_len: Minimum length after NaN removal (padded with first value).

        Returns:
            NaN-treated numpy array.
        """
        # No NaN
        if not np.isnan(seq).any():
            return seq
        # All NaN
        if not (~np.isnan(seq)).any():
            return np.zeros_like(seq)
        # Remove leading NaNs
        first_ix = np.isnan(seq).argmin()
        seq = seq[first_ix:]
        if not self._replace_nan:
            return seq
        # Fill trailing NaNs
        last_ix = np.flip(np.isnan(seq), axis=0).argmin()
        if last_ix != 0:
            seq[-last_ix:] = seq[-(last_ix + 1)]
        # Ensure minimum length
        if len(seq) < min_len:
            seq = np.concatenate([np.ones(min_len - len(seq)) * seq[0], seq])
        # Interpolate interior NaNs
        from scipy import interpolate
        inds = np.arange(seq.shape[0])
        good = np.where(np.isfinite(seq))
        f = interpolate.interp1d(inds[good], seq[good], bounds_error=False)
        return np.where(np.isfinite(seq), seq, f(inds))

    def _get_fixed_factor(self, freq: str, domain: str = None, no_daily: bool = False) -> float:
        """Compute FlowState scale factor from frequency and domain.

        Args:
            freq: Frequency string (e.g. "H", "D", "W", "M").
            domain: Optional domain string. Affects weekly seasonality for
                    Transport, Healthcare, Sales domains.
            no_daily: If True, divide factor by 7 (weekly-only seasonality).

        Returns:
            float scale factor.
        """
        B = self._BASE_SEASON
        has_weekly = domain in ("Transport", "Healthcare", "Sales")

        if freq == "4S":
            factor = B / (3600.0 / 4)
        elif freq == "10S":
            factor = B / 360
        elif freq == "T":
            factor = B / (24.0 * 60)
        elif freq[-1] == "T":
            n_min = int(freq[:-1])
            factor = B / (24 * 60 / n_min)
        elif freq == "H":
            factor = B / 24
        elif freq == "6H":
            factor = B / 4
        elif freq == "D":
            factor = B / 7 if has_weekly else B / 365
        elif freq[-1] == "D" and "WED" not in freq:
            n = int(freq[:-1])
            factor = (B / 7 if has_weekly else B / 365) * n
        elif freq == "W" or "W-" in freq:
            factor = B / (365.0 / 7)
        elif freq == "M" or "M-" in freq:
            factor = B / 12
        elif "Q" in freq:
            factor = B / 4.0
        elif "A" in freq:
            factor = B / 4.0
        else:
            print(f"{freq} not implemented, setting default factor = 1")
            factor = 1.0

        return factor / 7 if no_daily else factor

    def __call__(
        self,
        data: list[list[int | float]],
        prediction_length: list[int],
        freq_str: str = None,
        domain: str = None,
        no_daily: bool = None,
        enforce_only_positive: bool = True,
    ) -> list[dict]:
        """Run FlowState inference on a batch of univariate series.

        Args:
            data: List of time series (each a list of numbers).
            prediction_length: List of forecast horizons, one per series.
            freq_str: Frequency string for scale factor. Falls back to self.freq if None.
            domain: Domain string. Falls back to self.domain if None.
            no_daily: no_daily flag. Falls back to self.no_daily if None.
            enforce_only_positive: Clip negative forecasts to 0 for non-negative series.

        Returns:
            List of dicts with keys: median, quantile_levels, quantile_0 … quantile_8.
        """
        assert (
            len(data) == len(prediction_length) and len(data) > 0
        ), "data and prediction_length must have the same length and not empty"

        # Resolve call-time overrides vs instance defaults
        freq_str  = freq_str  if freq_str  is not None else self.freq
        domain    = domain    if domain    is not None else self.domain
        no_daily  = no_daily  if no_daily  is not None else self.no_daily

        if freq_str is not None:
            scale_factor = self._get_fixed_factor(freq_str, domain=domain, no_daily=no_daily)
            max_context_length = int(self.max_context_length / self._get_fixed_factor(freq_str, domain=domain))
        else:
            scale_factor = 1.0
            max_context_length = self.max_context_length

        self.model.eval()
        output = []

        for i, (series, pred_len) in enumerate(zip(data, prediction_length)):
            all_non_negative = np.nanmin(series) >= 0

            series = self._fill_nan(np.array(series, dtype=float))
            max_ctx_series = min(len(series), max_context_length)
            series = series[-max_ctx_series:]
            series = self._fill_nan(series)

            if len(series.shape) == 1:
                series = np.expand_dims(series, axis=0)
            if len(series.shape) == 2:
                series = np.expand_dims(series, axis=2)
            series = torch.from_numpy(series).float().to(self.device)

            try:
                with torch.no_grad():
                    model_outputs = self.model(
                        past_values=series,
                        scale_factor=scale_factor,
                        prediction_length=pred_len,
                        batch_first=True,
                    )
            except Exception as e:
                raise RuntimeError(f"FlowState forecasting failed for series {i}: {str(e)}")

            pred_quantiles = model_outputs.quantile_outputs.detach().cpu().numpy()

            forecast_i = {
                "median": [pred_quantiles[:, self.ix_median, t].squeeze() for t in range(pred_len)],
                "quantile_levels": [str(q) for q in self.quantile_levels],
            }
            if enforce_only_positive and all_non_negative:
                forecast_i["median"] = [max(x, 0) for x in forecast_i["median"]]

            for q_idx in range(len(self.quantile_levels)):
                forecast_i[f"quantile_{q_idx}"] = [
                    pred_quantiles[:, q_idx, t].squeeze() for t in range(pred_len)
                ]
                if enforce_only_positive and all_non_negative:
                    forecast_i[f"quantile_{q_idx}"] = [max(x, 0) for x in forecast_i[f"quantile_{q_idx}"]]

            output.append(forecast_i)

        return output

    def forecast_for_ensemble(
        self,
        data: list[list[int | float]],
        prediction_length: list[int],
        freq_str: str = None,
        domain: str = None,
        no_daily: bool = None,
        enforce_only_positive: bool = True,
    ) -> np.ndarray:
        """Returns forecast array of shape (n_samples, prediction_length, n_targets, n_quantiles).

        n_targets is 1 since each series is univariate.
        n_samples matches the number of series in data.
        """
        forecast = self.__call__(
            data=data,
            prediction_length=prediction_length,
            freq_str=freq_str,
            domain=domain,
            no_daily=no_daily,
            enforce_only_positive=enforce_only_positive,
        )

        n_quantiles = len(self.quantile_levels)
        series_arrays = []
        for forecast_i, pred_len in zip(forecast, prediction_length):
            # Stack quantiles: (pred_len, n_quantiles)
            quantile_matrix = np.stack(
                [forecast_i[f"quantile_{q_idx}"] for q_idx in range(n_quantiles)],
                axis=-1,
            )
            # Add n_targets=1 and n_samples=1 dims -> (1, pred_len, 1, n_quantiles)
            series_arrays.append(quantile_matrix[np.newaxis, :, np.newaxis, :])

        # Concatenate across series -> (n_samples, pred_len, n_targets=1, n_quantiles)
        return np.concatenate(series_arrays, axis=0)


# ---------------------------------------------------------------------------
# Ensemble factory
# ---------------------------------------------------------------------------

from tsfm_public.models.ensemble.modeling_ensemble import QuantileEnsembleForecaster
from tsfm_public.toolkit.ensemble_aggregation import (
    aggregate_linear_pool,
    aggregate_vincent,
)

_ENSEMBLE_FUNCTIONS = {
    "probability_space_aggregation": aggregate_linear_pool,
    "quantile_space_aggregation":    aggregate_vincent,
}

_DEFAULT_QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _single_member_result(predictions, quantile_levels, weights=None):
    """Return a single member unchanged instead of numerically re-aggregating it."""
    quantiles = np.asarray(predictions)[..., 0]
    median = quantiles[..., quantile_levels.index(0.5)]
    return ForecastResult(
        success=True,
        message="Single model forecast completed successfully.",
        predicted=median.tolist(),
        predicted_quantiles=quantiles.tolist(),
        metadata={
            "method": "single_member",
            "quantile_levels": quantile_levels,
            "n_models": 1,
            "weights": None,
        },
    )


class RecordingForecaster(Forecaster):
    """Record a member forecast for benchmark reporting without changing ensemble APIs."""

    def __init__(self, model_name, forecaster):
        self.model_name = model_name
        self.forecaster = forecaster
        self.last_forecast = None

    def __call__(self, *args, **kwargs):
        return self.forecaster(*args, **kwargs)

    def forecast_for_ensemble(self, *args, **kwargs):
        self.last_forecast = None
        kwargs.pop("quantile_levels", None)
        if not getattr(self.forecaster, "uses_series_ids", False):
            kwargs.pop("series_ids", None)
        self.last_forecast = np.asarray(
            self.forecaster.forecast_for_ensemble(*args, **kwargs)
        )
        return self.last_forecast


def build_gift_ensemble(
    candidate_models: list[str],
    ensemble_method: str = "probability_space_aggregation",
    freq: Optional[str] = None,
    domain: Optional[str] = None,
    term: Optional[str] = None,
    no_daily: bool = False,
    ttm_context_length: Optional[int] = None,
    ttm_pred_length: Optional[int] = None,
    ttm_scaling_data=None,
    device: str = "cpu",
    patchtst_use_fill_nan: bool = False,
    quantile_levels: list[float] = _DEFAULT_QUANTILE_LEVELS,
) -> QuantileEnsembleForecaster:
    """Build a QuantileEnsembleForecaster from a list of GIFT-eval model version strings.

    Each model version string is mapped to its forecaster class by prefix:
      - "ttm-*" / "granite-ttm-*"                            → TinyTimeMixerPreTrainedGiftModelForecaster
      - "patchtst-fm*" / "granite-patchtst-fm*" / "conf*"    → PatchTSTFMGiftModelForecaster
      - "flowstate-*" / "granite-flowstate-*"                → FlowstateGiftModelForecaster

    Args:
        candidate_models: List of model version keys (e.g. ["patchtst-fm-r1",
            "flowstate-r1.1", "ttm-r3-pt"]).
        ensemble_method: Aggregation method. One of:
            - "probability_space_aggregation" → aggregate_linear_pool (default)
            - "quantile_space_aggregation"    → aggregate_vincent
        freq: Frequency string forwarded to FlowState / TTM forecasters (e.g. "H", "D").
            Defaults to None.
        domain: Domain string forwarded to FlowState forecaster. Defaults to None.
        term: Forecast term ("short" / "medium" / "long") forwarded to TTM.
            Defaults to None.
        no_daily: no_daily flag forwarded to FlowState forecaster. Defaults to False.
        ttm_context_length: Minimum context window size in dataset used for TTM model selection.
            Only used when a TTM model is in candidate_models. Defaults to 512.
        ttm_pred_length: Maximum forecast horizon used for TTM model selection.
            Only used when a TTM model is in candidate_models. Defaults to 96.
        device: PyTorch device string ("cuda" / "cpu").
        patchtst_use_fill_nan: Whether to fill NaN values using ``_fill_nan`` for
            PatchTST-FM forecasters. Defaults to False.
        quantile_levels: Quantile levels for the ensemble output. Defaults to
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9].

    Returns:
        An initialised QuantileEnsembleForecaster ready to call.

    Raises:
        ValueError: If ensemble_method is not recognised or a model version
            cannot be mapped to a known forecaster class.

    Example:
        >>> ensemble = build_gift_ensemble(
        ...     candidate_models=["patchtst-fm-r1", "flowstate-r1.1"],
        ...     ensemble_method="probability_space_aggregation",
        ...     freq="H",
        ...     term="short",
        ... )
        >>> result = ensemble(data, prediction_length=[24])
    """
    if quantile_levels is None:
        quantile_levels = _DEFAULT_QUANTILE_LEVELS

    if ensemble_method not in _ENSEMBLE_FUNCTIONS:
        raise ValueError(
            f"Unknown ensemble_method '{ensemble_method}'. "
            f"Choose from: {list(_ENSEMBLE_FUNCTIONS.keys())}"
        )
    ensemble_function = _ENSEMBLE_FUNCTIONS[ensemble_method]

    members = []
    for model_version in candidate_models:
        try:
            forecaster = None
            if "ttm-r3" in model_version:
                forecaster = TinyTimeMixerPreTrainedGiftModelForecaster(
                    model_version=model_version,
                    device=device,
                    use_get_gift_model=True,
                    context_length=ttm_context_length,
                    prediction_length = ttm_pred_length,
                    freq=freq,
                    term=term,
                    scaling_data=ttm_scaling_data,
                )
                if forecaster.model.config.prediction_length < ttm_pred_length:
                    continue
            elif "patchtst-fm" in model_version:
                forecaster = PatchTSTFMGiftModelForecaster(
                    model_version=model_version,
                    device=device,
                    use_fill_nan=patchtst_use_fill_nan,
                )
            elif "flowstate" in model_version:
                forecaster = FlowstateGiftModelForecaster(
                    model_version=model_version,
                    device=device,
                    freq=freq,
                    domain=domain,
                    no_daily=no_daily,
                )
            if forecaster is not None:
                members.append(RecordingForecaster(model_version, forecaster))
            else:
                logging.warning(f"Skipping model '{model_version}', couldn't be loaded")
        except Exception as e:
            logging.warning(
                f"Skipping model '{model_version}': failed to load with error: {e}"
            )

    return QuantileEnsembleForecaster(
        members=members,
        quantile_levels=quantile_levels,
        ensemble_function=(
            _single_member_result if len(members) == 1 else ensemble_function
        ),
    )
