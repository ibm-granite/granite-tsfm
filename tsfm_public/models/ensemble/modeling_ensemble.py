
import logging

from tsfm_public.toolkit.forecasters import Forecaster
from tsfm_public.toolkit.ensemble_aggregation import (
    ForecastEnsembleFn,
    aggregate_linear_pool,
    aggregate_vincent,
    aggregate_iqr_weighted,
)
import numpy as np

class QuantileEnsembleTimeSeriesForecast:
    """Ensemble forecaster that aggregates predictions from multiple member forecasters.

    Iterates over member forecasters, collects their outputs via
    `forecast_for_ensemble`, and combines them using a pluggable aggregation
    function that satisfies the ForecastEnsembleFn protocol.

    Args:
        members: List of Forecaster instances. Each must implement
            `forecast_for_ensemble`.
        quantile_levels: List of quantile levels in [0, 1] of length n_quantiles,
            corresponding to the n_quantiles dimension produced by each member.
            The aggregated output is returned at these same quantile levels.
        ensemble_function: Aggregation function satisfying the ForecastEnsembleFn
            protocol. Defaults to aggregate_ensemble_forecasts (probability space).
            Extra arguments can be pre-bound using functools.partial.
        weights: Optional weights for each member model, shape (n_models,).
            Passed directly to ensemble_function.

    Example:
        >>> # Default: probability space aggregation
        >>> ensemble = QuantileEnsembleTimeSeriesForecast(
        ...     members=[forecaster_a, forecaster_b],
        ...     quantile_levels=[0.1, 0.5, 0.9],
        ... )

        >>> # Quantile space aggregation via functools.partial
        >>> from functools import partial
        >>> ensemble = QuantileEnsembleTimeSeriesForecast(
        ...     members=[forecaster_a, forecaster_b],
        ...     quantile_levels=[0.1, 0.5, 0.9],
        ...     ensemble_function=partial(
        ...         aggregate_ensemble_forecasts,
        ...         aggregation_method=AggregationMethod.QUANTILE_SPACE,
        ...     ),
        ... )

        >>> # Custom aggregation function
        >>> ensemble = QuantileEnsembleTimeSeriesForecast(
        ...     members=[forecaster_a, forecaster_b],
        ...     quantile_levels=[0.1, 0.5, 0.9],
        ...     ensemble_function=my_custom_aggregator,
        ... )

        >>> # IQR-weighted: confident models (narrower intervals) get higher weight
        >>> ensemble = QuantileEnsembleTimeSeriesForecast(
        ...     members=[forecaster_a, forecaster_b],
        ...     quantile_levels=[0.1, 0.5, 0.9],
        ...     ensemble_function=aggregate_iqr_weighted,
        ... )
    """

    def __init__(
            self,
            members: list[Forecaster],
            quantile_levels: list[float],
            ensemble_function: ForecastEnsembleFn = aggregate_linear_pool,
            weights=None,
            **kwargs
    ):
        self.members = members
        self.quantile_levels = quantile_levels
        self.ensemble_function = ensemble_function
        self.weights = weights


    def __call__(self, data, **kwargs):
        """Run all member forecasters and aggregate their predictions.

        Args:
            data: Input data compatible with all member forecasters.
            **kwargs: Additional keyword arguments forwarded to each member's
                `forecast_for_ensemble` call.

        Returns:
            ForecastResult from the aggregation function. The aggregation method is
            recorded in metadata["method"]. Actuals and cutoff dates are not provided
            by array-only aggregation functions.
        """

        # Collect forecasts from all members
        # Each forecast: (n_samples, pred_len, n_targets, n_quantiles)
        forecasts = []
        for memb in self.members:
            try:
                fcast = memb.forecast_for_ensemble(data, **kwargs)
                forecasts.append(np.asarray(fcast))
            except Exception as e:
                logging.warning(
                    f"Skipping member {memb.__class__.__name__}: inference failed with error: {e}"
                )

        # Stack into (n_samples, pred_len, n_targets, n_quantiles, n_models)
        ensemble_predictions = np.stack(forecasts, axis=-1)

        # Aggregate using the provided ensemble function
        result = self.ensemble_function(
            ensemble_predictions, self.quantile_levels, weights=self.weights
        )

        return result
