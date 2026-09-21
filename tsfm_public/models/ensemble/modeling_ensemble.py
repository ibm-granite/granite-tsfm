
import logging
from functools import partial

from tsfm_public.models.ensemble.configuration_ensemble import ProbabilisticEnsembleConfig
from tsfm_public.toolkit.forecasters import (
    Forecaster,
    PatchTSTFMDataFramePipelineForecaster,
    FlowStateDataFramePipelineForecaster,
    TinyTimeMixerDataFramePipelineForecaster,
)
from tsfm_public.toolkit.ensemble_aggregation import (
    ForecastEnsembleFn,
    aggregate_linear_pool,
    aggregate_vincent,
    aggregate_iqr_weighted,
)
import numpy as np

class QuantileEnsembleForecaster:
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
        >>> ensemble = QuantileEnsembleForecaster(
        ...     members=[forecaster_a, forecaster_b],
        ...     quantile_levels=[0.1, 0.5, 0.9],
        ... )

        >>> # Quantile space aggregation via functools.partial
        >>> from functools import partial
        >>> ensemble = QuantileEnsembleForecaster(
        ...     members=[forecaster_a, forecaster_b],
        ...     quantile_levels=[0.1, 0.5, 0.9],
        ...     ensemble_function=partial(
        ...         aggregate_ensemble_forecasts,
        ...         aggregation_method=AggregationMethod.QUANTILE_SPACE,
        ...     ),
        ... )

        >>> # Custom aggregation function
        >>> ensemble = QuantileEnsembleForecaster(
        ...     members=[forecaster_a, forecaster_b],
        ...     quantile_levels=[0.1, 0.5, 0.9],
        ...     ensemble_function=my_custom_aggregator,
        ... )

        >>> # IQR-weighted: confident models (narrower intervals) get higher weight
        >>> ensemble = QuantileEnsembleForecaster(
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

    @classmethod
    def from_config(cls, config: ProbabilisticEnsembleConfig, device=None):
        """Construct pretrained members from a validated ensemble recipe.

        Load the recipe with ``ProbabilisticEnsembleConfig.from_pretrained``.
        TTM selects its revision at inference time from context and horizon.
        """
        config.validate()
        forecasters = {
            "patchtst": PatchTSTFMDataFramePipelineForecaster,
            "flowstate": FlowStateDataFramePipelineForecaster,
            "ttm": TinyTimeMixerDataFramePipelineForecaster,
        }
        ensemble_function = aggregate_linear_pool
        if config.aggregation_method == "iqr_weighted":
            ensemble_function = partial(aggregate_iqr_weighted, **config.iqr_weighted_options)
        members = [
            forecasters[member["forecaster_type"]](
                device=device, **{key: value for key, value in member.items() if key != "forecaster_type"}
            )
            for member in config.members
        ]
        return cls(
            members=members,
            quantile_levels=list(config.quantile_levels),
            ensemble_function=ensemble_function,
            weights=np.asarray(config.weights) if config.weights is not None else None,
        )


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
        kwargs.setdefault("quantile_levels", self.quantile_levels)
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
