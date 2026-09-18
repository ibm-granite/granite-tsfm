"""Forecast ensemble aggregation functions.

This module provides functions for aggregating multiple forecasting models through:
- Linear pool aggregation aka probability space aggregation (used in MoiraiAgent)
- Vincentian style aka quantile space aggregation (used in Timescopilot)
- IQR-weighted aggregation: inverse-IQR softmax weights per model, then weighted mean
"""

import logging
from enum import Enum
from typing import Optional, Union, Protocol, runtime_checkable

import numpy as np
from sklearn.isotonic import IsotonicRegression
from tsfm_public.toolkit.forecasters import ForecastResult

logger = logging.getLogger(__name__)


class AggregationMethod(str, Enum):
    """Supported ensemble aggregation methods."""

    PROBABILITY_SPACE = "probability_space"
    QUANTILE_SPACE = "quantile_space"
    IQR_WEIGHTED = "iqr_weighted"


@runtime_checkable
class ForecastEnsembleFn(Protocol):
    """Protocol for ensemble aggregation functions.

    Any callable satisfying this signature can be used as an ensemble method
    in QuantileEnsembleTimeSeriesForecast. Extra arguments can be pre-bound
    using functools.partial before passing the function in.

    Args:
        ensemble_predictions: Predictions from multiple models.
            Shape: (n_samples, prediction_length, n_targets, n_quantiles, n_models)
        quantile_levels: List of quantile levels in [0, 1] of length n_quantiles,
            corresponding to the n_quantiles dimension of ensemble_predictions.
            The aggregated output is returned at these same quantile levels.
        weights: Optional weights for each model, shape (n_models,).
        **kwargs: Additional keyword arguments (implementation-specific).

    Returns:
        ForecastResult with aggregated forecasts and the aggregation method in metadata.
    """

    def __call__(
        self,
        ensemble_predictions: Union[np.ndarray, list],
        quantile_levels: list[float],
        weights: Union[np.ndarray, list, None] = None,
        **kwargs,
    ) -> ForecastResult: ...






def aggregate_linear_pool(
    predictions: np.ndarray,
    quantile_levels: list[float],
    weights: Union[np.ndarray, None]= None,
) -> ForecastResult:
    """Function for probability space aggregation.
    
    Concatenates predictions across quantiles and models dimensions,
    then computes target quantiles.
    
    Args:
        predictions: Predictions array with shape (..., n_quantiles, n_ensembles)
                     where ... represents any number of leading dimensions
                     (e.g., n_samples, prediction_length, n_targets).
        quantile_levels: Target quantile levels to compute (output quantiles).
                         Values must be in [0, 1].
        weights: Optional 1D array of shape (n_ensembles,) for model weighting.
                 If None, uniform (unweighted) aggregation is used.
                 Must be non-negative and sum to 1.
        
    Returns:
        ForecastResult
    """
    inputsOK, msg = _validate_ensemble_inputs(ensemble_predictions=predictions, quantile_levels=quantile_levels, weights=weights)

    if not inputsOK:
        return ForecastResult(success=inputsOK, message=msg)

    # Get the shape information
    # predictions shape: (..., n_quantiles, n_models)
    original_shape = predictions.shape
    n_quantiles = original_shape[-2]
    n_ensembles = original_shape[-1]
    
    # Reshape to combine the last two dimensions (quantiles and ensembles)
    # New shape: (..., n_quantiles * n_ensembles)
    leading_dims = original_shape[:-2]
    concatenated = predictions.reshape(*leading_dims, n_quantiles * n_ensembles) #linear pooling
    
    if weights is None:
        # Compute the specified quantiles along the concatenated dimension
        # np.quantile expects quantiles in [0, 1] range
        aggregated_predictions = np.quantile(concatenated, quantile_levels, axis=-1)
        
        # Transpose to get shape (..., len(quantile_levels)) because np.quantile returns shape (len(quantile_levels), ...)
        aggregated_predictions = np.moveaxis(aggregated_predictions, 0, -1)
        
    else:
        # Weighted quantile computation
        
        # Validate weights
        assert weights.shape[0] == n_ensembles, \
            f"Weights dimension {weights.shape[0]} must match n_models {n_ensembles}"
        assert np.all(weights >= 0), \
            f"All weights must be non-negative"
        assert np.isclose(weights.sum(), 1.0), \
            f"Weights must sum to 1, got {weights.sum()}"
        
        # Flatten leading dimensions to 2D for easier processing
        # Shape: (prod(leading_dims), n_quantiles * n_ensembles)
        leading_size = int(np.prod(leading_dims)) if leading_dims else 1
        concatenated_flat = concatenated.reshape(leading_size, n_quantiles * n_ensembles)

        # Expand weights from (n_models,) to (n_quantiles * n_models,)
        # Each model gets the same weight across all its quantile predictions
        expanded_weights = np.broadcast_to(weights[np.newaxis, :], (n_quantiles, n_ensembles)).ravel()
        # Normalize weights to sum to 1 because they were broadcasted to n_quantiles dimension
        expanded_weights = expanded_weights / n_quantiles

        # Compute weighted quantiles using helper function
        # concatenated_flat shape: (leading_size, n_quantiles * n_ensembles)
        # weights_normalized shape: (n_quantiles * n_ensembles,)
        # Result shape: (leading_size, len(quantile_levels))
        aggregated_predictions = _compute_weighted_quantiles(
            concatenated_flat, expanded_weights, quantile_levels
        )
        
        # Reshape back to original leading dimensions
        # Shape: (*leading_dims, len(quantile_levels))
        aggregated_predictions = aggregated_predictions.reshape(*leading_dims, len(quantile_levels))


    result = _build_ensemble_result(aggregated_predictions=aggregated_predictions,
                quantile_levels=quantile_levels, weights=weights,
                method="aggregate_linear_pool",
                msg = "Ensembling completed successfully.",
                n_models=n_ensembles)

    return result




def aggregate_vincent(
    predictions: np.ndarray,
    quantile_levels: list[float],
    weights: Union[np.ndarray, None] = None,
    aggregation= 'median',
) -> ForecastResult:
    """Vincentian aggregation of quantiles with isotonic regression.
    
    Performs weighted aggregation across ensemble members for each quantile level,
    then applies isotonic regression to ensure monotonicity.
    
    Args:
        predictions: Predictions from multiple models.
            Shape: (n_samples, prediction_length, n_targets, n_quantiles, n_models)
            Can be a numpy array or nested list.
        quantile_levels: List of quantile levels in [0, 1] of length n_quantiles,
            corresponding to the n_quantiles dimension of ensemble_predictions.
            The aggregated output is returned at these same quantile levels.
        weights: Optional weights for each model. Can be:
            - None: Equal weights (uniform or median aggregation)
            - 1D array of shape (n_models,): Same weight per model across all predictions
        aggregation: Aggregation method, either 'median' or 'weighted'.

    Returns:
        ForecastResult with:
            - success: True if aggregation succeeded
            - message: Status message
            - metadata["method"]: The aggregation method used
            - predicted: Point predictions, shape (n_samples, prediction_length, n_targets)
            - predicted_quantiles: Aggregated quantiles
            - metadata: Dict with quantile_levels, n_models, weights info

    Raises:
        ValueError: If inputs have incompatible shapes or invalid values.

    Example:
        >>> predictions = np.random.rand(10, 24, 2, 9, 5)  # 10 samples, 24 steps, 2 targets, 9 quantiles, 5 models
        >>> levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        >>> # Probability space aggregation (default)
        >>> result = aggregate_vincent(predictions, levels)
        >>> print(result.predicted_quantiles.shape)  # (10, 24, 2, 9)
    """

    inputsOK, msg = _validate_ensemble_inputs(ensemble_predictions=predictions, quantile_levels=quantile_levels, weights=weights)


    if not inputsOK:
        return ForecastResult(success=inputsOK, message=msg)
    
    predictions = np.asarray(predictions)
    
    # Get the shape information
    # predictions shape: (..., n_quantiles, n_ensembles)
    original_shape = predictions.shape
    n_quantiles = original_shape[-2]
    n_ensembles = original_shape[-1]
    leading_dims = original_shape[:-2]
    leading_size = np.prod(leading_dims) if leading_dims else 1

    if len(quantile_levels) != n_quantiles:
                return ForecastResult(
                    success=False,
                    message=f"For QUANTILE_SPACE aggregation, quantile_levels length ({len(quantile_levels)}) must match n_quantiles dimension ({n_quantiles})",
                    metadata={"method": "aggregate_vincent"},
                )

    
    # 1. Sort the predictions along the quantile dimension (-2) based on increasing order of quantile levels
    # Get the sorting indices for quantile_levels
    sorted_indices = np.argsort(quantile_levels)
    sorted_quantile_levels = np.array(quantile_levels)[sorted_indices]
    
    # Apply sorting to the quantile dimension (-2), quantile_levels should indicate the quantiles of predictions dimension -2.
    sorted_predictions = np.take(predictions, sorted_indices, axis=-2)
    
    if weights is None:
        # 2. Perform aggregation in the ensembles dimension (-1)
        if aggregation == 'median':
            ensemble_aggregation = np.median(sorted_predictions, axis=-1) #leading_dim , n_quantiles
        else: #default is mean
            ensemble_aggregation = np.mean(sorted_predictions, axis=-1) #leading_dim , n_quantiles
    else:
        # Validate weights
        assert weights.shape[0] == n_ensembles, \
            f"Weights dimension {weights.shape[0]} must match n_models {n_ensembles}"
        assert np.all(weights >= 0), \
            f"All weights must be non-negative"
        assert np.isclose(weights.sum(), 1.0), \
            f"Weights must sum to 1, got {weights.sum()}"

        sorted_predictions_flat = sorted_predictions.reshape(leading_size*n_quantiles, n_ensembles)
        if aggregation == 'median':
            ensemble_aggregation = _compute_weighted_quantiles(
                sorted_predictions_flat, weights, [0.5]
            )[...,0] #leading_size x n_quantiles
        else: #default is weighted mean
            ensemble_aggregation = np.sum(sorted_predictions_flat*weights[np.newaxis,:], axis=-1)/np.sum(weights[np.newaxis,:], axis=-1) #leading_size x n_quantiles

    # 3. Apply IsotonicRegression across the quantile dimension (now -1 after median aggregation)
    # median_predictions shape: (..., n_quantiles)
    
    # Reshape to 2D for easier processing: (batch_size, n_quantiles)
    # batch_size = np.prod(leading_dims) if leading_dims else 1
    ensemble_aggregation = ensemble_aggregation.reshape(leading_size, n_quantiles)
    
    # Apply isotonic regression only to non-monotonic rows
    # Vectorized monotonicity check for all rows
    is_monotonic = np.all(ensemble_aggregation[:, 1:] >= ensemble_aggregation[:, :-1], axis=1)
    non_monotonic_indices = np.where(~is_monotonic)[0]
    
    # Start with a copy of the original predictions
    isotonic_predictions = np.copy(ensemble_aggregation)
    
    # Only apply isotonic regression to non-monotonic rows
    if len(non_monotonic_indices) > 0:
        ir: IsotonicRegression = IsotonicRegression(increasing=True)
        for i in non_monotonic_indices:
            isotonic_predictions[i] = ir.fit_transform(sorted_quantile_levels, ensemble_aggregation[i])
    
    # Reshape back to original leading dimensions
    aggregated_predictions = isotonic_predictions.reshape(*leading_dims, n_quantiles)

    result = _build_ensemble_result(aggregated_predictions=aggregated_predictions,
                quantile_levels=quantile_levels, weights=weights,
                method="aggregate_vincent",
                msg = "Ensembling completed successfully.",
                n_models=n_ensembles)

    return result



def _validate_ensemble_inputs(ensemble_predictions: Union[np.ndarray, list],
                              quantile_levels: list[float],
                              weights: Union[np.ndarray, list, None] = None) -> tuple[bool, str]:
        """Validates ensemble inputs
        
        returns tuple (bool, str) where bool indiates validation and str contains error message
        """

        # Convert to numpy array
        predictions = np.asarray(ensemble_predictions)
        
        # Validate shape: (n_samples, prediction_length, n_targets, n_quantiles, n_models)
        if predictions.ndim != 5:
            return False, f"Expected 5D array with shape (n_samples, prediction_length, n_targets, n_quantiles, n_models), got {predictions.ndim}D array",
        
        n_samples, pred_len, n_targets, n_quantiles, n_models = predictions.shape
        
        # Validate quantile levels
        if not quantile_levels:
            return False, "quantile_levels cannot be empty" 
        
        if not all(0 <= q <= 1 for q in quantile_levels):
            return False, "All quantile levels must be in [0, 1]"
        
        # Validate weights if provided
        if weights is not None:
            weights = np.asarray(weights)
            # Weights must be 1D with shape (n_models,)
            if weights.ndim != 1:
                return False, f"Weights must be 1D array, got {weights.ndim}D array with shape {weights.shape}"
                
            if weights.shape[0] != n_models:
                return False,f"Weights must have length {n_models} (n_models), got {weights.shape[0]}"

        return True, "Ensemble input validation succeeded."



def _build_ensemble_result(aggregated_predictions, quantile_levels, weights, method:str, msg:str, n_models:int):
    """Builds ForecastResult object from aggregated predictions"""

    # Compute point predictions only if 0.5 is in quantile_levels
    predicted_list = None
    if 0.5 in quantile_levels:
        median_idx = quantile_levels.index(0.5)
        predicted = aggregated_predictions[..., median_idx]
        predicted_list = predicted.tolist()
    
    # Convert to nested lists
    predicted_quantiles_list = aggregated_predictions.tolist()
    
    return ForecastResult(
        success=True,
        message=msg,
        predicted=predicted_list,
        predicted_quantiles=predicted_quantiles_list,
        metadata={
            "method": method,
            "quantile_levels": quantile_levels,
            "n_models": n_models,
            "weights": weights.tolist() if weights is not None else None,
        }
    )



def aggregate_iqr_weighted(
    predictions: np.ndarray,
    quantile_levels: list[float],
    weights: Union[np.ndarray, None] = None,
    temperature: float = 1.0,
    max_weight: Optional[float] = None,
) -> ForecastResult:
    """IQR-based inverse-uncertainty weighted aggregation with isotonic regression.

    For each model computes IQR = upper_quantile - lower_quantile (closest available
    pair straddling 0.25 / 0.75). Models with a narrower interval (lower IQR) are
    more confident and receive a higher weight via softmax over inverse-IQR scores.

    A user-supplied ``weights`` array acts as a prior multiplier on the softmax scores.
    Set a model's entry to 0 to exclude it entirely regardless of its IQR.

    When ``max_weight`` is set, no single model's weight may exceed that value. Any
    excess is redistributed proportionally among the still-uncapped models, iterating
    until all weights are within the cap. This prevents a single highly-confident model
    from dominating the ensemble while still favouring lower-uncertainty models.

    Args:
        predictions: Predictions from multiple models.
            Shape: (n_samples, prediction_length, n_targets, n_quantiles, n_models)
        quantile_levels: List of quantile levels in [0, 1] of length n_quantiles,
            corresponding to the n_quantiles dimension of predictions.
            The aggregated output is returned at these same quantile levels.
        weights: Optional 1-D prior weights of shape (n_models,). Multiplied with
            the IQR-derived softmax weights before re-normalisation.
            Set a model entry to 0 to exclude it. None = equal prior weights.
        temperature: Softmax temperature (> 0). Higher values push weights towards
            uniform; lower values sharpen the winner. Default: 1.0.
        max_weight: Optional upper bound on any single model's final weight.
            Must be in ``[1/n_models, 1]``. Values below ``1/n_models`` are
            mathematically infeasible (weights cannot sum to 1 while all staying
            under that threshold). When None (default) no capping is applied.

    Returns:
        ForecastResult

    Example:
        >>> predictions = np.random.rand(10, 24, 2, 9, 5)
        >>> levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        >>> result = aggregate_iqr_weighted(predictions, levels)
        >>> np.array(result.predicted_quantiles).shape  # (10, 24, 2, 9)
        >>> # With per-model weight cap of 0.4:
        >>> result = aggregate_iqr_weighted(predictions, levels, max_weight=0.4)
    """
    inputsOK, msg = _validate_ensemble_inputs(
        ensemble_predictions=predictions, quantile_levels=quantile_levels, weights=weights
    )
    if not inputsOK:
        return ForecastResult(success=inputsOK, message=msg)

    if len(quantile_levels) != predictions.shape[-2]:
        return ForecastResult(
            success=False,
            message=f"For IQR_WEIGHTED aggregation, quantile_levels length ({len(quantile_levels)}) must match n_quantiles dimension ({predictions.shape[-2]})",
            metadata={"method": "aggregate_iqr_weighted"},
        )

    predictions = np.asarray(predictions, dtype=float)
    original_shape = predictions.shape
    n_quantiles = original_shape[-2]
    n_models = original_shape[-1]
    leading_dims = original_shape[:-2]
    leading_size = int(np.prod(leading_dims)) if leading_dims else 1
    quantile_arr = np.asarray(quantile_levels, dtype=float)

    if max_weight is not None:
        min_valid = 1.0 / n_models
        if not (min_valid <= max_weight <= 1.0):
            return ForecastResult(
                success=False,
                message=(
                    f"max_weight must be in [1/n_models, 1] = [{min_valid:.6g}, 1.0], "
                    f"got {max_weight}"
                ),
                metadata={"method": "aggregate_iqr_weighted"},
            )

    # 1. Identify lower / upper quantile indices for IQR
    lower_candidates = quantile_arr[quantile_arr <= 0.25]
    upper_candidates = quantile_arr[quantile_arr >= 0.75]
    lower_q = lower_candidates[-1] if len(lower_candidates) > 0 else quantile_arr[0]
    upper_q = upper_candidates[0] if len(upper_candidates) > 0 else quantile_arr[-1]
    lower_idx = int(np.argmin(np.abs(quantile_arr - lower_q)))
    upper_idx = int(np.argmin(np.abs(quantile_arr - upper_q)))

    # 2. Compute IQR per model: shape (*leading_dims, n_models)
    iqr = predictions[..., upper_idx, :] - predictions[..., lower_idx, :]
    iqr = np.where(np.isnan(iqr), 1e10, iqr)

    # 3. Inverse-IQR softmax weights
    inverse_iqr = 1.0 / (iqr + 1e-6)
    scaled = inverse_iqr / temperature
    scaled -= scaled.max(axis=-1, keepdims=True)  # numerically stable
    exp_scaled = np.exp(scaled)
    softmax_weights = exp_scaled / exp_scaled.sum(axis=-1, keepdims=True)

    # Apply prior weights as multiplicative mask
    if weights is not None:
        softmax_weights = softmax_weights * np.asarray(weights, dtype=float)
        denom = softmax_weights.sum(axis=-1, keepdims=True)
        softmax_weights = softmax_weights / (denom + 1e-10)

    # Iterative weight capping: redistribute excess from over-limit models to uncapped ones.
    # Models with weight=0 (excluded via prior) never receive redistributed excess.
    if max_weight is not None:
        participating = (softmax_weights > 0).astype(float)          # (..., n_models)
        for _ in range(n_models):
            over_mask = softmax_weights > max_weight
            if not over_mask.any():
                break
            excess = np.clip(softmax_weights - max_weight, 0.0, None)
            total_excess = excess.sum(axis=-1, keepdims=True)        # (..., 1)
            softmax_weights = np.where(over_mask, max_weight, softmax_weights)
            uncapped = participating * (softmax_weights < max_weight).astype(float)
            uncapped_sum = (softmax_weights * uncapped).sum(axis=-1, keepdims=True)
            uncapped_sum = np.where(uncapped_sum < 1e-12, 1e-12, uncapped_sum)
            softmax_weights = softmax_weights + softmax_weights * uncapped * (total_excess / uncapped_sum)

    # 4. Weighted mean across models for every quantile level
    w = softmax_weights[..., np.newaxis, :]        # (*leading, 1, n_models)
    ensemble_aggregation = (predictions * w).sum(axis=-1)  # (*leading, n_quantiles)

    # 5. Isotonic regression to enforce quantile monotonicity
    sorted_indices = np.argsort(quantile_levels)
    sorted_quantile_levels = quantile_arr[sorted_indices]
    sorted_agg = np.take(ensemble_aggregation, sorted_indices, axis=-1)

    flat = sorted_agg.reshape(leading_size, n_quantiles)
    is_monotonic = np.all(flat[:, 1:] >= flat[:, :-1], axis=1)
    non_monotonic = np.where(~is_monotonic)[0]
    isotonic_flat = flat.copy()
    if len(non_monotonic) > 0:
        ir = IsotonicRegression(increasing=True)
        for i in non_monotonic:
            isotonic_flat[i] = ir.fit_transform(sorted_quantile_levels, flat[i])

    aggregated_predictions = isotonic_flat.reshape(*leading_dims, n_quantiles)

    return _build_ensemble_result(
        aggregated_predictions=aggregated_predictions,
        quantile_levels=quantile_levels,
        weights=weights,
        method="aggregate_iqr_weighted",
        msg="IQR-weighted aggregation completed successfully.",
        n_models=n_models,
    )


def _compute_weighted_quantiles(
    samples: np.ndarray,
    weights: np.ndarray,
    quantile_levels: list[float],
) -> np.ndarray:
    """Compute weighted quantiles for a 2D array.
    
    Computes weighted quantiles across the samples dimension (axis=-1)
    independently for each row in the first dimension.
    
    Args:
        samples: 2D array of shape (n_rows, n_samples) where quantiles are
                 computed across n_samples for each row independently
        weights: 1D array of shape (n_samples,) containing normalized weights
                 Must sum to 1.
        quantile_levels: List of quantile levels to compute, values in [0, 1]
        
    Returns:
        np.ndarray: Weighted quantiles with shape (n_rows, len(quantile_levels))
        
    Example:
        >>> samples = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])  # 2 rows, 4 samples each
        >>> weights = np.array([0.1, 0.2, 0.4, 0.3])  # weights for 4 samples
        >>> result = _compute_weighted_quantiles(samples, weights, [0.5])
        >>> result.shape  # (2, 1) - median for each row
    """
    n_rows = samples.shape[0]
    n_samples = samples.shape[1]
    
    # Validate inputs
    assert weights.shape[0] == n_samples, \
        f"Weights length {weights.shape[0]} must match samples dimension {n_samples}"
    assert np.all(weights >= 0), \
        f"All weights must be non-negative"
    assert np.isclose(weights.sum(), 1.0), \
        f"Weights must sum to 1, got {weights.sum()}"
    
    # Sort values along samples axis (independently for each row)
    sorted_indices = np.argsort(samples, axis=-1)  # Shape: (n_rows, n_samples)
    sorted_values = np.take_along_axis(samples, sorted_indices, axis=-1)  # Shape: (n_rows, n_samples)
    
    # Reorder weights according to sorted values
    sorted_weights = np.take(weights, sorted_indices)  # Shape: (n_rows, n_samples)
    
    # Compute cumulative sum of sorted weights
    cumsum_weights = np.cumsum(sorted_weights, axis=-1)  # Shape: (n_rows, n_samples)
    
    # Find indices where cumulative weight >= each target quantile level
    indices = np.zeros((n_rows, len(quantile_levels)), dtype=int)
    weighted_quantiles = np.empty((n_rows, len(quantile_levels)))
    
    # For each row, find weighted quantiles across samples
    for i in range(n_rows):
        # Find insertion points for each quantile level in cumulative weights
        indices[i, :] = np.searchsorted(cumsum_weights[i], quantile_levels, side='right')
        # Clip indices to valid range to avoid out-of-bounds (safety check)
        indices[i, :] = np.clip(indices[i, :], 0, n_samples - 1)
        # Extract weighted quantile values at the found indices
        weighted_quantiles[i, :] = sorted_values[i, indices[i, :]]
    
    return weighted_quantiles
