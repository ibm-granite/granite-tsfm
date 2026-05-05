# Copyright contributors to the TSFM project
#
import numpy as np
from scipy.stats import cauchy, chi2

# from tsfm_public.toolkit.conformal import (
#     AdaptiveWeightedConformalScoreWrapper,
#     PostHocProbabilisticProcessor,
#     NonconformityScores,
#     nonconformity_score_functions,
# )
from .conformal import (
    AdaptiveWeightedConformalScoreWrapper,
    PostHocProbabilisticProcessor,
    nonconformity_score_functions,
)


"""
P-VALUE AGGREGATION FUNCTIONS
"""


def fisher_method_2d(p_values: np.ndarray) -> np.ndarray:
    """
    Combine p-values using Fisher's method along axis=1.

    Parameters
    ----------
    p_values : array-like, shape (n_rows, n_tests)
        Input p-values.

    Returns
    -------
    combined_p : np.ndarray, shape (n_rows,)
        Combined p-values for each row.
    """
    p_values = np.asarray(p_values)
    if p_values.ndim != 2:
        raise ValueError("p_values must be a 2D array")

    valid = ~np.isnan(p_values)
    k = valid.sum(axis=1)
    # Fisher statistic per row
    X = -2.0 * np.nansum(np.log(p_values), axis=1)
    # Survival function (1 - CDF) of chi-square
    combined_p = chi2.sf(X, df=2 * k)
    return combined_p


def harmonic_mean_pvalue_2d(p_values: np.ndarray) -> np.ndarray:
    """
    Combine p-values using the Harmonic Mean P-value (HMP) along axis=1.

    Uses the conservative adjustment:
        p_combined = min(1, HMP * e * log(k))

    Parameters
    ----------
    p_values : array-like, shape (n_rows, n_tests)
        Input p-values.

    Returns
    -------
    combined_p : np.ndarray, shape (n_rows,)
        Combined p-values for each row.
    """
    p_values = np.asarray(p_values)
    if p_values.ndim != 2:
        raise ValueError("p_values must be a 2D array")

    valid = ~np.isnan(p_values)
    k = valid.sum(axis=1)
    # Harmonic mean per row
    hmp = k / np.nansum(1.0 / p_values, axis=1)
    # Conservative adjustment (Wilson 2019)
    combined_p = np.minimum(1.0, hmp * np.e * np.log(k))
    return combined_p


def tippett_method_2d(p_values: np.ndarray) -> np.ndarray:
    """
    Combine p-values using Tippett's method along axis=1.

    Uses the minimum p-value in each row:
        p_combined = 1 - (1 - p_min)^k

    Parameters
    ----------
    p_values : array-like, shape (n_rows, n_tests)
        Input p-values.

    Returns
    -------
    combined_p : np.ndarray, shape (n_rows,)
        Combined p-values for each row.
    """
    p_values = np.asarray(p_values)
    if p_values.ndim != 2:
        raise ValueError("p_values must be a 2D array")

    valid = ~np.isnan(p_values)
    k = valid.sum(axis=1)
    p_min = np.nanmin(p_values, axis=1)
    combined_p = 1.0 - (1.0 - p_min) ** k
    return combined_p


def cauchy_combination_2d(p_values: np.ndarray) -> np.ndarray:
    """
    Combine p-values using Cauchy combination method along axis=1.

    Uses the Cauchy combination test which is robust to correlations.

    Parameters
    ----------
    p_values : array-like, shape (n_rows, n_tests)
        Input p-values.

    Returns
    -------
    combined_p : np.ndarray, shape (n_rows,)
        Combined p-values for each row.
    """
    p_values = np.asarray(p_values)
    if p_values.ndim != 2:
        raise ValueError("p_values must be a 2D array")

    # Create a copy to avoid modifying the original
    p_values_clipped = p_values.copy()

    # Only clip non-NaN values to avoid numerical issues
    valid_mask = ~np.isnan(p_values)
    p_values_clipped[valid_mask] = np.clip(p_values[valid_mask], 1e-15, 1 - 1e-15)

    # Compute Cauchy statistic per row, handling NaN values
    # np.tan will propagate NaN, and np.nanmean will ignore them
    T = np.nanmean(np.tan((0.5 - p_values_clipped) * np.pi), axis=1)

    # Combined p-value using Cauchy CDF
    combined_p = 1 - cauchy.cdf(T)

    return combined_p


def mean_aggregation_2d(values: np.ndarray) -> np.ndarray:
    """Compute mean along axis=1, handling NaN values."""
    return np.nanmean(values, axis=1)


def median_aggregation_2d(values: np.ndarray) -> np.ndarray:
    """Compute median along axis=1, handling NaN values."""
    return np.nanmedian(values, axis=1)


def min_aggregation_2d(values: np.ndarray) -> np.ndarray:
    """Compute minimum along axis=1, handling NaN values."""
    return np.nanmin(values, axis=1)


def max_aggregation_2d(values: np.ndarray) -> np.ndarray:
    """Compute maximum along axis=1, handling NaN values."""
    return np.nanmax(values, axis=1)


aggregation_methods = {
    "HMC": harmonic_mean_pvalue_2d,
    "Fisher": fisher_method_2d,
    "Tippett": tippett_method_2d,
    "Cauchy": cauchy_combination_2d,
    "mean": mean_aggregation_2d,
    "median": median_aggregation_2d,
    "min": min_aggregation_2d,
    "max": max_aggregation_2d,
}


"""
Wasserstein-1 distance-based Adaptive Conformal Anomaly Scoring (W1ACAS) - Function Wrapper
"""


def get_forecast_conformal_adaptive_online_score(
    prediction_output,
    significance_level=0.01,
    aggregation_forecast_horizon="median",
    nonconformity_score="absolute_error",
    forecast_steps=None,
    aggregation_features=None,
    n_epochs=1,
    n_batch_update=10,
    lr=0.001,
    prior_past_weights_value=0,
    return_weights=False,
    align_forecast=True,
):
    """
    Compute adaptive conformal anomaly scores (p-values) for time series forecasts using
    Wasserstein-1 distance-based Adaptive Conformal Anomaly Scoring (W1ACAS).

    This function implements an online adaptive conformal prediction framework for time series
    anomaly detection. It computes p-values for each time point by comparing forecast errors
    against past observations with adaptive weighting.

    Parameters
    ----------
    prediction_output : dict
        Dictionary containing forecast predictions and ground truth values.
        Required keys:
            - 'y_pred' : np.ndarray, shape (n_samples, n_horizons, n_features)
                Forecasted values
            - 'y_true' : np.ndarray, shape (n_samples, n_horizons, n_features)
                Ground truth values

    significance_level : float, default=0.01
        Target significance level (alpha) for anomaly detection. This sets the p-value
        resolution - lower values provide finer resolution but require more calibration data.
        Determines the minimum calibration window size as ceil(1/alpha).

    aggregation_forecast_horizon : str or None, default="median"
        Method to aggregate p-values across forecast horizons. Options:
            - "median", "mean", "min", "max": Statistical aggregations
            - "Fisher", "HMC", "Tippett", "Cauchy": P-value combination methods
            - None: No aggregation, returns all horizons

    nonconformity_score : str, default="absolute_error"
        Type of nonconformity score to compute forecast errors.
        Common options: "absolute_error", "squared_error", etc.

    forecast_steps : int, list of int, or None, default=None
        Controls which forecast horizons are processed:
            - None: uses all available horizons from y_pred.shape[1].
            - int: uses all horizons from 0 up to (but not including) that integer,
              i.e. ix_h in range(forecast_steps). Equivalent to forecast steps 1..forecast_steps.
            - list of int: uses only the specified forecast steps (1-based). Each value k
              in the list corresponds to horizon index ix_h = k - 1. The align_forecast
              slicing will use max(forecast_steps) as the upper bound.

    aggregation_features : str or None, default=None
        Method to aggregate p-values across features. Options same as
        aggregation_forecast_horizon. If None, returns separate p-values per feature.

    n_epochs : int, default=1
        Number of optimization epochs for adaptive weight learning.

    n_batch_update : int, default=10
        Batch size for updating adaptive weights during online learning.

    lr : float, default=0.001
        Learning rate for adaptive weight optimization.

    prior_past_weights_value : int or str, default=0
        Initialization strategy for past weights in adaptive weighting:
            - 0: Uniform initialization
            - "proximity": Weight based on temporal proximity
            - Other numeric values: Custom initialization

    return_weights : bool, default=False
        If True, returns calibration scores and weights along with p-values.

    align_forecast : bool, default=True
        If True, applies forecast_horizon_aggregation to align forecasted and ground truth
        values such that each row corresponds to the same observation and each column
        represents the prediction at each horizon for that observation. This ensures proper
        temporal alignment for multi-step forecasts. When forecast_steps is a list, the
        slicing uses max(forecast_steps) as the upper bound.

    Returns
    -------
    scores : np.ndarray
        P-values (anomaly scores) for each time point. Shape depends on aggregation:
            - If both aggregations applied: (n_samples,)
            - If only forecast horizon aggregated: (n_samples, n_features)
            - If only features aggregated: (n_samples, n_horizons)
            - If no aggregation: (n_samples, n_horizons, n_features)
        Lower p-values indicate higher anomaly likelihood.

    weights_dict : dict, optional
        Only returned if return_weights=True. Contains:
            - 'cal_scores': Calibration scores for each horizon and feature
            - 'cal_weights': Adaptive weights for each horizon and feature

    Notes
    -----
    The function implements an adaptive conformal prediction framework where:
    1. Nonconformity scores are computed from forecast errors
    2. A sliding calibration window maintains recent scores
    3. Adaptive weights are learned to emphasize relevant historical data
    4. P-values are computed as the weighted proportion of calibration scores >= test score
    5. Multiple p-values (across horizons/features) are combined using statistical methods

    The adaptive weighting mechanism allows the method to adjust to non-stationary
    time series by giving more weight to recent, relevant calibration data.

    When forecast_steps is a list (e.g. [1, 3, 6]), only those specific forecast steps
    (1-based) are processed. Forecast step k corresponds to horizon index ix_h = k - 1.
    The output array retains the full horizon dimension (up to max(forecast_steps)), with
    NaN filled for horizons that were not processed.
    """
    y_pred = prediction_output["y_pred"]
    y_true = prediction_output["y_true"]
    # sample_index = prediction_output['sample_index']
    # label_train = prediction_output["label_train"]

    ## Resolve forecast_steps into:
    ##   - forecast_steps_max : int, upper bound for slicing (exclusive)
    ##   - horizon_indices    : list of ix_h values to process
    if forecast_steps is None:
        forecast_steps_max = y_pred.shape[1]
        horizon_indices = list(range(forecast_steps_max))
    elif isinstance(forecast_steps, (list, tuple)):
        # forecast_steps is a list of 1-based step numbers; convert to 0-based indices
        horizon_indices = [int(s) - 1 for s in forecast_steps]
        forecast_steps_max = max(horizon_indices) + 1  # inclusive upper bound for slicing
    else:
        forecast_steps_max = int(forecast_steps)
        horizon_indices = list(range(forecast_steps_max))

    ## Align forecast at different steps for the same observation
    if align_forecast:
        phpp = PostHocProbabilisticProcessor()
        y_pred = phpp.forecast_horizon_aggregation(y_pred[:, :forecast_steps_max, ...], aggregation=None)
        y_true = phpp.forecast_horizon_aggregation(y_true[:, :forecast_steps_max, ...], aggregation=None)
    else:
        # Use the data as-is without alignment
        y_pred = y_pred[:, :forecast_steps_max, ...]
        y_true = y_true[:, :forecast_steps_max, ...]

    """
    2. Parameters
    """
    false_alarm = significance_level

    ### Window Size
    critical_efficient_size = int(
        int(np.ceil(1 / significance_level)) + forecast_steps_max
    )  # minimum size for desired significance level

    window_size = int(np.maximum(critical_efficient_size, y_pred.shape[0]))
    window_size = int(np.minimum(window_size, 5000))

    weighting_optim_params = {}
    weighting_optim_params["n_batch_update"] = n_batch_update
    weighting_optim_params["false_alarm"] = false_alarm
    weighting_optim_params["lr"] = lr
    weighting_optim_params["n_epochs"] = n_epochs
    weighting_optim_params["stride"] = 1
    weighting_optim_params["window_size"] = window_size
    weighting_optim_params["prior_past_weights_value"] = prior_past_weights_value

    beta_prior = None
    if nonconformity_score in ["error"]:
        beta_prior = (1.0, 1.0)

    ### Initialize AD_MODEL
    print("#Optimization parameters :: ")
    print(weighting_optim_params)

    outliers_scores = np.full(
        (y_pred.shape[0], y_pred.shape[1], y_pred.shape[2]),
        np.nan,
    )

    if return_weights:
        cal_weights = {}
        cal_scores = {}
    for ix_f in range(y_pred.shape[2]):
        for ix_h in horizon_indices:
            nonconformity_scores_values = nonconformity_score_functions(
                y_gt=y_true[:, ix_h, ix_f],
                y_pred=y_pred[:, ix_h, ix_f],
                nonconformity_score=nonconformity_score,
            )
            awcs = AdaptiveWeightedConformalScoreWrapper(
                false_alarm=weighting_optim_params["false_alarm"],
                window_size=weighting_optim_params["window_size"],
                weighting="uniform",
                weighting_params={
                    "n_batch_update": weighting_optim_params["n_batch_update"],
                    "conformal_weights_update": False,
                    "stride": weighting_optim_params["stride"],
                    "lr": weighting_optim_params["lr"],
                    "n_epochs": weighting_optim_params["n_epochs"],
                    "prior_past_weights_value": weighting_optim_params["prior_past_weights_value"],
                },
            )
            ini_i_cal = ix_h
            end_i_cal = ix_h + int(np.ceil(1 / significance_level))
            beta_cal = awcs.fit(nonconformity_scores_values[ini_i_cal:end_i_cal], beta_prior=beta_prior)
            if nonconformity_score in ["error"]:
                beta_cal = np.minimum(1, 2 * np.minimum(beta_cal, 1 - beta_cal))
            outliers_scores[ini_i_cal:end_i_cal, ix_h, ix_f] = beta_cal

            beta_all = awcs.predict(nonconformity_scores_values[end_i_cal:], beta_prior=beta_prior)
            if nonconformity_score in ["error"]:
                beta_all = np.minimum(1, 2 * np.minimum(beta_all, 1 - beta_all))
            outliers_scores[end_i_cal:, ix_h, ix_f] = beta_all

            if return_weights:
                cal_scores[ix_h, ix_f] = awcs.cal_scores
                cal_weights[ix_h, ix_f] = awcs.cal_weights

    outliers_scores[np.isnan(np.array(y_true))] = np.nan

    # print(outliers_scores.shape)
    # print(np.nanmean(outliers_scores))

    # Aggregate across forecast horizon dimension (axis=1)
    if aggregation_forecast_horizon is not None:
        if aggregation_forecast_horizon in aggregation_methods:
            # Use aggregation methods from dictionary (requires looping over features)
            scores = np.full((outliers_scores.shape[0], outliers_scores.shape[2]), np.nan)
            for ix_f in range(outliers_scores.shape[2]):
                # Extract scores for this feature across all observations and forecast horizons
                feature_scores = outliers_scores[:, :, ix_f]  # shape: (n_obs, n_horizons)
                scores[:, ix_f] = aggregation_methods[aggregation_forecast_horizon](feature_scores)
        else:
            raise ValueError(f"Unknown aggregation_forecast_horizon: {aggregation_forecast_horizon}")
    else:
        # No aggregation across forecast horizon, keep all dimensions
        scores = outliers_scores

    # Aggregate across features dimension (axis=1 after forecast horizon aggregation, or axis=2 if no prior aggregation)
    if aggregation_features is not None:
        if aggregation_features in aggregation_methods:
            # Determine the correct axis based on whether forecast horizon was aggregated
            if aggregation_forecast_horizon is not None:
                # scores shape: (n_obs, n_features)
                scores = aggregation_methods[aggregation_features](scores)
            else:
                # scores shape: (n_obs, n_horizons, n_features)
                # Need to aggregate over features (axis=2) for each horizon
                aggregated_scores = np.full((scores.shape[0], scores.shape[1]), np.nan)
                for ix_h in range(scores.shape[1]):
                    horizon_scores = scores[:, ix_h, :]  # shape: (n_obs, n_features)
                    aggregated_scores[:, ix_h] = aggregation_methods[aggregation_features](horizon_scores)
                scores = aggregated_scores
        else:
            raise ValueError(f"Unknown aggregation_features: {aggregation_features}")

    if return_weights:
        return scores, {"cal_scores": cal_scores, "cal_weights": cal_weights}
    else:
        return scores
