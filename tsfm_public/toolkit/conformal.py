# Copyright contributors to the TSFM project
#
"""Utilities to support conformal forecasts"""

import enum
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy.stats import norm
from transformers.feature_extraction_utils import PreTrainedFeatureExtractor

from tsfm_public.toolkit.processor import BaseProcessor


LOGGER = logging.getLogger(__file__)


class NonconformityScores(enum.Enum):
    """`Enum` for the different kinds of nonconformity scores."""

    ABSOLUTE_ERROR = "absolute_error"
    ERROR = "error"


class PositiveNonconformityScores(enum.Enum):
    """`Enum` for the different kinds of positive nonconformity scores."""

    ABSOLUTE_ERROR = "absolute_error"


class ThresholdFunction(enum.Enum):
    """`Enum` for the different kinds of nonconformity score thresholding functions."""

    WEIGHTING = "weighting"


class Weighting(enum.Enum):
    """`Enum` for the different kinds of nonconformity score weighting approaches."""

    UNIFORM = "uniform"
    EXPONENTIAL_DECAY = "exponential_decay"


class WeightingOptimization(enum.Enum):
    """`Enum` for the different nonconformity score weighting optimization approaches."""

    WASS1 = "wass1"


class PostHocProbabilisticMethod(enum.Enum):
    CONFORMAL = "conformal"
    GAUSSIAN = "gaussian"


"""
Post-Hoc Probabilistic Wrapper Classes
"""


class PostHocProbabilisticProcessor(BaseProcessor):
    """Entry point for posthoc probabilistic approaches for forecasting

    We adopt HF FeatureExtractionMixin through the BaseProcessor class to create the processor and support
    serialization and deserialization

    """

    PROCESSOR_NAME = "conformal_config.json"

    def __init__(
        self,
        window_size: Optional[int] = None,
        quantiles: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        nonconformity_score: str = NonconformityScores.ABSOLUTE_ERROR.value,
        method: str = PostHocProbabilisticMethod.CONFORMAL.value,
        weighting: str = Weighting.UNIFORM.value,
        weighting_params: Dict[str, Any] = {},
        threshold_function: str = ThresholdFunction.WEIGHTING.value,
        aggregation: Union[str, int] = "median",
        aggregation_axis: Union[int, Tuple[int, ...]] = 1,
        **kwargs,
    ):
        """
        PostHoc Probabilistic Processor. Turns the point estimates of a multivariate forecast model into probabilistic
        forecasts (quantile estimate of the target variable)

        Args:
            window_size (int, optional): Maximum context window size for considering past residuals. If None
                (default), all previous provided observations are used.
            quantiles (List[float]): List of target quantiles to compute, with values in the open interval (0, 1).
            nonconformity_score (str, optional): Name of the nonconformity score to use, as defined in the
                `NonconformityScores` enum. Applicable only if the method is conformal.
            method (str): Name of the post-hoc probabilistic method to use, as defined in the
                `PostHocProbabilisticMethod` enum.
            weighting (str): Strategy for weighting nonconformity scores, as defined in the `Weighting` enum. Only
                applicable if method = "conformal".
            weighting_params (dict,optional): Parameters for the selected weighting strategy, if applicable. Only applicable
                if method = "conformal".
                Supported keys include:
                - 'optimization' (str, optional): Past nonconformity score weighting optimization method, as defined in the 'WeightingOptimization' enum.
                - 'decay_param' (float, optional): Exponential decay factor used when 'optimization' is set to 'exponential_decay'.
            threshold_function (str): Method for computing the threshold, as defined in the `ThresholdFunction` enum.
                Only applicable if method = "conformal".
            aggregation (str, int, or None, optional):
                Determines how to aggregate the outlier scores across dimensions for the method outlier_score.
                - If a string (e.g., 'mean', 'max', 'min', 'median'), applies the specified aggregation function over the axes provided in `aggregation_axis`.
                - If an integer, selects the outlier scores corresponding to that specific forecast horizon index.
                - If None, returns outlier scores for all dimensions independently without aggregation.
            aggregation_axis (tuple of int, optional) to consider in outlier_score:
                Specifies the axes over which to aggregate when `aggregation` is a string.
        """

        self.window_size = window_size
        self.quantiles = quantiles
        self.false_alarm = (np.min(quantiles) * 2).item()
        self.nonconformity_score = nonconformity_score
        self.method = method
        self.aggregation = aggregation
        self.aggregation_axis = aggregation_axis

        if self.nonconformity_score in [NonconformityScores.ERROR.value]:
            self.false_alarm = np.min(quantiles).item()

        self.critical_size = np.ceil(1 / self.false_alarm).item()

        if self.method not in [PostHocProbabilisticMethod.CONFORMAL.value, PostHocProbabilisticMethod.GAUSSIAN.value]:
            raise ValueError(f"Provided Post Hoc probabilistic method {self.method} is not valid.")

        if self.nonconformity_score not in [
            NonconformityScores.ABSOLUTE_ERROR.value,
            NonconformityScores.ERROR.value,
        ]:
            raise ValueError(f"Provided nonconformity_score {self.nonconformity_score } is not valid.")

        self.model = kwargs.pop("model", None)
        if self.model is None:
            if self.method == PostHocProbabilisticMethod.CONFORMAL.value:
                self.model = WeightedConformalForecasterWrapper(
                    window_size=self.window_size,
                    false_alarm=self.false_alarm,
                    nonconformity_score=self.nonconformity_score,
                    weighting=weighting,
                    weighting_params=weighting_params,
                    threshold_function=threshold_function,
                )
            elif self.method == PostHocProbabilisticMethod.GAUSSIAN.value:
                self.model = PostHocGaussian(window_size=self.window_size, quantiles=self.quantiles)
            else:
                raise ValueError(f"Invalid method provided {self.method}")

        kwargs["processor_class"] = self.__class__.__name__
        super().__init__(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this feature extractor instance.
        """
        output = super().to_dict()

        output["model"] = output["model"].to_dict()

        return output

    def to_json_string(self) -> str:
        """
        Serializes this instance to a JSON string.

        Returns:
            `str`: String containing all the attributes that make up this feature_extractor instance in JSON format.
        """
        dictionary = self.to_dict()

        def recursive_check_ndarray(dictionary):
            for key, value in dictionary.items():
                if key == "dtype":
                    # to do: ensure deserializable
                    dictionary[key] = value.__name__
                elif isinstance(value, np.ndarray):
                    dictionary[key] = value.tolist()
                elif isinstance(value, np.int64):
                    dictionary[key] = int(value)
                elif isinstance(value, list):
                    dictionary[key] = [vv.tolist() if isinstance(vv, np.ndarray) else vv for vv in value]
                elif isinstance(value, dict):
                    dictionary[key] = recursive_check_ndarray(value)
            return dictionary

        dictionary = recursive_check_ndarray(dictionary)

        # make sure private name "_processor_class" is correctly
        # saved as "processor_class"
        _processor_class = dictionary.pop("_processor_class", None)
        if _processor_class is not None:
            dictionary["processor_class"] = _processor_class

        return json.dumps(dictionary, indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_dict(cls, feature_extractor_dict: Dict[str, Any], **kwargs) -> "PreTrainedFeatureExtractor":
        """
        Instantiates a type of [`~feature_extraction_utils.FeatureExtractionMixin`] from a Python dictionary of
        parameters.

        Args:
            feature_extractor_dict (`Dict[str, Any]`):
                Dictionary that will be used to instantiate the feature extractor object. Such a dictionary can be
                retrieved from a pretrained checkpoint by leveraging the
                [`~feature_extraction_utils.FeatureExtractionMixin.to_dict`] method.
            kwargs (`Dict[str, Any]`):
                Additional parameters from which to initialize the feature extractor object.

        Returns:
            [`~feature_extraction_utils.FeatureExtractionMixin`]: The feature extractor object instantiated from those
            parameters.
        """
        model = feature_extractor_dict.get("model", None)
        if model is not None:
            method = feature_extractor_dict.get("method", None)
            if method == PostHocProbabilisticMethod.CONFORMAL.value:
                feature_extractor_dict["model"] = WeightedConformalForecasterWrapper.from_dict(model)
            elif method == PostHocProbabilisticMethod.GAUSSIAN.value:
                feature_extractor_dict["model"] = PostHocGaussian.from_dict(model)
            else:
                raise ValueError(f"Unknown method provided: {method}")

        return super().from_dict(feature_extractor_dict, **kwargs)

    def train(self, y_cal_pred: np.ndarray, y_cal_gt: np.ndarray) -> "PostHocProbabilisticProcessor":
        """Fit posthoc probabilistic wrapper.
        Input:
        y_cal_pred model perdictions: nsamples x forecast_horizon x number_features
        y_cal_gt ground truth values: nsamples x forecast_horizon x number_features

        """
        if len(y_cal_pred.shape) != 3:
            raise ValueError("y_cal_pred should have 3 dimensions: nsamples x forecast_horizon x number_features")

        if len(y_cal_gt.shape) != 3:
            raise ValueError("y_cal_gt should have 3 dimensions: nsamples x forecast_horizon x number_features")

        # WMG to do: check that updated window size is used in the fit call
        # (update) WMG: I don't think we need this
        # if self.window_size is None:
        #     window_size = y_cal_gt.shape[0]
        # else:
        #     window_size = self.window_size

        self.model.fit(
            y_cal_gt=y_cal_gt,
            y_cal_pred=y_cal_pred,  # ttm predicted values
            # ttm corresponding gt values
        )
        return self

    def predict(self, y_test_pred: np.ndarray, quantiles: List[float] = []) -> np.ndarray:
        """Predict posthoc probabilistic wrapper.
        Input:
        y_test_pred: nsamples x forecast_horizon x number_features
        Returns:
        y_test_prob_pred: nsamples x forecast_horizon x number_features x len(quantiles)
        """
        if len(quantiles) == 0:
            quantiles = self.quantiles

        assert (
            len(y_test_pred.shape) == 3
        ), " y_test_pred should have 3 dimensions : nsamples x forecast_horizon x number_features"

        y_test_prob_pred = np.zeros([y_test_pred.shape[0], y_test_pred.shape[1], y_test_pred.shape[2], len(quantiles)])
        if self.method == PostHocProbabilisticMethod.CONFORMAL.value:
            ix_q = 0
            for q in quantiles:
                if self.model.nonconformity_score in [
                    NonconformityScores.ABSOLUTE_ERROR.value,
                    NonconformityScores.ERROR.value,
                ]:
                    if q < 0.5:
                        q_pi_error_rate = q * 2
                        output_q = self.model.predict(y_test_pred, false_alarm=q_pi_error_rate)
                        y_test_prob_pred[..., ix_q] = output_q["prediction_interval"]["y_low"]
                    elif q > 0.5:
                        q_pi_error_rate = (1 - q) * 2
                        output_q = self.model.predict(y_test_pred, false_alarm=q_pi_error_rate)
                        y_test_prob_pred[..., ix_q] = output_q["prediction_interval"]["y_high"]
                    else:
                        if self.model.nonconformity_score in [NonconformityScores.ERROR.value]:
                            q_pi_error_rate = 0.5
                            output_q = self.model.predict(y_test_pred, false_alarm=q_pi_error_rate)
                            y_test_prob_pred[..., ix_q] = output_q["prediction_interval"]["y_high"]
                        else:
                            y_test_prob_pred[..., ix_q] = y_test_pred

                ix_q += 1
        if self.method == PostHocProbabilisticMethod.GAUSSIAN.value:
            y_test_prob_pred = self.model.predict(y_test_pred)

        return y_test_prob_pred

    def update(
        self, y_gt: np.ndarray, y_pred: np.ndarray, X: np.ndarray = None, timestamps: np.ndarray = None
    ) -> "PostHocProbabilisticProcessor":
        """
        Update the probabilistic post hoc model

        Args:
            y_gt (np.ndarray): Ground truth values. Shape: (n_samples, forecast_length, num_features).
            y_pred (np.ndarray): Predicted values. Shape: (n_samples,forecast_length, num_features).
            X (np.ndarray, optional): Input covariates for input-dependent methods. Shape: (n_samples, n_features).
            timestamps (np.ndarray, optional): Timestamps associated with each predicted value. Shape: (n_samples,).
        """
        if self.method == PostHocProbabilisticMethod.CONFORMAL.value:
            self.model.update(y_gt=y_gt, y_pred=y_pred, X=X, timestamps=timestamps)
        if self.method == PostHocProbabilisticMethod.GAUSSIAN.value:
            self.model.update(y_gt=y_gt, y_pred=y_pred)

        return self

    def outlier_score(
        self,
        y_gt: np.ndarray,
        y_pred: np.ndarray,
        X: np.ndarray = None,
        timestamps: np.ndarray = None,
        aggregation: Union[str, int] = -1,
        significance: float = 0.01,
        aggregation_axis: Union[int, Tuple[int, ...]] = -1,
        outlier_label: bool = True,
    ) -> np.ndarray:
        """
        PROTOTYPE: METHOD TO GET NORMALIZED OUTLIER CONFORMAL SCORE (P-VALUE) BASED ON FORECASTED PREDICTION ERRORS

        Args:
            y_gt (np.ndarray): Ground truth values. Shape: (n_samples, forecast_length, num_features).
            y_pred (np.ndarray): Predicted values. Shape: (n_samples,forecast_length, num_features).
            X (np.ndarray, optional): Input covariates for input-dependent methods. Shape: (n_samples, n_features).
            timestamps (np.ndarray, optional): Timestamps associated with each predicted value. Shape: (n_samples,).
            aggregation (str, int, or None, optional):
                Determines how to aggregate the outlier scores across dimensions.
                - If a string (e.g., 'mean', 'max', 'min', 'median'), applies the specified aggregation function over the axes provided in `aggregation_axis`.
                - If an integer, selects the outlier scores corresponding to that specific forecast horizon index.
                - If None, returns outlier scores for all dimensions independently without aggregation.
                - If -1 sets default aggregation in self.aggreggation attribute

            aggregation_axis (tuple of int, optional):
                Specifies the axes over which to aggregate when `aggregation` is a string.
        Returns:
            output_array (np.ndarray): The resulting array after aggregation. Its shape depends on the chosen aggregation strategy, but the last dimension is always 2, representing the outlier score (p-value) and the outlier label.
        """
        if self.method == PostHocProbabilisticMethod.CONFORMAL.value:
            if self.model.nonconformity_score in [
                NonconformityScores.ABSOLUTE_ERROR.value,
            ]:
                output = self.model.predict(
                    y_pred=y_pred, y_gt=y_gt, X=X, timestamps=timestamps, false_alarm=significance
                )
                outliers_scores = output["outliers_scores"]

                """
                Aggregation
                """
                if aggregation == -1:
                    aggregation = self.aggregation
                if aggregation_axis == -1:
                    aggregation_axis = self.aggregation_axis

                if aggregation is None:
                    outliers = output["outliers"]

                elif isinstance(aggregation, int):
                    outliers_scores = self.forecast_horizon_aggregation(outliers_scores, aggregation=aggregation)
                    # outliers_scores = outliers_scores[:, aggregation, :]
                    outliers = np.array(np.array(outliers_scores) <= significance).astype("int")

                elif isinstance(aggregation, str):
                    if aggregation_axis is None:
                        raise ValueError("aggregation_axis must be specified when aggregation is a string")

                    if isinstance(aggregation_axis, int):
                        aggregation_axis = (aggregation_axis,)

                    ### If aggregation across forecast horizon (axis = 1) was selected
                    if 1 in aggregation_axis:
                        aggregation_axis = tuple(x for x in aggregation_axis if x != 1)
                        outliers_scores = self.forecast_horizon_aggregation(outliers_scores, aggregation=aggregation)
                        if aggregation_axis:
                            outliers_scores = outliers_scores[:, np.newaxis, :]

                    ### If aggreagtion axis for other dimension != 1 were selected
                    if aggregation_axis:
                        if aggregation == "mean":
                            outliers_scores = np.mean(outliers_scores, axis=aggregation_axis)
                        elif aggregation == "median":
                            outliers_scores = np.median(outliers_scores, axis=aggregation_axis)
                        elif aggregation == "max":
                            outliers_scores = np.max(outliers_scores, axis=aggregation_axis)
                        elif aggregation == "min":
                            outliers_scores = np.min(outliers_scores, axis=aggregation_axis)
                        else:
                            raise ValueError(f"Unsupported aggregation method: {aggregation}")
                    outliers = np.array(np.array(outliers_scores) <= significance).astype("int")
                else:
                    raise TypeError("aggregation must be either an int or a supported aggregation string")

                if outlier_label:
                    return np.stack([outliers_scores, outliers], axis=-1)
                else:
                    return outliers_scores

    def forecast_horizon_aggregation(
        self, outliers_scores: np.ndarray, aggregation: Union[str, int] = "mean"
    ) -> np.ndarray:
        N, H, F = outliers_scores.shape
        # we want to align predictions/forecasts for each timestamp (observation)
        aligned = np.full(
            (N, H, F), np.nan
        )  # nan for padding initial items for which we have less than horizon H predictions.
        for h in range(H):
            # shift each row of forecast horizon h by h steps into the future.
            aligned[h:N, h, :] = outliers_scores[
                : N - h, h, :
            ]  # item i should have a values for [i,j,:] for j \in [0, min(i,H)]

        if isinstance(aggregation, int):  ## choose a particular forecast horizon score
            out = aligned[:, aggregation, :]
            for h in range(aggregation + 1):  # aligned in an upper triangular NaNs matrix
                out[h, :] = aligned[h, h, :]
            return out

        elif isinstance(aggregation, str):
            # aggregation but ignore nans
            if aggregation == "mean":
                return np.nanmean(aligned, axis=1)
            elif aggregation == "median":
                return np.nanmedian(aligned, axis=1)
            elif aggregation == "max":
                return np.nanmax(aligned, axis=1)
            elif aggregation == "min":
                return np.nanmin(aligned, axis=1)
            else:
                raise ValueError(f"Unsupported aggregation method: {aggregation}")
        else:
            raise TypeError("aggregation must be either an int or a supported aggregation string")


def absolute_error(y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute the absolute error between `y` and `y_pred`. If the inputs are multi-dimensional, the average is computed over the last dimension.

    Args:
        y (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        np.ndarray: Absolute error. If inputs are multi-dimensional, returns the mean absolute error over the last axis.
    """

    assert y.shape == y_pred.shape, (
        "Shapes of y and y_pred do not match: y.shape, y_pred.shape = " + str(y.shape) + ";" + str(y_pred.shape)
    )
    error = np.abs(y - y_pred)
    if len(error.shape) > 1:
        error = np.mean(error, axis=-1)
    return error


def error(y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute the difference between `y` and `y_pred` (signed difference y - y_pred). If the inputs are multi-dimensional, the average is computed over the last dimension.

    Args:
        y (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        np.ndarray: signed error. If inputs are multi-dimensional, returns the mean absolute error over the last axis.
    """
    assert y.shape == y_pred.shape, (
        "Shapes of y and y_pred do not match: y.shape, y_pred.shape = " + str(y.shape) + ";" + str(y_pred.shape)
    )
    error = y - y_pred
    if len(error.shape) > 1:
        error = np.mean(error, axis=-1)
    return error


def nonconformity_score_functions(
    y_gt: np.ndarray,
    y_pred: np.ndarray,
    X: Optional[np.ndarray] = None,
    nonconformity_score: str = NonconformityScores.ABSOLUTE_ERROR.value,
) -> np.ndarray:
    """
    Compute a predictive nonconformity score between ground truth and predictions with optional input covariates.

    Args:
        y_gt (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predicted values.
        X (np.ndarray, optional): Input covariates. Used by some nonconformity scoring methods.
        nonconformity_score (str, optional): Type of nonconformity score to compute, as defined in the `NonconformityScores` enum.

    Returns:
        np.ndarray: Computed nonconformity scores.
    """
    assert nonconformity_score in [
        s.value for s in NonconformityScores
    ], "Selected nonconformity score is not supported, choose from {}".format(
        str([s.value for s in NonconformityScores])
    )
    if nonconformity_score == NonconformityScores.ABSOLUTE_ERROR.value:
        return absolute_error(y_gt, y_pred)
    if nonconformity_score == NonconformityScores.ERROR.value:
        return error(y_gt, y_pred)


def conformal_set(
    y_pred: np.ndarray,
    score_threshold: Union[float, List[float]],
    nonconformity_score: str = NonconformityScores.ABSOLUTE_ERROR.value,
) -> Dict[str, np.ndarray]:
    """
    Compute the conformal prediction set or interval for a given prediction.

    Args:
        y_pred (np.ndarray): Predicted values.
        score_threshold (float or List[float]): Quantile(s) of the nonconformity scores used to determine the prediction intervals.
        nonconformity_score (str, optional): Type of nonconformity score to apply, as defined in the `NonconformityScores` enum.

    Returns:
        Dict[str, np.ndarray]: A dictionary containing the lower and upper bounds of the prediction interval or set. Keys are typically 'y_low' and 'y_high' for prediction intervals.
    """

    if nonconformity_score == NonconformityScores.ABSOLUTE_ERROR.value:
        return {"y_low": y_pred - score_threshold, "y_high": y_pred + score_threshold}
    if nonconformity_score == NonconformityScores.ERROR.value:
        return {"y_low": score_threshold[0] + y_pred, "y_high": score_threshold[1] + y_pred}


class PostHocGaussian:
    def __init__(self, window_size=None, quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
        self.window_size = window_size
        self.quantiles = quantiles
        self.critical_size = 1
        self.variance = None
        self.errors = None  # stores the errors, needed for the online updates

        """
        PostHoc Probabilistic Gaussian Wrapper.

        Transforms the point forecasts of a multivariate model into probabilistic forecasts under the assumption of independent Gaussian residuals.

        Args:
            window_size (int, optional): Maximum number of past residuals to consider when estimating variance. If None (default), all available past residuals are used.
            quantiles (List[float]): List of target quantiles to compute. Each value must lie in the open interval (0, 1).
        """

    def fit(self, y_cal_gt: np.ndarray, y_cal_pred: np.ndarray):
        """
        Fit the PostHoc Probabilistic Gaussian Wrapper.

        Args:
            y_cal_gt (np.ndarray): Ground truth values used for calibration.  Shape: (n_samples, forecast_horizon, num_features).
            y_cal_pred (np.ndarray): Model predictions corresponding to the ground truth. Shape: (n_samples, forecast_horizon, num_features).
        """

        if self.window_size is None:
            window_size = y_cal_gt.shape[0]
        else:
            window_size = self.window_size
        assert (
            len(y_cal_pred.shape) == 3
        ), " y_cal_pred should have 3 dimensions : nsamples x forecast_horizon x number_features"
        assert (
            len(y_cal_gt.shape) == 3
        ), " y_cal_gt should have 3 dimensions : nsamples x forecast_horizon x number_features"

        self.errors = y_cal_gt[-window_size:] - y_cal_pred[-window_size:]
        self.variance = np.sum(self.errors**2, axis=0) / (len(self.errors) - 1)  # dimension should be

    def update(self, y_gt: np.ndarray, y_pred: np.ndarray):
        """
        Update the PostHoc Probabilistic Gaussian Wrapper.

        Args:
            y_cal_gt (np.ndarray): Ground truth values used for calibration.  Shape: (n_samples, forecast_horizon, num_features).
            y_cal_pred (np.ndarray): Model predictions corresponding to the ground truth. Shape: (n_samples, forecast_horizon, num_features).
        """
        errors = y_gt - y_pred
        self.errors = np.concatenate([self.errors, errors], axis=0)
        self.errors = self.errors[-self.window_size :]
        self.variance = np.sum(self.errors**2, axis=0) / (len(self.errors) - 1)

    def predict(self, y_test_pred: np.ndarray, quantiles=[]):
        """
        Predict using the PostHoc Probabilistic Gaussian Wrapper.

        Args:
            y_test_pred (np.ndarray): Model point predictions. Shape: (n_samples, forecast_horizon, num_features).

        Returns:
            np.ndarray: Quantile estimates for each prediction. Shape: (n_samples, forecast_horizon, num_features, len(quantiles)).
        """

        if len(quantiles) == 0:
            quantiles = self.quantiles
        assert self.variance is not None, "Method needs to be fitted"
        assert (
            len(y_test_pred.shape) == 3
        ), " y_test_pred should have 3 dimensions: nsamples x forecast_horizon x number_features"
        std_devs = np.sqrt(self.variance)[..., None]  # Standard deviation for each distribution
        quantiles = norm.ppf(quantiles) * std_devs  # Compute quantiles
        y_test_prob_pred = y_test_pred[..., np.newaxis] + quantiles[np.newaxis, ...]
        return y_test_prob_pred

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this object.
        """

        output = {
            "window_size": self.window_size,
            "quantiles": self.quantiles,
            "critical_size": self.critical_size,
            "variance": self.variance,
        }

        return output

    @classmethod
    def from_dict(cls, params: Dict[str, Any], **kwargs) -> "PostHocGaussian":
        """
        Instantiates a type of [`~PostHocGaussian`] from a Python dictionary of
        parameters.

        Args:
            dict (`Dict[str, Any]`):
                Dictionary that will be used to instantiate the object.
            kwargs (`Dict[str, Any]`):
                Additional parameters from which to initialize the object.

        Returns:
            [`~PostHocGaussian`]: The PostHocGaussian object instantiated from those
            parameters.
        """

        variance = params.pop("variance", None)
        critical_size = params.pop("critical_size", 1)
        obj = cls(**params)
        obj.variance = variance
        obj.critical_size = critical_size
        return obj


class WeightedConformalForecasterWrapper:
    def __init__(
        self,
        nonconformity_score: str = NonconformityScores.ABSOLUTE_ERROR.value,
        false_alarm: float = 0.05,
        weighting: str = Weighting.UNIFORM.value,
        weighting_params: Dict[str, Any] = {},
        threshold_function: str = ThresholdFunction.WEIGHTING.value,
        window_size: Optional[int] = None,
    ):
        self.nonconformity_score = nonconformity_score
        # self.nonconformity_score_func = nonconformity_score_functions

        self.quantile = 1 - false_alarm
        self.false_alarm = false_alarm

        self.weighting = weighting
        self.weighting_params = weighting_params
        self.window_size = window_size
        self.online_size = 1
        self.threshold_function = threshold_function
        self.weights_adaptive = []
        self.univariate_wrappers = {}

        """
        Weighted Split Conformal Forecasting Wrapper Class.

        Args:
            nonconformity_score (str): Type of nonconformity score to use, as defined in the `NonconformityScores` enum.
            false_alarm (float): Desired false alarm (error) rate for the prediction intervals.
            weighting (str): Strategy for weighting nonconformity scores, as defined in the `Weighting` enum.
            weighting_params (dict): Parameters for the selected weighting strategy, if applicable.
            threshold_function (str): Method for computing the threshold, as defined in the `ThresholdFunction` enum.
            window_size (int, optional): Maximum number of past nonconformity scores to use for calibration. Default is None (use all available scores).
        """

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this object.
        """

        output = {
            "nonconformity_score": self.nonconformity_score,
            "false_alarm": self.false_alarm,
            "weighting": self.weighting,
            "weighting_params": self.weighting_params,
            "window_size": self.window_size,
            "threshold_function": self.threshold_function,
            "weights_adaptive": self.weights_adaptive,
            "univariate_wrappers": {json.dumps(k): v.to_dict() for k, v in self.univariate_wrappers.items()},
        }

        return output

    @classmethod
    def from_dict(cls, params: Dict[str, Any], **kwargs) -> "WeightedConformalForecasterWrapper":
        """
        Instantiates a type of [`~WeightedConformalForecasterWrapper`] from a Python dictionary of
        parameters.

        Args:
            dict (`Dict[str, Any]`):
                Dictionary that will be used to instantiate the object.
            kwargs (`Dict[str, Any]`):
                Additional parameters from which to initialize the object.

        Returns:
            [`~WeightedConformalForecasterWrapper`]: The WeightedcCnformalForecaster object instantiated from those
            parameters.
        """

        params_copy = params.copy()

        save_attrs = ["weights_adaptive", "univariate_wrappers"]
        copy_attrs = ["weights_adaptive"]

        save_attrs_dict = {}
        for attr in save_attrs:
            save_attrs_dict[attr] = params_copy.pop(attr)

        obj = cls(**params_copy, **kwargs)

        for attr_name, attr_value in save_attrs_dict.items():
            if attr_name in copy_attrs:
                setattr(obj, attr_name, attr_value)

        univariate_wrappers = {}
        for k, v in save_attrs_dict["univariate_wrappers"].items():
            k_decoded = tuple(json.loads(k))
            univariate_wrappers[k_decoded] = WeightedConformalWrapper.from_dict(v)

        obj.univariate_wrappers = univariate_wrappers

        return obj

    def fit(
        self, y_cal_gt: np.ndarray, y_cal_pred: np.ndarray, X_cal: np.ndarray = None, cal_timestamps: np.ndarray = None
    ):
        """
        Fit the Weighted Split Conformal Forecasting method.

        Args:
            y_cal_gt (np.ndarray): Ground truth values. Shape: (num_samples, forecast_length, num_features).
            y_cal_pred (np.ndarray): Forecasted values from the time series model. Shape: (num_samples, forecast_length, num_features).
            X_cal (np.ndarray, optional): Input covariates for input-dependent conformal methods. Shape: (num_samples, num_covariates).
            cal_timestamps (np.ndarray, optional): Timestamps associated with each forecasted value. Shape: (num_samples, forecast_length).
        """

        if self.window_size is None:
            self.window_size = y_cal_pred.shape[0]

        window_critical_size = int(np.ceil(1 / self.false_alarm))
        if (self.window_size < window_critical_size) and (y_cal_pred.shape[0] >= window_critical_size):
            self.window_size = window_critical_size
        assert (
            self.window_size >= window_critical_size
        ), "Not enough calibration points for the desired error rate. For an error rate of {} we need at least {} calibration points".format(
            self.false_alarm, window_critical_size
        )

        self.univariate_wrappers = {}
        for ix_f in range(y_cal_pred.shape[2]):
            cal_weights = None

            """
            Weights Optimization
            """
            # Copy the dictionary
            weighting_params = self.weighting_params.copy()
            if "optimization" in self.weighting_params.keys():
                if self.weighting_params["optimization"] == WeightingOptimization.WASS1.value:  # "wass1":
                    weighting_params.pop("optimization", None)
                    """
                    Fitting Weights
                    1. compute non-conformity scores.3
                    2. Initialize Wass1 weight adapter
                    """
                    critical_efficient_size = int(np.ceil(1 / self.false_alarm))
                    cal_scores = nonconformity_score_functions(
                        y_cal_gt[:, :, ix_f],
                        y_cal_pred[:, :, ix_f],
                        X=X_cal,
                        nonconformity_score=self.nonconformity_score,
                    )

                    stride = 1
                    n_batch_update = int((critical_efficient_size + 2))
                    n_updates = 100
                    n_cal_optimization = (n_updates - 1) * stride + n_batch_update
                    n_cal_init = np.maximum(int(critical_efficient_size) * 2, cal_scores.shape[0] - n_cal_optimization)
                    lr = 0.001

                    if cal_scores.shape[0] >= n_cal_init + n_batch_update:
                        awcsw = AdaptiveWeightedConformalScoreWrapper(
                            false_alarm=self.false_alarm,
                            window_size=self.window_size,
                            weighting=Weighting.UNIFORM.value,
                            weighting_params={
                                "n_batch_update": n_batch_update,
                                "conformal_weights_update": False,
                                "stride": stride,
                                "lr": lr,
                            },
                        )

                        # Need to call these because of side effects
                        _ = awcsw.fit(cal_scores[0 : int(n_cal_init)])  # betas_fit
                        _ = awcsw.predict(cal_scores[int(n_cal_init) :])  # betas_update
                        cal_weights = awcsw.cal_weights

            for ix_h in range(y_cal_pred.shape[1]):
                cal_timestamps_i = None
                if cal_timestamps is not None:
                    cal_timestamps_i = cal_timestamps[:, ix_h, ix_f]
                self.univariate_wrappers[ix_h, ix_f] = WeightedConformalWrapper(
                    nonconformity_score=self.nonconformity_score,
                    false_alarm=self.false_alarm,
                    weighting=self.weighting,
                    weighting_params=weighting_params,
                    threshold_function=self.threshold_function,
                    window_size=self.window_size,
                    online_size=self.online_size,
                )
                self.univariate_wrappers[ix_h, ix_f].fit(
                    y_cal_gt[:, ix_h, ix_f], y_cal_pred[:, ix_h, ix_f], X_cal=X_cal, cal_timestamps=cal_timestamps_i
                )
                if cal_weights is not None:
                    self.univariate_wrappers[ix_h, ix_f].weights.append(cal_weights)

    def update(self, y_gt: np.ndarray, y_pred: np.ndarray, X: np.ndarray = None, timestamps: np.ndarray = None):
        """
        Update the nonconformity scores and threshold function.

        Args:
            y_gt (np.ndarray): Ground truth values. Shape: (n_samples, forecast_length, num_features).
            y_pred (np.ndarray): Predicted values. Shape: (n_samples,forecast_length, num_features).
            X (np.ndarray, optional): Input covariates for input-dependent methods. Shape: (n_samples, n_features).
            timestamps (np.ndarray, optional): Timestamps associated with each predicted value. Shape: (n_samples,).
        """
        # print(y_pred.shape, y_gt.shape)
        for ix_h in range(y_pred.shape[1]):
            for ix_f in range(y_pred.shape[2]):
                timestamps_i = None
                if timestamps is not None:
                    timestamps_i = timestamps[:, ix_h, ix_f]

                y_gt_i = y_gt[:, ix_h, ix_f]

                _ = self.univariate_wrappers[ix_h, ix_f].predict(
                    y_pred[:, ix_h, ix_f], y_gt=y_gt_i, X=X, timestamps=timestamps_i, update=True
                )

    def predict(
        self,
        y_pred: np.ndarray,
        y_gt: np.ndarray = None,
        X: np.ndarray = None,
        timestamps: np.ndarray = None,
        false_alarm: float = None,
        update: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Generate prediction intervals and, if ground truth (`y_gt`) is provided, optionally return outlier flags and nonconformity p-values.

        If `update` is enabled, the method calls `predict_batch` with the `update=True` flag every `self.online_size` samples to update the nonconformity scores.

        Args:
            y_pred (np.ndarray): Predicted values. Shape: (n_samples,forecast_length, num_features).
            y_gt (np.ndarray, optional): Ground truth values. Shape: (n_samples, forecast_length, num_features).
            X (np.ndarray, optional): Input covariates. Shape: (n_samples, n_features).
            timestamps (np.ndarray, optional): Timestamps associated with each predicted value. Shape: (num_samples, forecast_length ).
            false_alarm (float, optional): Desired error rate in [0, 1]. If None, defaults to `self.false_alarm`.
            update (bool, optional): Whether to update the nonconformity scores using `y_gt` if provided. Default is False.

        Returns:
            dict: A dictionary containing:
                - prediction intervals for each forecast step and feature (always).
                - and optionally, outlier flags and nonconformity p-values if `y_gt` is provided.
        """

        output = {}
        if self.nonconformity_score in [s.value for s in NonconformityScores]:  #'absolute_error':
            output["prediction_interval"] = {"y_low": np.zeros_like(y_pred), "y_high": np.zeros_like(y_pred)}

        if y_gt is not None:
            output["outliers"] = np.zeros_like(y_pred)
            output["outliers_scores"] = np.zeros_like(y_pred)

        for ix_h in range(y_pred.shape[1]):
            for ix_f in range(y_pred.shape[2]):
                timestamps_i = None
                if timestamps is not None:
                    timestamps_i = timestamps[:, ix_h, ix_f]

                y_gt_i = None
                if y_gt is not None:
                    y_gt_i = y_gt[:, ix_h, ix_f]

                output_i = self.univariate_wrappers[ix_h, ix_f].predict(
                    y_pred[:, ix_h, ix_f],
                    y_gt=y_gt_i,
                    X=X,
                    timestamps=timestamps_i,
                    false_alarm=false_alarm,
                    update=update,
                )

                if self.nonconformity_score in [s.value for s in NonconformityScores]:  # 'absolute_error':
                    output["prediction_interval"]["y_low"][:, ix_h, ix_f] = output_i["prediction_interval"]["y_low"]
                    output["prediction_interval"]["y_high"][:, ix_h, ix_f] = output_i["prediction_interval"]["y_high"]

                if y_gt is not None:
                    output["outliers"][:, ix_h, ix_f] = output_i["outliers"]
                    output["outliers_scores"][:, ix_h, ix_f] = output_i["outliers_scores"]

        return output


class WeightedConformalWrapper:
    def __init__(
        self,
        nonconformity_score: str = NonconformityScores.ABSOLUTE_ERROR.value,
        false_alarm: float = 0.05,
        weighting: str = Weighting.UNIFORM.value,
        weighting_params: Dict[str, Any] = {},
        threshold_function: str = ThresholdFunction.WEIGHTING.value,
        window_size: Optional[int] = None,
        online_adaptive: bool = False,
        online_size: Optional[int] = 1,
    ):
        """
        Weighted Split Conformal Univariate Wrapper Class.

        Args:
            nonconformity_score (str): Type of nonconformity score to use, as defined in the `NonconformityScores` enum.
            false_alarm (float): Desired false alarm (error) rate for the prediction intervals.
            weighting (str): Weighting strategy for the nonconformity scores, as defined in the `Weighting` enum.
            weighting_params (dict): Dictionary of parameters for the weighting method, if applicable.
            threshold_function (str): Method used to compute the threshold, as defined in the `ThresholdFunction` enum.
            window_size (int, optional): Maximum number of past nonconformity scores to consider for calibration.
            online_adaptive (bool): Whether the method uses an online/adaptive update strategy.
            online_size (int, optional): Stride or update frequency for online adaptation.
        """
        self.nonconformity_score = nonconformity_score
        assert self.nonconformity_score in [
            s.value for s in NonconformityScores
        ], "Selected nonconformity score is not supported, choose from {}".format(
            str([s.value for s in NonconformityScores])
        )
        # self.nonconformity_score_func = nonconformity_score_functions

        self.quantile = 1 - false_alarm
        self.false_alarm = false_alarm

        self.weighting = weighting
        self.weighting_params = weighting_params
        self.window_size = window_size
        self.online_size = online_size
        self.online = online_adaptive

        self.threshold_function = threshold_function

        self.cal_scores = []
        self.weights = []
        self.cal_X = []
        self.cal_timestamps = []

        self.score_threshold = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this object.
        """
        output = {
            "nonconformity_score": self.nonconformity_score,
            "false_alarm": self.false_alarm,
            "weighting": self.weighting,
            "weighting_params": self.weighting_params,
            "window_size": self.window_size,
            "online_adaptive": self.online,
            "online_size": self.online_size,
            "threshold_function": self.threshold_function,
            "cal_scores": self.cal_scores,
            "weights": self.weights,
            "cal_X": self.cal_X,
            "cal_timestamps": self.cal_timestamps,
            "score_threshold": self.score_threshold,
        }

        return output

    @classmethod
    def from_dict(cls, params: Dict[str, Any], **kwargs) -> "WeightedConformalWrapper":
        """
        Instantiates a type of [`~WeightedConformalWrapper`] from a Python dictionary of
        parameters.

        Args:
            dict (`Dict[str, Any]`):
                Dictionary that will be used to instantiate the object.
            kwargs (`Dict[str, Any]`):
                Additional parameters from which to initialize the object.

        Returns:
            [`~WeightedConformalWrapper`]: The WeightedcCnformalWrapper object instantiated from those
            parameters.
        """

        params_copy = params.copy()

        save_attrs = ["cal_scores", "weights", "cal_X", "cal_timestamps", "score_threshold"]
        save_attrs_dict = {}
        for attr in save_attrs:
            save_attrs_dict[attr] = params_copy.pop(attr)

        obj = cls(**params_copy, **kwargs)

        for attr_name, attr_value in save_attrs_dict.items():
            if isinstance(attr_value, list):
                setattr(obj, attr_name, np.asarray(attr_value))
            else:
                setattr(obj, attr_name, attr_value)

        return obj

    def fit(
        self,
        y_cal_gt: np.ndarray,
        y_cal_pred: np.ndarray,
        X_cal: Optional[np.ndarray] = None,
        cal_timestamps: Optional[np.ndarray] = None,
    ):
        """
        Fit the Weighted Split Conformal method.

        Args:
            y_cal_gt (np.ndarray): Ground truth values. Shape: (num_samples).
            y_cal_pred (np.ndarray): Forecasted values from the time series model. Shape: (num_samples).
            X_cal (np.ndarray, optional): Input covariates used for input-dependent conformal methods. Shape: (num_samples, num_covariates).
            cal_timestamps (np.ndarray, optional): Timestamps corresponding to each forecasted value. Shape: (num_samples).
        """

        if self.window_size is None:
            self.window_size = y_cal_pred.shape[0]
        self.cal_scores = nonconformity_score_functions(
            y_cal_gt, y_cal_pred, X=X_cal, nonconformity_score=self.nonconformity_score
        )

        self.cal_scores = self.cal_scores[-self.window_size :]

        if X_cal is not None:
            self.cal_X = X_cal[-self.window_size :]

        if cal_timestamps is not None:
            self.cal_timestamps = cal_timestamps[-self.window_size :]

        critical_efficient_size = int(np.ceil(1 / self.false_alarm))

        # Certain Weighting Methods May Require Fitting
        if self.weighting in [
            Weighting.UNIFORM.value,
            Weighting.EXPONENTIAL_DECAY.value,
        ]:  # ["uniform", "exponential_decay"]:
            cal_weights = self.get_weights()
            self.weights.append(cal_weights)
            # self.weights.append(cal_weights[-self.cal_scores.shape[0]:])
            if self.threshold_function == ThresholdFunction.WEIGHTING.value:  #  "weighting":
                self.score_threshold = self.score_threshold_func(cal_weights, false_alarm=self.false_alarm)
        assert (
            np.sum(cal_weights) >= critical_efficient_size
        ), " The effective size is too small for the desired false alarm of {}, the calibration set should be larger than {} ".format(
            self.false_alarm, critical_efficient_size
        )

    def get_weights(
        self,
        y_pred: Optional[np.ndarray] = None,
        X: Optional[np.ndarray] = None,
        timestamps: Optional[np.ndarray] = None,
        false_alarm: Optional[float] = None,
    ) -> np.ndarray:
        """
        Return the nonconformity calibration weights.

        Args:
            y_pred (np.ndarray, optional): Predicted values. Shape: (num_samples,).
            X (np.ndarray, optional): Input covariates for input-dependent conformal methods. Shape: (num_samples, num_covariates).
            timestamps (np.ndarray, optional): Timestamps associated with each predicted value. Shape: (num_samples,).
            false_alarm (float, optional): Desired error rate (false alarm rate) for the prediction interval.

        Returns:
            np.ndarray: Array of calibration weights. Shape is either (window_size,) or (num_samples, window_size), depending on the weighting strategy.
        """

        if false_alarm is None:
            false_alarm = self.false_alarm

        if self.weighting in [
            Weighting.UNIFORM.value,
            Weighting.EXPONENTIAL_DECAY.value,
        ]:  # ["uniform", "exponential_decay"]:
            if len(self.weights) > 0:
                return self.weights[-1]
            else:
                if self.weighting == Weighting.UNIFORM.value:
                    # return np.ones(y_pred.shape[0],self.cal_scores.shape[0])
                    # return np.ones(self.cal_scores.shape[0])
                    return np.ones(self.window_size)

                if self.weighting == Weighting.EXPONENTIAL_DECAY.value:
                    decay_param = self.weighting_params.get("decay_param", 0.99)
                    return decay_param ** (self.window_size - np.arange(self.window_size))

    def score_threshold_func(
        self,
        cal_weights,
        cal_scores: Optional[np.ndarray] = None,
        y_pred: Optional[np.ndarray] = None,
        X: Optional[np.ndarray] = None,
        timestamps: Optional[np.ndarray] = None,
        false_alarm: Optional[float] = None,
    ) -> Union[float, List[float]]:
        """
        Compute the nonconformity score threshold corresponding to the desired false alarm (error) rate.

        Args:
            cal_weights (np.ndarray): Calibration weights. Can be either:
                - 1D array of shape (num_calibration_scores,)
                - 2D array of shape (num_samples, num_calibration_scores)
            cal_scores (np.ndarray, optional): Array of nonconformity scores used for threshold computation. If not provided, `self.cal_scores` is used.
            y_pred (np.ndarray, optional): Predicted values. Shape: (num_samples,).
            X (np.ndarray, optional): Input covariates for input-dependent thresholding. Shape: (num_samples, num_covariates).
            timestamps (np.ndarray, optional): Timestamps associated with each predicted value. Shape: (num_samples,).
            false_alarm (float, optional): Desired false alarm (error) rate for the prediction interval.

        Returns:
            Union[float, List[float]]: Threshold value(s) for the nonconformity scores, used to determine the prediction intervals.
        """

        if cal_scores is None:
            cal_scores = self.cal_scores

        if false_alarm is None:
            false_alarm = self.false_alarm

        score_threshold = []
        n_cal_scores = cal_scores.shape[0]
        assert n_cal_scores >= np.ceil(1 / false_alarm), " not enough calibration scores for error rate " + str(
            false_alarm
        )
        if self.threshold_function == ThresholdFunction.WEIGHTING.value:  # "weighting":
            if self.nonconformity_score in [s.value for s in PositiveNonconformityScores]:
                if len(cal_weights.shape) == 1:  # same weights for all y
                    score_threshold = weighted_conformal_quantile(
                        np.append(cal_scores, np.array([np.inf]), axis=0),
                        np.append(cal_weights[-n_cal_scores:], np.array([1]), axis=0),
                        alpha=false_alarm,
                    )
                else:
                    for i in range(cal_weights.shape[0]):
                        score_threshold_i = weighted_conformal_quantile(
                            np.append(cal_scores, np.array([np.inf]), axis=0),
                            np.append(cal_weights[i, -n_cal_scores:], np.array([1]), axis=0),
                            alpha=false_alarm,
                        )
                        score_threshold.append(score_threshold_i)
                    score_threshold = np.array(score_threshold)

            elif self.nonconformity_score == NonconformityScores.ERROR.value:
                if len(cal_weights.shape) == 1:  # same weights for all y
                    cal_scores_infty = np.append(cal_scores, np.array([np.inf, -np.inf]), axis=0)
                    # cal_scores_infty = np.append(np.array([]),cal_scores_infty,axis=0)

                    cal_weights_infty = np.append(cal_weights[-n_cal_scores:], np.array([1, 1]), axis=0)
                    # cal_weights_infty = np.append(np.array([1]),cal_weights_infty,axis=0)

                    score_threshold_low = weighted_conformal_quantile(
                        cal_scores_infty, cal_weights_infty, alpha=np.maximum(false_alarm / 2, 1 - (false_alarm / 2))
                    )

                    score_threshold_up = weighted_conformal_quantile(
                        cal_scores_infty, cal_weights_infty, alpha=np.minimum(false_alarm / 2, 1 - (false_alarm / 2))
                    )
                    assert (
                        score_threshold_up >= score_threshold_low
                    ), " score_threshold_up is not greater than score_threshold_low"

                    score_threshold = [score_threshold_low, score_threshold_up]

                else:
                    for i in range(cal_weights.shape[0]):
                        cal_scores_infty = np.append(cal_scores, np.array([np.inf, -np.inf]), axis=0)
                        # cal_scores_infty = np.append(np.array([-np.inf]),cal_scores_infty,axis=0)

                        cal_weights_infty = np.append(cal_weights[i, -n_cal_scores:], np.array([1, 1]), axis=0)
                        # cal_weights_infty = np.append(np.array([1]),cal_weights_infty,axis=0)

                        # score_threshold_i = weighted_conformal_quantile(np.append(cal_scores,np.array([np.inf]),axis=0),
                        #                     np.append(cal_weights[i,-n_cal_scores:],np.array([1]),axis=0),
                        #                     alpha=false_alarm)

                        score_threshold_low_i = weighted_conformal_quantile(
                            cal_scores_infty,
                            cal_weights_infty,
                            alpha=np.maximum(false_alarm / 2, 1 - (false_alarm / 2)),
                        )

                        score_threshold_up_i = weighted_conformal_quantile(
                            cal_scores_infty,
                            cal_weights_infty,
                            alpha=np.minimum(false_alarm / 2, 1 - (false_alarm / 2)),
                        )
                        score_threshold_i = [score_threshold_low_i, score_threshold_up_i]
                        assert (
                            score_threshold_up_i >= score_threshold_low_i
                        ), " score_threshold_up is not greater than score_threshold_low"

                        score_threshold.append(score_threshold_i)
                    score_threshold = np.array(score_threshold).transpose()  # first dimension is low/up score

            return score_threshold

    def predict_batch(
        self,
        y_pred: np.ndarray,
        y_gt: Optional[np.ndarray] = None,
        X: Optional[np.ndarray] = None,
        timestamps: Optional[np.ndarray] = None,
        false_alarm: Optional[float] = None,
        update: Optional[bool] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Generate prediction intervals for a batch of data and, if ground truth (`y_gt`) is provided, optionally return outliers and nonconformity p-values.

        Args:
            y_pred (np.ndarray): Predicted values. Shape: (n_samples, 1).
            y_gt (np.ndarray, optional): Ground truth values. Shape: (n_samples, 1).
            X (np.ndarray, optional): Input covariates. Shape: (n_samples, n_features).
            false_alarm (float, optional): Desired error rate in [0, 1]. If None, defaults to `self.false_alarm`.
            update (bool, optional): Whether to update the nonconformity scores using `y_gt` if provided. Default is False.

        Returns:
            dict: A dictionary containing:
                - prediction intervals (always),
                - and optionally, outlier flags and nonconformity p-values if `y_gt` is provided.
        """

        if false_alarm is None:
            false_alarm = self.false_alarm

        if update is None:
            update = self.online

        # Weight Computation
        cal_weights = self.get_weights(y_pred, X=X, timestamps=timestamps, false_alarm=false_alarm)

        # Score Threshold
        if (
            (false_alarm == self.false_alarm)
            and (self.weighting in [Weighting.UNIFORM.value])
            and (self.threshold_function in [ThresholdFunction.WEIGHTING.value])
        ):
            score_threshold = self.score_threshold
        else:
            score_threshold = self.score_threshold_func(
                cal_weights, y_pred=y_pred, X=X, timestamps=timestamps, false_alarm=false_alarm
            )

        # Prediction Interval
        prediction_interval = conformal_set(y_pred, score_threshold, nonconformity_score=self.nonconformity_score)

        output = {}
        if y_gt is not None:
            # Compute nonconformity scores of input
            test_scores = nonconformity_score_functions(
                y_gt, y_pred, X=X, nonconformity_score=self.nonconformity_score
            )

            # Outlier Flag and Outlier Scores
            test_outliers = np.array(test_scores > score_threshold).astype("int")
            test_ad_scores = []
            for score in test_scores:
                ad_score = weighted_conformal_alpha(
                    np.append(self.cal_scores, np.array([np.inf]), axis=0),
                    np.append(cal_weights, np.array([1]), axis=0),
                    score,
                )
                # test_ad_scores.append(1 - ad_score)
                test_ad_scores.append(ad_score)  # p-value (significance)

            # Update
            if update:
                self.update(test_scores, X=X, timestamps=timestamps)
            output["outliers"] = test_outliers
            output["outliers_scores"] = np.array(test_ad_scores).flatten()

        output["prediction_interval"] = prediction_interval
        return output

    def predict(
        self,
        y_pred,
        y_gt: Optional[np.ndarray] = None,
        X: Optional[np.ndarray] = None,
        timestamps: Optional[np.ndarray] = None,
        false_alarm: Optional[float] = None,
        update: Optional[bool] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Generate prediction intervals and, if ground truth (`y_gt`) is provided, optionally return outlier flags and nonconformity p-values.

        If `update` is enabled, the method calls `predict_batch` with the `update=True` flag every `self.online_size` samples to update the nonconformity scores.

        Args:
            y_pred (np.ndarray): Predicted values. Shape: (n_samples, 1).
            y_gt (np.ndarray, optional): Ground truth values. Shape: (n_samples, 1).
            X (np.ndarray, optional): Input covariates. Shape: (n_samples, n_features).
            false_alarm (float, optional): Desired error rate in [0, 1]. If None, defaults to `self.false_alarm`.
            timestamps (np.ndarray, optional): Timestamps associated with each predicted value. Shape: (num_samples,).
            update (bool, optional): Whether to update the nonconformity scores using `y_gt` if provided. Default is False.

        Returns:
            dict: A dictionary containing:
                - prediction intervals (always),
                - and optionally, outlier flags and nonconformity p-values if `y_gt` is provided.
        """

        if false_alarm is None:
            false_alarm = self.false_alarm

        if update is None:
            update = self.online

        n_samples = y_pred.shape[0]
        n_batches = int(np.ceil(n_samples / self.online_size))

        if (y_gt is not None) and update:
            """
            Batch Inference
            """
            output = None
            for ix_b in range(n_batches):
                ix_ini = int(ix_b * self.online_size)
                ix_end = np.minimum(int(ix_b * self.online_size + self.online_size), y_pred.shape[0])

                y_pred_b = y_pred[ix_ini:ix_end]
                y_gt_b = y_gt[ix_ini:ix_end]
                X_b = None
                if X is not None:
                    X_b = X[ix_ini:ix_end]
                timestamps_b = None
                if timestamps is not None:
                    timestamps_b = timestamps[ix_ini:ix_end]
                output_b = self.predict_batch(
                    y_pred_b, y_gt=y_gt_b, X=X_b, timestamps=timestamps_b, false_alarm=false_alarm, update=update
                )
                if output is None:
                    output = output_b.copy()
                else:
                    for k in output_b.keys():
                        if k == "prediction_interval":
                            for k2 in output_b[k].keys():
                                output[k][k2] = np.append(output[k][k2], np.array(output_b[k][k2]), axis=0)
                        else:
                            output[k] = np.append(output[k], np.array(output_b[k]), axis=0)
        else:
            output = self.predict_batch(
                y_pred, y_gt=y_gt, X=X, timestamps=timestamps, false_alarm=false_alarm, update=update
            )

        return output

    def predict_interval(
        self,
        y_pred: np.ndarray,
        X: Optional[np.ndarray] = None,
        timestamps: Optional[np.ndarray] = None,
        false_alarm: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Generate prediction intervals. (no update)

        Args:
            y_pred (np.ndarray): Predicted values. Shape: (n_samples, 1).
            X (np.ndarray, optional): Input covariates. Shape: (n_samples, n_features).
            timestamps (np.ndarray, optional): Timestamps associated with each predicted value. Shape: (num_samples,).
            false_alarm (float, optional): Desired error rate in [0, 1]. If None, defaults to `self.false_alarm`.

        Returns:
            dict: A dictionary containing prediction intervals
        """

        if false_alarm is None:
            false_alarm = self.false_alarm

        # Weight Computation
        cal_weights = self.get_weights(y_pred, X=X, timestamps=timestamps, false_alarm=false_alarm)

        # Score Threshold
        score_threshold = self.score_threshold_func(
            cal_weights, y_pred=y_pred, X=X, timestamps=timestamps, false_alarm=false_alarm
        )

        # Prediction Interval
        prediction_interval = conformal_set(y_pred, score_threshold, nonconformity_score=self.nonconformity_score)

        return prediction_interval

    def update(self, scores: np.ndarray, X: Optional[np.ndarray] = None, timestamps: Optional[np.ndarray] = None):
        """
        Update the nonconformity scores and threshold function.

        Args:
            scores (np.ndarray): Nonconformity scores. Shape: (n_samples, 1).
            X (np.ndarray, optional): Input covariates for input-dependent methods. Shape: (n_samples, n_features).
            timestamps (np.ndarray, optional): Timestamps associated with each predicted value. Shape: (n_samples,).
        """

        self.cal_scores = np.append(self.cal_scores, scores, axis=0)
        self.cal_scores = self.cal_scores[-self.window_size :]

        if timestamps is not None:
            self.cal_timestamps.extend(timestamps)
            self.cal_timestamps = self.cal_timestamps[-self.window_size :]
        if X is not None:
            self.cal_X = np.append(self.cal_X, X, axis=0)
            self.cal_X = self.cal_X[-self.window_size :]

        if self.weighting == Weighting.UNIFORM.value:  # "uniform":
            cal_weights = self.get_weights()
            if self.threshold_function == ThresholdFunction.WEIGHTING.value:
                self.score_threshold = self.score_threshold_func(cal_weights, false_alarm=self.false_alarm)


def weighted_conformal_quantile(
    scores: np.ndarray,
    weights: np.ndarray,
    alpha: float = 0.05,
    conformal_correction: bool = False,
    max_score: float = np.inf,
) -> float:
    """
    Predict the weighted conformal quantile.

    Args:
        scores (np.ndarray): Nonconformity scores. Shape: (n,).
        weights (np.ndarray): Weights corresponding to each score. Shape: (n,).
        alpha (float): Significance level for the quantile (e.g., 0.05 for a 95% prediction interval).
        conformal_correction (bool): Whether to apply conformal quantile correction.
        max_score (float): Maximum allowable value for the quantile.

    Returns:
        float: Estimated weighted conformal quantile.
    """

    if weights is None:
        weights = np.ones_like(scores)

    assert np.max(weights) <= 1, "Maximum weight needs to be smaller or equal than 1"
    assert np.min(weights) >= 0, "Minimum weight needs to be greater or equal to 0"
    assert weights.shape[0] == scores.shape[0], " Scores shape does not match weights shape"

    """Add infinite score to the score list"""
    if conformal_correction:
        weights = np.append(weights, np.array([1]))
        scores = np.append(scores, np.array([max_score]))

    # Ensure weights sum to 1
    weights = np.array(weights) / np.sum(weights)

    # Sort the residuals and corresponding weights
    sorted_indices = np.argsort(scores)
    sorted_scores = scores[sorted_indices]
    sorted_weights = weights[sorted_indices]

    # Compute cumulative sum of weights
    cumulative_weights = np.cumsum(sorted_weights)

    # Find the smallest residual where the cumulative weight exceeds 1 - alpha
    quantile_index = np.searchsorted(cumulative_weights, 1 - alpha)
    conformal_quantile = sorted_scores[quantile_index]

    return conformal_quantile


def weighted_conformal_alpha(
    scores: np.ndarray,
    weights: np.ndarray,
    score_observed: np.ndarray,
    conformal_correction: bool = False,
    max_score: float = np.inf,
) -> float:
    """
    Compute the weighted conformal p-value.

    Args:
        scores (np.ndarray): Calibration or previously observed nonconformity scores. Shape: (n,).
        score_observed (float): Nonconformity score of the test (new) observation.
        weights (np.ndarray): Weights for each calibration score. Shape: (n,).
        conformal_correction (bool): Whether to apply conformal correction (e.g., additive smoothing).
        max_score (float): Maximum allowable value for scores (used for clipping or normalization).

    Returns:
        float: Weighted conformal p-value.
    """

    if weights is None:
        weights = np.ones_like(score_observed)

    """Add infinite score to the score list"""
    if conformal_correction:
        weights = np.append(weights, np.array([1]))
        scores = np.append(scores, np.array([max_score]))

    weights = np.array(weights) / np.sum(weights)

    # Sort the residuals and corresponding weights
    sorted_indices = np.argsort(scores)
    sorted_scores = scores[sorted_indices]
    sorted_weights = weights[sorted_indices]

    smaller_scores = sorted_scores > score_observed

    return np.sum(sorted_weights[smaller_scores])


"""
Adaptive Weight Class
"""


class AdaptiveWeightedConformalScoreWrapper:
    def __init__(
        self,
        false_alarm: float = 0.05,
        window_size: Optional[int] = None,
        weighting: str = Weighting.UNIFORM.value,
        weighting_params: Dict[str, Any] = {},
    ):  # , threshold_function = 'weighting'):
        """
        Adaptive Weighted Conformal Score Class.

        Args:
            window_size (int, optional): Maximum number of past nonconformity scores to consider for calibration.
            false_alarm (float, optional): Desired false alarm (error) rate for the prediction intervals.
            weighting (str, optional): Weighting strategy for the nonconformity scores, as defined in the `Weighting` enum.
            weighting_params (dict, optional): Dictionary of parameters for the weighting method, if applicable.
                Supported keys include:
                - 'lr' (float, optional): Learning rate for weight optimization.
                - 'n_batch_update' (int, optional): Number of consecutive observations used to compute the empirical CDF for the Wasserstein-1 distance loss.
                - 'stride' (int, optional): Number of new observations between successive optimization updates.
                - 'epochs' (int, optional): Number of optimization steps performed on the same data batch.
                - 'conformal_weights_update' (bool, optional): Whether to include conformal correction when computing the Wasserstein-1 distance loss for optimization. (Recommended: False for most use cases.)
        """

        self.weighting = weighting
        self.weighting_params = weighting_params
        self.window_size = window_size
        self.weights_optimizer = None  # initializaed in fit
        self.losses = []

        self.cal_scores = []
        self.cal_weights = []

        self.false_alarm = false_alarm
        self.weights_critical_norm = np.ceil(1 / self.false_alarm - 1)
        if self.window_size is not None:
            assert (
                self.window_size > self.weights_critical_norm
            ), "Given the false alarm window size must be larger than {}".format(np.ceil(self.weights_critical_norm))

        if "lr" not in self.weighting_params.keys():
            self.weighting_params["lr"] = 0.001
        if "n_batch_update" not in self.weighting_params.keys():
            self.weighting_params["n_batch_update"] = int(self.weights_critical_norm * 4)
        if "stride" not in self.weighting_params.keys():
            self.weighting_params["stride"] = int(1)
        if "epochs" not in self.weighting_params.keys():
            self.weighting_params["epochs"] = int(1)
        if "conformal_weights_update" not in self.weighting_params.keys():
            self.weighting_params["conformal_weights_update"] = False

        self.weights_average = None
        self.weights_average_count = 0

        assert (
            self.weighting_params["n_batch_update"] > self.weights_critical_norm
        ), "Given the false alarm n_batch_update must be larger than {}".format(np.ceil(self.weights_critical_norm))

    def fit(self, scores: np.ndarray) -> np.ndarray:
        """
        Fit the model using calibration nonconformity scores.

        Args:
            scores (np.ndarray): Nonconformity calibration scores. Shape: (n_samples,).
        """

        """
        1. Add Calibration Scores
        """
        assert scores.shape[0] > self.weights_critical_norm, "Provide a minimum of {} scores for fitting".format(
            self.weights_critical_norm
        )
        self.cal_scores = torch.tensor(scores)
        if self.window_size is None:
            self.window_size = scores.shape[0]

        ## Initialize Weights + Optimizer
        w_init = torch.zeros(self.window_size)
        w_init[-scores.shape[0] :] = 1

        self.weights_parameters = torch.nn.Parameter(w_init)

        self.weights_optimizer = torch.optim.Adam([self.weights_parameters], lr=self.weighting_params["lr"])

        """
        2. Compute p-values for the given scores (should be ignored)
        """

        # Expand to nxn by repeating along the row dimension
        torch_scores_past = self.cal_scores.unsqueeze(0).repeat(scores.shape[0], 1)  # Shape: (n, n)

        with torch.no_grad():
            betas = (
                get_beta(
                    self.cal_scores,
                    torch_scores_past,
                    weights=self.weights_parameters[-scores.shape[0] :],
                    conformal_weights=True,
                )
                .clone()
                .detach()
                .numpy()
            )

        return betas

    def predict(self, scores: np.ndarray, verbose: bool = False) -> np.ndarray:
        """
        Perform online prediction of conformal p-values for observed scores.

        This method updates the weighting parameters in an online fashion based on `self.weighting` and `self.weights_parameters`.

        Args:
            scores (np.ndarray): Observed nonconformity scores. Shape: (n_samples,).
            verbose (bool, optional): If True, enables verbose output. Default is False.


        Returns:
            np.ndarray: Array of weighted p-scores (beta_t) for each test score. Shape: (n_samples,).
        """

        n_scores = scores.shape[0]
        n_batch = int(self.weighting_params["n_batch_update"])
        n_epochs = int(self.weighting_params["epochs"])
        stride = int(self.weighting_params["stride"])
        conformal_weights_update = self.weighting_params["conformal_weights_update"]

        n_batch_updates = (n_scores - n_batch) / stride + 1
        n_batch_updates = int(np.ceil(n_batch_updates))

        losses = []
        beta_output = []
        for i in range(n_batch_updates):
            ini_t_i = int(i * stride)
            end_t_i = int(i * stride + n_batch)
            if verbose:
                LOGGER.info("ini/end batch :", ini_t_i, end_t_i)
            scores_i = torch.tensor(scores[ini_t_i:end_t_i])

            """Generate Matrix With Past Scores per batch observation"""
            scores_concat_i = torch.cat((self.cal_scores, scores_i), dim=0)
            scores_matrix = [
                scores_concat_i[i - self.cal_scores.shape[0] : i]
                for i in range(self.cal_scores.shape[0], len(scores_concat_i))
            ]
            scores_matrix = torch.stack(scores_matrix)

            """Get Betas"""
            if self.weights_average is None:
                self.weights_average = self.weights_parameters.clone().detach()
                self.weights_average_count = 1

            with torch.no_grad():
                if (i < n_batch_updates - 1) & (i > 0):
                    ## Only inference for the stride items (betas should rely on weoghts updated with past observations)
                    betas = (
                        get_beta(
                            scores_i[-stride:],
                            scores_matrix[-stride:, :],
                            weights=self.weights_parameters[-scores_matrix.shape[1] :],
                            # weights=self.weights_average[-scores_matrix.shape[1]:]/self.weights_average_count,
                            conformal_weights=True,
                        )
                        .clone()
                        .detach()
                        .numpy()
                    )
                else:
                    betas = (
                        get_beta(
                            scores_i,
                            scores_matrix,
                            weights=self.weights_parameters[-scores_matrix.shape[1] :],
                            # weights=self.weights_average[-scores_matrix.shape[1]:]/self.weights_average_count,
                            conformal_weights=True,
                        )
                        .clone()
                        .detach()
                        .numpy()
                    )
                if verbose:
                    LOGGER.info("Beta shape :", betas.shape)
                beta_output.extend(betas.tolist())

            """Optimize Weights"""
            losses_i = []
            for _ in range(n_epochs):
                self.weights_optimizer.zero_grad()
                loss = get_w1_distance(
                    scores_i,
                    scores_matrix,
                    weights=self.weights_parameters[-scores_matrix.shape[1] :],
                    conformal_weights=conformal_weights_update,
                )
                losses_i.append(loss.item())

                loss.backward()
                self.weights_optimizer.step()

                ## Project weights into the simplex
                self.weights_parameters.data = project_l1_box_torch(
                    self.weights_parameters.data, min_l1_norm=self.weights_critical_norm, max_value=1.0
                )
                if verbose:
                    LOGGER.info(
                        "loss {}, w_norm_l1 {}, w_min {}, w_max {}".format(
                            loss.item(),
                            self.weights_parameters.data.sum(),
                            self.weights_parameters.data.min(),
                            self.weights_parameters.data.max(),
                        )
                    )

                self.weights_average = self.weights_average + self.weights_parameters.clone().detach()
                self.weights_average_count += 1
                # self.weights_parameters.data.clamp_(0, 1.0)
            losses.append(losses_i)

            """ Update Past Calibration Scores """
            self.cal_scores = torch.cat((self.cal_scores, scores_i[0:stride]), dim=0)
            self.cal_scores = self.cal_scores[-self.window_size :]
            # self.cal_weights = self.weights_average.clone().detach().numpy()/self.weights_average_count
            self.cal_weights = self.weights_parameters.clone().detach().numpy()

        self.losses.extend(losses)

        return np.array(beta_output)


"""
Optimization functions
"""


def get_beta(
    test_scores: torch.Tensor,
    observed_scores: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    conformal_weights: bool = True,
) -> torch.Tensor:
    """
    Compute the weighted conformal p-scores (beta_t) for a batch of test scores given observed calibration scores.

    Args:
        test_scores (np.ndarray): Array of test (observed) nonconformity scores. Shape: (b,).
        observed_scores (np.ndarray): Array of previously observed scores. Shape: (b, w), where each row contains the calibration scores for a given test point.
        weights (np.ndarray): Array of time-based weights applied to observed scores. Shape: (w,). All weights should lie in the range [0, 1].
        conformal_weights (bool, optional): Apply conformal quantile conservative correction.

    Returns:
        np.ndarray: Array of weighted p-scores (beta_t) for each test score. Shape: (b,).
    """

    b, w = observed_scores.shape

    assert (
        test_scores.ndim == 1 and test_scores.shape[0] == b
    ), "incorrect dimensions for test and observed scores, got {} and {}".format(
        test_scores.shape, observed_scores.shape
    )

    if weights is None:
        weights = torch.ones(w)

    assert (
        weights.ndim == 1 and weights.shape[0] == w
    ), "incorrect dimensions for weights and observed scores, got {} and {}".format(
        weights.shape, observed_scores.shape
    )
    assert (
        weights.max() <= 1 and weights.min() >= 0
    ), "weights should be constrained between 0 and 1, got {} and {}".format(weights.max(), weights.min())

    betas = (weights.unsqueeze(0) * (observed_scores <= test_scores.unsqueeze(1))).sum(1)

    weight_norm = weights.sum()
    if conformal_weights:
        weight_norm = weight_norm + 1
        betas = betas + 1
    betas = betas / weight_norm

    # betas = ((weights.unsqueeze(0) * (observed_scores<= test_scores.unsqueeze(1))).sum(1)+1)/weight_norm
    return betas


def w1_distance_from_betas(beta: torch.Tensor) -> torch.Tensor:
    """
    Compute the 1-Wasserstein distance between the empirical distribution
    given by sorted samples `beta` in [0,1] and the uniform distribution on [0,1].

    Args:
        beta: 1D tensor of shape (N,), sorted (beta[0] <= beta[1] <= ... <= beta[N-1])
              with values in [0,1].

    Returns:
        A scalar tensor representing W1 distance.
    """
    # Number of samples
    N = beta.shape[0]

    # For k = 1..N, define the subinterval [a_k, b_k] = [ (k-1)/N, k/N ]
    k = torch.arange(1, N + 1, dtype=beta.dtype, device=beta.device)
    a = (k - 1) / N  # left endpoints
    b = k / N  # right endpoints

    # ----- Piecewise closed-form integrals -----
    #
    # 1) If beta_k < a_k, then integral = ∫_{a_k}^{b_k} (p - beta_k) dp
    left_val = 0.5 * (b**2 - a**2) - beta * (b - a)

    # 2) If beta_k > b_k, then integral = ∫_{a_k}^{b_k} (beta_k - p) dp
    right_val = beta * (b - a) - 0.5 * (b**2 - a**2)

    # 3) If a_k <= beta_k <= b_k, split at beta_k:
    #    ∫ (|beta_k - p| dp) = (area left of beta_k) + (area right of beta_k)
    #    = beta^2 + 0.5(a^2 + b^2) - beta(a + b)
    middle_val = beta**2 + 0.5 * (a**2 + b**2) - beta * (a + b)

    # ----- Select which formula applies for each i -----
    mask_left = beta < a
    mask_right = beta > b

    val = torch.where(mask_left, left_val, torch.where(mask_right, right_val, middle_val))

    # Summation over i=1..N gives W1 distance
    return val.sum()


def get_w1_distance(test_scores, observed_scores, weights=None, conformal_weights=True) -> float:
    """
    Compute the 1-Wasserstein distance (W1) between the empirical distribution of weighted conformal p-values and the uniform distribution on [0, 1].

    The p-values are estimated using `test_scores` and `observed_scores`, weighted by `weights`.

    Args:
        test_scores (np.ndarray): Test nonconformity scores. Shape: (b,).
        observed_scores (np.ndarray): Previously observed calibration scores. Shape: (b, w), where each row corresponds to calibration scores for a test instance.
        weights (np.ndarray): Time-based weights applied to the calibration scores. Shape: (w,). Values must lie in the range [0, 1].
        conformal_weights (bool, optional): Whether to apply a conservative conformal correction when estimating p-values.

    Returns:
        float: Scalar representing the 1-Wasserstein distance between the empirical p-value distribution and the uniform distribution on [0, 1].
    """

    beta = get_beta(test_scores, observed_scores, weights, conformal_weights=conformal_weights)
    # Sort beta to enforce monotonicity
    beta_sorted = torch.sort(beta)[0]
    return w1_distance_from_betas(beta_sorted)


"""
Weight Projections
"""


def euclidean_proj_simplex_torch(vector: torch.Tensor, radius: int = 1) -> torch.Tensor:
    """Euclidean projection on a positive simplex.
    Algorithm 1 in
    Efficient Projections onto the .1-Ball for Learning in High Dimensions, Duchi et al.
    http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
    """
    assert radius > 0, "radius parameter must be strictly positive"
    n = vector.shape[0]
    # Check if vector is on the simplex
    if torch.isclose(vector.sum(), torch.tensor(radius)) and torch.all(vector >= 0):
        return vector
    v_sorted, _ = torch.sort(vector, descending=True)  # Sort vector in decreasing order
    cum_vector = torch.cumsum(v_sorted, dim=0)
    rho = torch.nonzero(
        v_sorted * torch.arange(1, n + 1, dtype=vector.dtype, device=vector.device) > (cum_vector - radius)
    )[-1]  # Find rho
    theta = (cum_vector[rho] - radius) / (rho + 1)  # Compute theta
    w = torch.clamp(vector - theta, min=0)  # Compute projection
    return w


def project_l1_box_torch(v, min_l1_norm: float = 1, max_value: float = 1) -> torch.Tensor:
    w = torch.clamp(v, min=0, max=max_value)
    if v.sum() > torch.tensor(min_l1_norm):
        return w
    else:
        condition = False
        n_iter = 0
        w_i = torch.clamp(v, max=max_value)  # clamp maximum values
        while not condition:
            w_proj = w_i[w_i < max_value]  # consider only entries that are allowed to grow
            l1_reduce = (
                min_l1_norm - torch.sum(w_i == max_value) * max_value
            )  # adjust l1 norm constrain based on norm of entries that are already in their max values
            if l1_reduce <= 0:
                return torch.clamp(w_i, min=0)

            w_proj = euclidean_proj_simplex_torch(w_proj, radius=l1_reduce)  # project
            w_i[w_i < max_value] = torch.clamp(w_proj, max=max_value, min=0)  # assign

            if torch.sum(w_i) >= torch.tensor(min_l1_norm):
                return w_i
            w_i[w_i == 0] = v[w_i == 0]  ## assign original values to items truncated to zero

            n_iter += 1
            condition = n_iter > 10
            LOGGER.info("iter : ", n_iter)

        w_i = torch.clamp(w_i, max=max_value, min=0)
        LOGGER.info(
            "proj ending due to max iterations, norm_l1 is {} and objective was >= {}".format(w_i.sum(), min_l1_norm)
        )
        return w_i
