# Copyright contributors to the TSFM project
#
"""Utilities to support conformal forecasts"""

import enum
import json
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from scipy.stats import norm
from transformers.feature_extraction_utils import FeatureExtractionMixin, PreTrainedFeatureExtractor


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
"""
Post-Hoc Probabilistic Wrapper Classes
"""


class PosthocProbabilisticWrapperBase:
    def __init__(
        self,
        window_size: Optional[int] = None,
        quantiles: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        **kwargs,
    ):
        self.window_size = window_size
        self.quantiles = quantiles
        self.critical_size = 1

    def fit(self, y_cal_gt: np.ndarray, y_cal_pred: np.ndarray,  **kwargs):
        """Fit posthoc probabilistic wrapper.
        Input:
        y_cal_gt ground truth values: nsamples x forecast_horizon x number_features
        y_cal_pred model perdictions: nsamples x forecast_horizon x number_features
        """
        return self

    def predict(self, y_test_pred: np.ndarray, quantiles=[], **kwargs):
        """Predic posthoc probabilistic wrapper.
        Input:
        y_test_pred: nsamples x forecast_horizon x number_features
        Output:
        y_test_prob_pred: nsamples x forecast_horizon x number_features x len(quantiles)
        """
        pass


class PostHocGaussian(PosthocProbabilisticWrapperBase):
    def __init__(self, window_size=None, quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
        super().__init__(window_size=window_size, quantiles=quantiles)
        self.variance = None

    def fit(self, y_cal_gt: np.ndarray, y_cal_pred: np.ndarray ):
        """Fit posthoc probabilistic wrapper.
        Input:
        y_cal_gt ground truth values: nsamples x forecast_horizon x number_features
        y_cal_pred model perdictions: nsamples x forecast_horizon x number_features
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

        self.variance = np.sum((y_cal_gt[-window_size:] - y_cal_pred[-window_size:]) ** 2, axis=0) / (
            len(y_cal_pred[-window_size:]) - 1
        )  # dimension should be

    def predict(self, y_test_pred: np.ndarray, quantiles=[]):
        """Predict posthoc probabilistic wrapper.
        Input:
        y_test_pred: nsamples x forecast_horizon x number_features
        Output:
        y_test_prob_pred: nsamples x forecast_horizon x number_features x len(quantiles)
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


# class PostHocConformalWrapper(PosthocProbabilisticWrapperBase):
#     def __init__(
#         self,
#         window_size: Optional[int] = None,
#         quantiles: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
#         params: Dict[str, Any] = {},
#     ):
#         if "window_size" in params.keys():
#             window_size = params["window_size"]
#         super().__init__(window_size=window_size, quantiles=quantiles)
#         params["window_size"] = window_size
#         params["false_alarm"] = np.min(quantiles) * 2
#         if "nonconformity_score" in params.keys():
#             if params["nonconformity_score"] in ["error"]:
#                 params["false_alarm"] = np.min(quantiles)
#         self.critical_size = np.ceil(1 / params["false_alarm"])

#         self.model = WeightedConformalForecasterWrapper(**params)

#     def fit(self, y_cal_pred, y_cal_gt):
#         """Fit posthoc probabilistic wrapper.
#         Input:
#         y_cal_pred model perdictions: nsamples x forecast_horizon x number_features
#         y_cal_gt ground truth values: nsamples x forecast_horizon x number_features
#         """
#         if self.window_size is None:
#             window_size = y_cal_gt.shape[0]
#         else:
#             window_size = self.window_size
#         assert (
#             len(y_cal_pred.shape) == 3
#         ), " y_cal_pred should have 3 dimensions : nsamples x forecast_horizon x number_features"
#         assert (
#             len(y_cal_gt.shape) == 3
#         ), " y_cal_gt should have 3 dimensions : nsamples x forecast_horizon x number_features"

#         self.model.fit(
#             y_cal_pred=y_cal_pred,  ## ttm predicted values
#             y_cal_gt=y_cal_gt,
#         )  ## ttm corresponding gt values

#     def predict(self, y_test_pred, quantiles=[]):
#         """Predict posthoc probabilistic wrapper.
#         Input:
#         y_test_pred: nsamples x forecast_horizon x number_features
#         Output:
#         y_test_prob_pred: nsamples x forecast_horizon x number_features x len(quantiles)
#         """
#         if len(quantiles) == 0:
#             quantiles = self.quantiles

#         assert (
#             len(y_test_pred.shape) == 3
#         ), " y_test_pred should have 3 dimensions : nsamples x forecast_horizon x number_features"

#         y_test_prob_pred = np.zeros([y_test_pred.shape[0], y_test_pred.shape[1], y_test_pred.shape[2], len(quantiles)])
#         ix_q = 0
#         for q in quantiles:
#             if self.model.nonconformity_score in [
#                 NonconformityScores.ABSOLUTE_ERROR.value,
#                 NonconformityScores.ERROR.value,
#             ]:
#                 if q < 0.5:
#                     q_pi_error_rate = q * 2
#                     output_q = self.model.predict(y_test_pred, false_alarm=q_pi_error_rate)
#                     y_test_prob_pred[..., ix_q] = output_q["prediction_interval"]["y_low"]
#                 elif q > 0.5:
#                     q_pi_error_rate = (1 - q) * 2
#                     output_q = self.model.predict(y_test_pred, false_alarm=q_pi_error_rate)
#                     y_test_prob_pred[..., ix_q] = output_q["prediction_interval"]["y_high"]
#                 else:
#                     if self.model.nonconformity_score in ["error"]:
#                         q_pi_error_rate = 0.5
#                         output_q = self.model.predict(y_test_pred, false_alarm=q_pi_error_rate)
#                         y_test_prob_pred[..., ix_q] = output_q["prediction_interval"]["y_high"]
#                     else:
#                         y_test_prob_pred[..., ix_q] = y_test_pred

#             ix_q += 1

#         return y_test_prob_pred


class PostHocProbabilisticProcessor(FeatureExtractionMixin):
    """Entry point for posthoc probabilistic approaches

    We adopt HF FeatureExtractionMixin to create the processor and support serialization and deserialization

    Currently focusses on the conformal approach
    To do:
     - Integrate the Guassian approach above
     - Full serialization support
     - Better output, in dataframe format
    """

    def __init__(
        self,
        window_size: Optional[int] = None,
        quantiles: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        nonconformity_score: str = NonconformityScores.ABSOLUTE_ERROR.value,
        **kwargs,
    ):
        # if "window_size" in params.keys():
        #     window_size = params["window_size"]
        # super().__init__(window_size=window_size, quantiles=quantiles)
        # params["window_size"] = window_size
        # params["false_alarm"] = np.min(quantiles) * 2
        # if "nonconformity_score" in params.keys():
        #     if params["nonconformity_score"] in ["error"]:
        #         params["false_alarm"] = np.min(quantiles)

        self.window_size = window_size
        self.quantiles = quantiles
        self.false_alarm = (np.min(quantiles) * 2).item()
        self.nonconformity_score = nonconformity_score

        if self.nonconformity_score in [NonconformityScores.ERROR.value]:
            self.false_alarm = np.min(quantiles).item()

        self.critical_size = np.ceil(1 / self.false_alarm).item()

        if self.nonconformity_score not in [
            NonconformityScores.ABSOLUTE_ERROR.value,
            NonconformityScores.ERROR.value,
        ]:
            raise ValueError(f"Provided nonconformity_score {self.nonconformity_score } is not valid.")

        # WMG
        # check that these are the right parameters
        self.model = kwargs.pop("model", None)
        if self.model is None:
            self.model = WeightedConformalForecasterWrapper(
                window_size=window_size, false_alarm=self.false_alarm, nonconformity_score=nonconformity_score
            )

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
            feature_extractor_dict["model"] = WeightedConformalForecasterWrapper.from_dict(model)

        return super().from_dict(feature_extractor_dict, **kwargs)

    def train(self, y_cal_pred, y_cal_gt):
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

    def predict(self, y_test_pred: np.ndarray, quantiles: List[float] = []) -> np.ndarray:
        """Predict posthoc probabilistic wrapper.
        Input:
        y_test_pred: nsamples x forecast_horizon x number_features
        Output:
        y_test_prob_pred: nsamples x forecast_horizon x number_features x len(quantiles)
        """
        if len(quantiles) == 0:
            quantiles = self.quantiles

        assert (
            len(y_test_pred.shape) == 3
        ), " y_test_pred should have 3 dimensions : nsamples x forecast_horizon x number_features"

        y_test_prob_pred = np.zeros([y_test_pred.shape[0], y_test_pred.shape[1], y_test_pred.shape[2], len(quantiles)])
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
                    if self.model.nonconformity_score in ["error"]:
                        q_pi_error_rate = 0.5
                        output_q = self.model.predict(y_test_pred, false_alarm=q_pi_error_rate)
                        y_test_prob_pred[..., ix_q] = output_q["prediction_interval"]["y_high"]
                    else:
                        y_test_prob_pred[..., ix_q] = y_test_pred

            ix_q += 1

        return y_test_prob_pred


def absolute_error(y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    assert y.shape == y_pred.shape, (
        "Shapes of y and y_pred do not match: y.shape, y_pred.shape = " + str(y.shape) + ";" + str(y_pred.shape)
    )
    error = np.abs(y - y_pred)
    if len(error.shape) > 1:
        error = np.mean(error, axis=-1)
    return error


def error(y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
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
    assert (
        nonconformity_score in NonconformityScores
    ), "Selected nonconformity score is not supported, choose from {}".format(
        str([s.value for s in NonconformityScores])
    )
    if nonconformity_score == NonconformityScores.ABSOLUTE_ERROR.value:
        return absolute_error(y_gt, y_pred)
    if nonconformity_score == NonconformityScores.ERROR.value:
        return error(y_gt, y_pred)


def conformal_set(
    y_pred: np.ndarray, score_threshold: float, nonconformity_score: str = NonconformityScores.ABSOLUTE_ERROR.value
) -> Dict[str, np.ndarray]:
    if nonconformity_score == NonconformityScores.ABSOLUTE_ERROR.value:
        return {"y_low": y_pred - score_threshold, "y_high": y_pred + score_threshold}
    if nonconformity_score == NonconformityScores.ERROR.value:
        return {"y_low": score_threshold[0] + y_pred, "y_high": score_threshold[1] + y_pred}


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
        """Weighted Split Conformal Wrapper.
        Input:
        nonconformity_score: nonconformity score to be considered
        false_alarm: false alarm or error rate for the prediction intervals
        weighting: type of nonconformity score weights
        weighting_params: dictionary with weighting parameters if applicable
        threshold_function: type of nonconformity score threshold function
        window_size: maximum number of calibration (past values) nonconformity scores to be considered
        online_adaptive: flag indicating if the approach is adaptive/online
        online_size: integer indicating the stride between online updates
        """
        self.nonconformity_score = nonconformity_score
        assert (
            self.nonconformity_score in NonconformityScores
        ), "Selected nonconformity score is not supported, choose from {}".format(
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
            setattr(obj, attr_name, attr_value)

        return obj

    def fit(
        self,
        y_cal_gt: np.ndarray,
        y_cal_pred: np.ndarray,
        X_cal: Optional[np.ndarray] = None,
        cal_timestamps: Optional[np.ndarray] = None,
    ):
        if self.window_size is None:
            self.window_size = y_cal_pred.shape[0]
        self.cal_scores = nonconformity_score_functions(
            y_cal_gt, y_cal_pred,  X=X_cal, nonconformity_score=self.nonconformity_score
        )

        self.cal_scores = self.cal_scores[-self.window_size :]

        if X_cal is not None:
            self.cal_X = X_cal[-self.window_size :]

        if cal_timestamps is not None:
            self.cal_timestamps = cal_timestamps[-self.window_size :]

        critical_efficient_size = int(np.ceil(1 / self.false_alarm))

        # Certain Weighting Methods May Require Fitting
        if self.weighting in [Weighting.UNIFORM.value, Weighting.EXPONENTIAL_DECAY.value]: #["uniform", "exponential_decay"]:
            cal_weights = self.get_weights()
            self.weights.append(cal_weights)
            # self.weights.append(cal_weights[-self.cal_scores.shape[0]:])
            if self.threshold_function == ThresholdFunction.WEIGHTING.value: #  "weighting":
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
        false_alarm=None,
    ):
        if false_alarm is None:
            false_alarm = self.false_alarm

        if self.weighting in [Weighting.UNIFORM.value, Weighting.EXPONENTIAL_DECAY.value]:#["uniform", "exponential_decay"]:
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
        self, cal_weights, cal_scores=None, y_pred=None, X=None, timestamps=None, false_alarm=None
    ):
        """
        cal_weights: 1D or 2D numpy array. If 1D its size is num_calibration_scores = self.cal_scores.shape[0]. If 2D is (num_sample, num_calibration_scores)

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
        if self.threshold_function == ThresholdFunction.WEIGHTING.value: #"weighting":
            if self.nonconformity_score in PositiveNonconformityScores:
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

    def predict_batch(self, y_pred, y_gt=None, X=None, timestamps=None, false_alarm=None, update=None):
        """
        y_pred: n_samples x 1
        y_gt: n_samples x 1
        X: n_samples x n_features
        false_alarm: None or float in [0,1]
        update: boolean
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
                y_gt, y_pred,  X=X, nonconformity_score=self.nonconformity_score
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
                test_ad_scores.append(1 - ad_score)

            # Update
            if update:
                self.update(test_scores, X=X, timestamps=timestamps)
            output["outliers"] = test_outliers
            output["outliers_scores"] = np.array(test_ad_scores).flatten()

        output["prediction_interval"] = prediction_interval
        return output

    def predict(self, y_pred, y_gt=None, X=None, timestamps=None, false_alarm=None, update=None):
        """
        y_pred: n_samples x 1
        y_gt: n_samples x 1
        X: n_samples x n_features
        false_alarm: None or float in [0,1]
        update: boolean

        """

        if false_alarm is None:
            false_alarm = self.false_alarm

        if update is None:
            update = self.online

        n_samples = y_pred.shape[0]
        n_batches = np.ceil(n_samples / self.online_size)

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

    def predict_interval(self, y_pred, X=None, timestamps=None, false_alarm=None):
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

    def update(self, scores, X=None, timestamps=None):
        self.cal_scores = np.append(self.cal_scores, scores, axis=0)
        self.cal_scores = self.cal_scores[-self.window_size :]

        if timestamps is not None:
            self.cal_timestamps.extend(timestamps)
            self.cal_timestamps = self.cal_timestamps[-self.window_size :]
        if X is not None:
            self.cal_X = np.append(self.cal_X, X, axis=0)
            self.cal_X = self.cal_X[-self.window_size :]

        if self.weighting == Weighting.UNIFORM.value: #"uniform":
            cal_weights = self.get_weights()
            if self.threshold_function == ThresholdFunction.WEIGHTING.value:
                self.score_threshold = self.score_threshold_func(cal_weights, false_alarm=self.false_alarm)


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

    def fit(self, y_cal_gt, y_cal_pred,  X_cal=None, cal_timestamps=None):
        """
        y_cal_gt : ground truth values, size is num_samples x forecast_length x num_features
        y_cal_pred : tsfm forecasts, size is num_samples x forecast_length x num_features
        X_cal (optional): input covariates for input dependent conformal approaches size is num_samples x num_covariates
        cal_timestamps (optional): timestamps associated to the forecasted values, size is num_samples x forecast_length
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
                if self.weighting_params["optimization"] == WeightingOptimization.WASS1.value: #"wass1":
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
                    y_cal_gt[:, ix_h, ix_f], y_cal_pred[:, ix_h, ix_f],  X_cal=X_cal, cal_timestamps=cal_timestamps_i
                )
                if cal_weights is not None:
                    self.univariate_wrappers[ix_h, ix_f].weights.append(cal_weights)

    def update(self, y_pred, y_gt, X=None, timestamps=None):
        """
        y_pred : tsfm forecasts, size is num_samples x forecast_length x num_features
        y_gt (optional for anomaly detection) : ground truth values, size is num_samples x forecast_length x num_features
        X_cal (optional): input covariates for input dependent conformal approaches size is num_samples x num_covariates
        cal_timestamps (optional): timestamps associated to the forecasted values, size is num_samples x forecast_length
        """
        for ix_h in range(y_pred.shape[1]):
            for ix_f in range(y_pred.shape[2]):
                timestamps_i = None
                if timestamps is not None:
                    timestamps_i = timestamps[:, ix_h, ix_f]

                y_gt_i = y_gt[:, ix_h, ix_f]

                _ = self.univariate_wrappers[ix_h, ix_f].predict(
                    y_pred[:, ix_h, ix_f], y_gt=y_gt_i, X=X, timestamps=timestamps_i, update=True
                )

    def predict(self, y_pred, y_gt=None, X=None, timestamps=None, false_alarm=None, update=False):
        """
        y_pred : tsfm forecasts, size is num_samples x forecast_length x num_features
        y_gt (optional for anomaly detection) : ground truth values, size is num_samples x forecast_length x num_features
        X_cal (optional): input covariates for input dependent conformal approaches size is num_samples x num_covariates
        cal_timestamps (optional): timestamps associated to the forecasted values, size is num_samples x forecast_length
        """
        output = {}
        if self.nonconformity_score in NonconformityScores:  #'absolute_error':
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

                if self.nonconformity_score in NonconformityScores:  # 'absolute_error':
                    output["prediction_interval"]["y_low"][:, ix_h, ix_f] = output_i["prediction_interval"]["y_low"]
                    output["prediction_interval"]["y_high"][:, ix_h, ix_f] = output_i["prediction_interval"]["y_high"]

                if y_gt is not None:
                    output["outliers"][:, ix_h, ix_f] = output_i["outliers"]
                    output["outliers_scores"][:, ix_h, ix_f] = output_i["outliers_scores"]

        return output


def weighted_conformal_quantile(
    scores: np.ndarray,
    weights: np.ndarray,
    alpha: float = 0.05,
    conformal_correction: bool = False,
    max_score: float = np.inf,
) -> float:
    """
    Predicts the weighted conformal quantile.

    :param scores: scores (vector of length n)
    :param weights: Weights for each observation (vector of length n)
    :param alpha: Significance level for the quantile (default 0.05 for 95% confidence interval)
    :param conformal_correction: If we want to apply the conformal quantile estimation
    :param max_score: maximum value that the score can take
    :return: Weighted conformal quantile
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
    Predicts the weighted conformal p-value.

    :param scores: The calibration/previously observed scores
    :param score_observed: The observed test/new score
    :param weights: Weights for each previously observed score (vector of length n)
    :param conformal_correction: apply conformal correction
    :param max_score: maximum score
    :return: Weighted conformal p-value.
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
            self.weighting_params["conformal_weights_update"] = True

        self.weights_average = None
        self.weights_average_count = 0

        assert (
            self.weighting_params["n_batch_update"] > self.weights_critical_norm
        ), "Given the false alarm n_batch_update must be larger than {}".format(np.ceil(self.weights_critical_norm))

    def fit(self, scores: np.ndarray) -> np.ndarray:
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
    get weighted p-score (beta_t) for observed test scores given the observed scores

    test_scores: b sized array with test scores
    observed_scores: b x w sized array with previous observations
    weights: w sized array with time-weigting per sample. Weights should be constrained [0,1]

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


def get_w1_distance(test_scores, observed_scores, weights=None, conformal_weights=True):
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
    n = vector.shape[0]  # Ensure 1D tensor

    # Check if we are already on the simplex
    if torch.isclose(vector.sum(), torch.tensor(radius)) and torch.all(vector >= 0):
        return vector

    # Sort v in decreasing order
    v_sorted, _ = torch.sort(vector, descending=True)
    cum_vector = torch.cumsum(v_sorted, dim=0)

    # Find rho
    rho = torch.nonzero(
        v_sorted * torch.arange(1, n + 1, dtype=vector.dtype, device=vector.device) > (cum_vector - radius)
    )[-1]

    # Compute theta
    theta = (cum_vector[rho] - radius) / (rho + 1)

    # Compute projection
    w = torch.clamp(vector - theta, min=0)
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
