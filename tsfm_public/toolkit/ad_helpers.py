from abc import ABCMeta, abstractmethod
from typing import List, Union

import numpy as np
import torch
from transformers.utils.generic import ModelOutput


ScoreType = Union[np.ndarray, torch.Tensor]
ScoreListType = Union[ScoreType, List[ScoreType]]


class TSADHelperUtility:
    """Abstract class for Anomaly detection pipeline.
    Model specific implementation. Implements three API calls required for integration to Anomaly Detection Pipeline.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def is_valid_mode(self, mode_str: str) -> bool:
        """Validates given Anomaly Prediction mode is supported by the model or not.

        Args:
            mode_str (str): prediction mode string

        Returns:
            bool: returns true if supported
        """
        raise NotImplementedError

    @abstractmethod
    def compute_score(
        self,
        payload: dict,
        **kwargs,
    ) -> ModelOutput:
        """Invokes model to generate output required for anomaly score computation.

        Args:
            payload (dict): data batch

        Returns:
            ModelOutput: return model outputs
        """
        raise NotImplementedError

    @abstractmethod
    def adjust_boundary(
        self,
        key: str,
        x: ScoreListType,
        **kwargs,
    ) -> np.ndarray:
        """API to adjust scores at the data boundary.

        Args:
            key (str): key associated with the model output
            x (ScoreListType): full model outputs

        Returns:
            np.ndarray: combined scores
        """
        raise NotImplementedError
