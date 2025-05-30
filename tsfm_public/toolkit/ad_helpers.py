from abc import ABCMeta, abstractmethod

import numpy as np
from transformers.utils.generic import ModelOutput


class TSADHelperUtility:
    __metaclass__ = ABCMeta

    @abstractmethod
    def is_valid_mode(self, mode_str: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def compute_score(
        self,
        payload: dict,
        **kwargs,
    ) -> ModelOutput:
        raise NotImplementedError

    @abstractmethod
    def adjust_boundary(
        self,
        key: str,
        x: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        raise NotImplementedError
