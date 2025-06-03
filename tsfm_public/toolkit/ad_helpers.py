from abc import ABCMeta, abstractmethod
from typing import List, Union

import numpy as np
import torch
from transformers.utils.generic import ModelOutput


ScoreType = Union[np.ndarray, torch.Tensor]
ScoreListType = Union[ScoreType, List[ScoreType]]


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
        x: ScoreListType,
        **kwargs,
    ) -> np.ndarray:
        raise NotImplementedError
