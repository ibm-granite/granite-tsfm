"""Service Handler for TSFM Models"""

import logging
from typing import Any, Dict, Optional

from tsfm_public import TimeSeriesPreprocessor

from .hf_service_handler import HuggingFaceHandler
from .inference_payloads import (
    ForecastingParameters,
)


LOGGER = logging.getLogger(__file__)


class TinyTimeMixerHandler(HuggingFaceHandler):
    def _get_config_kwargs(
        self,
        parameters: Optional[ForecastingParameters] = None,
        preprocessor: Optional[TimeSeriesPreprocessor] = None,
    ) -> Dict[str, Any]:
        config_kwargs = {
            "num_input_channels": preprocessor.num_input_channels,
            "prediction_filter_length": parameters.prediction_length,
        }
        return config_kwargs
