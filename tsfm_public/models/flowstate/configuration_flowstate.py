# Copyright contributors to the TSFM project
#
"""FlowState model configuration"""

from typing import List

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)

FLOWSTATE_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class FlowStateConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`FlowState`]. It is used to instantiate a
    FlowState model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the FlowState {} architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        context_length (`int`, *optional*, defaults to 2048)
            The context/history length of the input sequence.
        batch_first (`bool`):
            Indicates whether the `batch_size` or the `seq_length` is the first dimension of `past_values`.
        scale_factor (`float`):
            The scaling factor to adjust the parameter `Delta` of the S5 block and the Functional Basis Decoder
        prediction_length (`int`, *optional*):
            Number of time steps to forecast for a forecasting task. Also known as the Forecast Horizon.
            If not provided, or < 0, one forecasting patch is returned.

        embedding_feature_dim (`int`, *optional*, defaults to 512):
            Feature dimension of the linear input embedding. Recommended range is 128-512.

        encoder_num_layers (`int`, *optional*, defaults to 6):
            Number of encoder layers to use. I.e., number of S5 Layers in the FlowState encoder. Recommended range is 3-15. Larger value indicates more complex model.
        encoder_state_dim (`int`, *optional*, defaults to 512):
            State dimension of the S5 block. Recommended range is 128-1024. Larger value indicates more complex model.
        encoder_num_hippo_blocks (`int`, *optional*, defaults to 8):
            Number of HiPPo blocks to use for initialiaztion for the A matrixs of the S5 blocks. The `encoder_state_dim` needs to be divisibly by `encoder_num_hippo_blocks`.

        decoder_prediction_length (`int`, *optional*):
            Number of time steps to forecast for a forecasting task. Also known as the Forecast Horizon.
            If not provided, or < 0, one forecasting patch is returned.
        decoder_patch_len (`int`, *optional*, defaults to 24)
            The patch length used by the decoder when producing the forecasts.
        decoder_dim (`int`, *optional*, defaults to 256)
            Dimension of the produced forecast, e.g., number of expected output channels.
        decoder_type (`string`, *optional*, defaults to legS)
            The type of decoder used in the Functional Basis Decoder. The type of the decoder determines which basis functions are used.
            Possible choices are: ['legs', 'hlegs', 'four']

        quantiles (`list[float]`, *optional*, defaults to [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
            The quantiles used to compute the decoder output.

    Example:

    ```python
    >>> from transformers import FlowStateConfig, FlowStateModel

    >>> # Initializing a default FlowState configuration
    >>> configuration = FlowStateConfig()

    >>> # Randomly initializing a model (with random weights) from the configuration
    >>> model = FlowStateModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "flowstate"

    def __init__(
        self,
        # Time series specific configuration
        context_length: int = 2048,
        batch_first: bool = True,
        scale_factor: float = 1.0,
        prediction_length: int = -1,
        # Embedding specific configuration
        embedding_feature_dim: int = 512,
        # Encoder specific configuration
        encoder_num_layers: int = 6,
        encoder_state_dim: int = 512,
        encoder_num_hippo_blocks: int = 8,
        # Decoder specific configuration
        decoder_patch_len: int = 24,
        decoder_dim: int = 256,
        decoder_type: str = "legs",
        # Loss function / Prediction
        quantiles: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        prediction_type: str = "quantile",
        **kwargs,
    ):
        self.init_processing = False

        self.context_length = context_length
        self.batch_first = batch_first
        self.scale_factor = scale_factor
        self.prediction_length = prediction_length

        self.encoder_num_layers = encoder_num_layers
        self.encoder_state_dim = encoder_state_dim
        self.encoder_num_hippo_blocks = encoder_num_hippo_blocks
        self.embedding_feature_dim = embedding_feature_dim

        self.decoder_patch_len = decoder_patch_len
        self.decoder_dim = decoder_dim
        self.decoder_type = decoder_type

        self.quantiles = quantiles
        self.prediction_type = prediction_type

        super().__init__(**kwargs)

    def check_and_init_preprocessing(self):
        self.init_processing = True

        # Check general configuration for FlowStateModel and set defaults
        if not hasattr(self, "min_context"):
            self.min_context = self.context_length
        if not hasattr(self, "with_missing"):
            self.with_missing = True
        if self.context_length <= 0:
            raise ValueError("context_length should be positive")

        # Check embedding parameters
        if not hasattr(self, "embedding_feature_dim") or self.embedding_feature_dim <= 0:
            raise ValueError("embedding_feature_dim must be provided and positive")

        # Check encoder parameters
        if not hasattr(self, "encoder_num_layers") or self.encoder_num_layers <= 0:
            raise ValueError("encoder_num_layers must be provided and positive")
        if not hasattr(self, "encoder_state_dim") or self.encoder_state_dim <= 0:
            raise ValueError("encoder_state_dim must be provided and positive")
        if not hasattr(self, "encoder_num_hippo_blocks") or self.encoder_num_hippo_blocks <= 0:
            raise ValueError("encoder_num_hippo_blocks must be provided and positive")
        if self.encoder_state_dim % self.encoder_num_hippo_blocks != 0:
            raise ValueError("encoder_state_dim has to be divisible by encoder_num_hippo_blocks.")

        # Check decoder parameters
        if not hasattr(self, "decoder_patch_len") or self.decoder_patch_len <= 0:
            raise ValueError("decoder_patch_len  must be provided and positive.")
        if not hasattr(self, "decoder_dim") or self.decoder_dim <= 0:
            raise ValueError("decoder_dim must be provided and positive")
        if not hasattr(self, "decoder_type") or self.decoder_type not in ["legs", "hlegs", "four"]:
            raise ValueError("decoder_type must be provided and one of `['legs', 'hlegs', 'four']`")

        # Check loss paramter
        if not hasattr(self, "quantiles") or min(self.quantiles) < 0.0 or max(self.quantiles) > 1.0:
            raise ValueError("The values of quantiles must be provided and between [0, 1]")

        if not hasattr(self, "prediction_type") and self.prediction_type not in ["quantile", "mean", "median"]:
            raise ValueError("Unknown prediction_type detected. Should be one of ['quantile', 'mean', 'median']")
