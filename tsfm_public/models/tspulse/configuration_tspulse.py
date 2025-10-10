# Copyright contributors to the TSFM project
#
"""TSPulse model configuration"""

from typing import List, Optional, Union

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)

TSPULSE_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class TSPulseConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`TSPulseModel`]. It is used to instantiate a
    TSPulse model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the TSPulse {} architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more detailed information.


    Args:
        context_length (`int`, *optional*, defaults to 64):
            The context/history length for the input sequence.
        patch_length (`int`, *optional*, defaults to 8):
            The patch length for the input sequence.
        num_input_channels (`int`, *optional*, defaults to 1):
            Number of input variates. For univariate series, set to 1.
        patch_stride (`int`, *optional*, defaults to 8):
            Stride value for patching. Equal to patch_length yields non-overlapping patches.
        d_model (`int`, *optional*, defaults to 16):
            Hidden feature dimension of the model.
        expansion_factor (`int`, *optional*, defaults to 2):
            Expansion ratio for MLP layers. Controls MLP capacity.
        num_layers (`int`, *optional*, defaults to 3):
            Number of backbone layers in the encoder.
        dropout (`float`, *optional*, defaults to 0.2):
            Dropout probability used across backbone layers.
        mode (`str`, *optional*, defaults to `"common_channel"`):
            Channel mixing mode. Choose from `"common_channel"` or `"mix_channel"`.
        gated_attn (`bool`, *optional*, defaults to `True`):
            Whether to use gated attention inside mixer blocks.
        norm_mlp (`str`, *optional*, defaults to `"LayerNorm"`):
            Normalization strategy inside MLP blocks. `"LayerNorm"` or `"BatchNorm"`.
        self_attn (`bool`, *optional*, defaults to `False`):
            Enable tiny self-attention over patches. Typically, not recommended.
        self_attn_heads (`int`, *optional*, defaults to 1):
            Number of heads for self-attention, used if `self_attn=True`.
        use_positional_encoding (`bool`, *optional*, defaults to `False`):
            Enable positional encodings for self-attention layers.
        positional_encoding_type (`str`, *optional*, defaults to `"sincos"`):
            Type of positional encoding: `"random"` or `"sincos"`.
        scaling (`str` or `bool`, *optional*, defaults to `"revin"`):
            Local scaling strategy. `"mean"`, `"std"`, `"revin"` or `None`. Recommended to always use `"revin"`
            for improved performance.
        loss (`str`, *optional*, defaults to `"mse"`):
            Loss function to apply, e.g., `"mse"`, `"mae", "cross_entropy". For classification task,
            set loss as `"cross_entropy"`.
        init_std (`float`, *optional*, defaults to 0.02):
            Std deviation for truncated normal weight initialization.
        post_init (`bool`, *optional*, defaults to `False`):
            Whether to use Huggingface-style custom initialization.
        norm_eps (`float`, *optional*, defaults to 1e-5):
            Epsilon for normalization layers for numerical stability.
        head_dropout (`float`, *optional*, defaults to 0.2):
            Dropout for head layers.
        reconstruction_type (`str`, *optional*, defaults to `"patchwise"`):
            Type of reconstruction - `"patchwise"` or `"full"`.
        decoder_num_layers (`int`, *optional*, defaults to 3):
            Number of decoder layers.
        decoder_d_model (`int`, *optional*, defaults to 16):
            Hidden size for decoder layers.
        decoder_mode (`str`, *optional*, defaults to `"common_channel"`):
            Channel mixing strategy for decoder.
        num_patches_layerwise_scale (`List[float]`, *optional*):
            List of patch scaling factors per layer in encoder. Recommended to set to None
        num_channels_layerwise_scale (`List[float]`, *optional*):
            List of channel scaling factors per layer in encoder. Recommended to set to None
        d_model_layerwise_scale (`List[float]`, *optional*):
            List of d_model scaling factors per layer in encoder.  Recommended to set to None
        decoder_num_patches_layerwise_scale (`List[float]`, *optional*):
            List of patch expansion per layer in decoder.  Recommended to set to None
        decoder_num_channels_layerwise_scale (`List[float]`, *optional*):
            List of channel expansion per layer in decoder.  Recommended to set to None
        decoder_d_model_layerwise_scale (`List[float]`, *optional*):
            List of d_model expansion per layer in decoder.  Recommended to set to None
        num_targets (`int`, *optional*, defaults to 3):
            Refers to the number of labels in a class for a classification task.
        output_range (`List[int]`, *optional*):
            Output range to set. Recommended to set to None
        head_aggregation (`str`, *optional*):
            Aggregation strategy for the classification head. Recommended to set to None.
        head_aggregation_dim (`str`, *optional*, defaults to `"patch"`):
            Aggregation dimension for the classification head. `"patch"` or `"channel"`.
        init_linear (`str`, *optional*, defaults to `"pytorch"`):
            Strategy for linear layer initialization. <todo>
        init_embed (`str`, *optional*, defaults to `"pytorch"`):
            Strategy for embedding layer initialization.
        mask_ratio (`float`, *optional*):
            Mask ratio during training.
        mask_type (`str`, *optional*, defaults to `"block"`):
            Type of masking strategy to apply. Supported options:

            - `"block"`: Applies patch-level masking (contiguous chunks).
            - `"user"`: Applies a user-defined mask passed via `past_observed_mask` in the forward call.
            - `"hybrid"`: Combines both point-level and patch-level masking.
            - `"var_hybrid"`: Similar to `"hybrid"`, but uses a variable mask ratio randomly sampled in the range `[0, mask_ratio]`.
            - `"random"`: Applies masking to randomly selected individual time-points (point masking).
        mask_block_length (`int`, *optional*, defaults to 8):
            Block length for masking if `mask_type` is `"block"`.
        loss_apply_mode (`str`, *optional*, defaults to `"mask"`):
            Specifies how the loss should be applied. Supported options:

            - `"mask"`: Apply loss only on the masked positions.
            - `"full"`: Apply loss across the entire output sequence.
            - `"mask_and_full"`: Apply both masked-point loss and full-sequence loss together.
        use_learnable_mask_token (`bool`, *optional*, defaults to `True`):
            Whether to use a learnable token to replace masked inputs.
        minimum_scale (`float`, *optional*, defaults to 1e-5):
            Minimum value used in scaling operations.
        categorical_vocab_size_list (`List[int]`, *optional*):
            Vocabulary sizes for categorical input features. Disabled currently.
        fuse_fft (`bool`, *optional*, defaults to `False`):
            Whether to use FFT-based feature fusion.
        fft_weight (`float`, *optional*, defaults to 1.0):
            Weight of FFT loss term.
        fft_prob_weight (`float`, *optional*, defaults to 1.0):
            Weight for probabilistic FFT loss term.
        fft_prob_length (`int`, *optional*):
            Length of signal to use in FFT probabilistic loss.
        fft_prob_mode (`str`, *optional*, defaults to `"plain"`):
            Mode of FFT probability loss. Allowed values `"plain"` and `"log"`
        enable_fft_prob_loss (`bool`, *optional*, defaults to `True`):
            Whether to enable FFT probability loss.
        fft_original_signal_loss_weight (`float`, *optional*, defaults to 1.0):
            Weight for original TS reconstruction signal loss from FFT branch.
        fft_time_add_forecasting_pt_loss (`bool`, *optional*, defaults to `False`):
            Whether to add forecasting point loss in time domain.
        fft_time_add_forecasting_pt_loss_weight (`float`, *optional*, defaults to 1.0):
            Weight for forecasting point loss.
        prediction_length (`int`, *optional*, defaults to 8):
            Number of future points to forecast, if forecast head is enabled.
        channel_consistent_masking (`bool`, *optional*, defaults to `True`):
            Whether masking is consistent across channels.
        hydra_class_head (`List[int]`, *optional*):
            Number of classes per task in Hydra head. Recommended to set to None
        hydra_class_attention (`bool`, *optional*, defaults to `True`):
            Whether to apply attention in classification hydra head.
        free_channel_flow (`bool`, *optional*, defaults to `True`):
            Enables free gradient flow in channel mixer by initializing based on `"channel_mix_init"` param
        patch_register_tokens (`int`, *optional*, defaults to 8):
            Number of patch register tokens to add.
        channel_register_tokens (`int`, *optional*):
            Number of channel register tokens. Recommended to set to None
        fft_time_consistent_masking (`bool`, *optional*, defaults to `True`):
            If True, masked time-series is used for FFT. If False, Masking is enabled seperately in FFT space.
            Recommended to set to True.
        fft_mask_ratio (`float`, *optional*):
            Masking ratio for FFT components.
        fft_mask_strategy (`str`, *optional*, defaults to `"magnitude"`):
            Strategy to mask FFT components - `"magnitude"` or `"random"`.
        fft_remove_component (`str`, *optional*, defaults to `"last"`):
            Which frequency component to remove in FFT to match to the shape requirements.
            Allowed values are `"last"` or `"dc"`. Recommended to set to `"last"`
        head_reduce_d_model (`int`, *optional*):
            Reduction dimension for classification head. if head_aggregation is None, set it to 1 or 2. Else, set it to None
        channel_virtual_expand_scale (`int`, *optional*, defaults to 1):
            Virtual expansion factor for channel mixing. Enable only in number of channels are very low (say 1 to 3)
        classification_mode (`str`, *optional*, defaults to `"full_embedding"`):
            Strategy to determine which part of the decoder embedding is used in the classification head. Supported options:

            - `"full_embedding"`: Use the entire embedding.
            - `"long_embedding"`: Use only the first `2N` patch embeddings (excluding register tokens), where `N` is the number of input patches.
            - `"short_embedding"`: Use only the register token embeddings.
            - `"fft_embedding"`: Use only the patch embeddings from positions `N+1` to `2N`.
            - `"time_embedding"`: Use only the patch embeddings from positions `0` to `N`.
        head_reduce_channels (`int`, *optional*):
            Channel reduction parameter in classification head. Recommended to set to None
        head_attention (`bool`, *optional*, defaults to `False`):
            Whether to use attention in final head layer. Recommended to set to False, as we enable it in decoder.
        gated_attention_activation (`str`, *optional*, defaults to `"softmax"`):
            Activation function for gated attention. `"softmax"` or `"sigmoid"`.
        head_gated_attention_activation (`str`, *optional*, defaults to `"softmax"`):
            Activation for gated attention in the head. `"softmax"` or `"sigmoid"`.
        register_mixer_layers (`int`, *optional*):
            Number of mixer layerss to use in the mini-decoder for fft-prob and forecast predictions. If None,
            no decoder used.
        reconstruction_loss_weight (`int`, *optional*, defaults to 1):
            Weight for the reconstruction loss.
        masked_reconstruction_loss_weight (`int`, *optional*, defaults to 1):
            Weight for the masked reconstruction loss.
        channel_mix_init (`str`, *optional*, defaults to `"identity"`):
            Initialization strategy for channel mixing - `"identity"` or `"zero"`. Recommended to set to identity
        batch_aware_masking (`bool`, *optional*, defaults to `False`):
            If enabled, some samples in batches are time masked while others are fft masked. Recommended to set to False
        disable_mask_in_classification_eval (`bool`, *optional*, defaults to `True`):
            Whether to disable masking during evaluation in classification.
        num_full_patches_for_hybrid_mask (`int`, *optional*, defaults to 2):
            Number of full patches to mask for `"hybrid"` mask type.
        full_patch_mask_percentage (`float`, *optional*, defaults to 0.2):
            Percentage of full patches to mask in `"var_hybrid"` mask type.
        revin_affine (`bool`, *optional*, defaults to `True`):
            Whether to use affine RevIN transformation.
        data_actual_context_length (`int`, *optional*):
            Original context length before padding or sampling. Recommended to set to None. Not allowed.
        fft_applied_on (`str`, *optional*, defaults to `"scaled_ts"`):
            FFT applied on scaled or raw time-series: `"raw_ts"` or `"scaled_ts"`. Recommended to set to scaled_ts

    """

    model_type = "tspulse"
    attribute_map = {
        "hidden_size": "d_model",
        "num_hidden_layers": "num_layers",
    }

    def __init__(
        self,
        # Time series specific configuration
        context_length: int = 64,
        patch_length: int = 8,
        num_input_channels: int = 1,
        patch_stride: int = 8,
        # General model configuration
        d_model: int = 16,
        expansion_factor: int = 2,
        num_layers: int = 3,
        dropout: float = 0.2,
        mode: str = "common_channel",
        gated_attn: bool = True,
        norm_mlp: str = "LayerNorm",
        self_attn: bool = False,
        self_attn_heads: int = 1,
        use_positional_encoding: bool = False,
        positional_encoding_type: str = "sincos",
        scaling: Optional[Union[str, bool]] = "revin",
        loss: str = "mse",
        init_std: float = 0.02,
        post_init: bool = False,
        norm_eps: float = 1e-5,
        head_dropout: float = 0.2,
        reconstruction_type: str = "patchwise",
        decoder_num_layers: int = 3,
        decoder_d_model: int = 16,
        decoder_mode: str = "common_channel",
        num_patches_layerwise_scale: Optional[List[float]] = None,
        num_channels_layerwise_scale: Optional[List[float]] = None,
        d_model_layerwise_scale: Optional[List[float]] = None,
        decoder_num_patches_layerwise_scale: Optional[List[float]] = None,
        decoder_num_channels_layerwise_scale: Optional[List[float]] = None,
        decoder_d_model_layerwise_scale: Optional[List[float]] = None,
        num_targets: int = 3,
        output_range: Optional[list] = None,
        head_aggregation: Optional[str] = None,
        head_aggregation_dim: str = "patch",
        init_linear: str = "pytorch",
        init_embed: str = "pytorch",
        mask_ratio: Optional[float] = None,
        mask_type: str = "block",  # block, point
        mask_block_length: int = 8,
        loss_apply_mode: str = "mask",
        use_learnable_mask_token: bool = True,
        minimum_scale: float = 1e-5,
        categorical_vocab_size_list: Optional[list] = None,
        fuse_fft: Optional[bool] = False,
        fft_weight: float = 1,
        fft_prob_weight: float = 1,
        fft_prob_length: Optional[int] = None,
        fft_prob_mode: str = "plain",
        enable_fft_prob_loss: bool = True,
        fft_original_signal_loss_weight: float = 1,
        fft_time_add_forecasting_pt_loss: bool = False,
        fft_time_add_forecasting_pt_loss_weight: float = 1,
        prediction_length: Optional[int] = 96,
        channel_consistent_masking: Optional[bool] = True,
        hydra_class_head: Optional[list] = None,
        hydra_class_attention: Optional[bool] = True,
        free_channel_flow: bool = True,
        patch_register_tokens: Optional[int] = 8,
        channel_register_tokens: Optional[int] = None,
        fft_time_consistent_masking: Optional[bool] = True,
        fft_mask_ratio: Optional[float] = None,
        fft_mask_strategy: Optional[str] = "magnitude",
        fft_remove_component: Optional[str] = "last",
        head_reduce_d_model: Optional[int] = None,
        channel_virtual_expand_scale: Optional[int] = 1,
        classification_mode: Optional[str] = "full_embedding",
        head_reduce_channels: Optional[int] = None,
        head_attention: Optional[bool] = False,
        gated_attention_activation: Optional[str] = "softmax",
        head_gated_attention_activation: str = "softmax",
        register_mixer_layers: Optional[int] = None,
        reconstruction_loss_weight: Optional[int] = 1,
        masked_reconstruction_loss_weight: Optional[int] = 1,
        channel_mix_init: Optional[str] = "identity",
        batch_aware_masking: Optional[bool] = False,
        disable_mask_in_classification_eval: Optional[bool] = True,
        num_full_patches_for_hybrid_mask: Optional[int] = 2,
        full_patch_mask_percentage: Optional[float] = 0.2,
        revin_affine: Optional[bool] = True,
        data_actual_context_length: Optional[int] = None,
        fft_applied_on: Optional[str] = "scaled_ts",  # "raw_ts",  # "scaled_ts",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.decoder_d_model = decoder_d_model
        self.patch_stride = patch_stride
        self.num_input_channels = num_input_channels
        self.context_length = context_length
        self.patch_length = patch_length
        self.expansion_factor = expansion_factor

        self.num_full_patches_for_hybrid_mask = num_full_patches_for_hybrid_mask
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.masked_reconstruction_loss_weight = masked_reconstruction_loss_weight
        self.num_layers = num_layers
        self.dropout = dropout
        self.mode = mode
        self.gated_attn = gated_attn
        self.norm_mlp = norm_mlp
        self.scaling = scaling
        self.head_dropout = head_dropout
        self.patch_last = True
        self.use_positional_encoding = use_positional_encoding
        self.positional_encoding_type = positional_encoding_type
        self.self_attn = self_attn
        self.self_attn_heads = self_attn_heads
        self.init_std = init_std
        self.post_init = post_init
        self.enable_fft_prob_loss = enable_fft_prob_loss
        self.loss = loss
        self.norm_eps = norm_eps
        self.min_allowed_range = 4
        self.decoder_num_layers = decoder_num_layers
        self.decoder_mode = decoder_mode
        self.reconstruction_type = reconstruction_type
        self.fft_prob_weight = fft_prob_weight
        self.gated_attention_activation = gated_attention_activation
        self.register_mixer_layers = register_mixer_layers
        self.fft_prob_mode = fft_prob_mode
        self.classification_mode = classification_mode
        self.head_gated_attention_activation = head_gated_attention_activation
        self.fft_remove_component = fft_remove_component
        self.use_learnable_mask_token = use_learnable_mask_token
        self.channel_consistent_masking = channel_consistent_masking
        self.fft_prob_length = fft_prob_length
        self.num_patches_layerwise_scale = num_patches_layerwise_scale
        self.num_channels_layerwise_scale = num_channels_layerwise_scale
        self.d_model_layerwise_scale = d_model_layerwise_scale
        self.channel_mix_init = channel_mix_init
        self.decoder_d_model_layerwise_scale = decoder_d_model_layerwise_scale
        self.decoder_num_channels_layerwise_scale = decoder_num_channels_layerwise_scale
        self.decoder_num_patches_layerwise_scale = decoder_num_patches_layerwise_scale
        self.num_targets = num_targets
        self.output_range = output_range
        self.head_aggregation = head_aggregation
        self.head_aggregation_dim = head_aggregation_dim

        self.data_actual_context_length = data_actual_context_length

        self.revin_affine = revin_affine

        self.hydra_class_attention = hydra_class_attention
        self.free_channel_flow = free_channel_flow
        self.head_attention = head_attention
        self.fft_original_signal_loss_weight = fft_original_signal_loss_weight

        # init
        self.init_linear = init_linear
        self.init_embed = init_embed
        self.head_reduce_d_model = head_reduce_d_model
        self.channel_virtual_expand_scale = channel_virtual_expand_scale

        self.init_processing = False

        self.disable_mask_in_classification_eval = disable_mask_in_classification_eval
        self.fuse_fft = fuse_fft
        self.fft_weight = fft_weight
        self.mask_ratio = mask_ratio
        self.mask_block_length = mask_block_length
        self.mask_type = mask_type
        self.loss_apply_mode = loss_apply_mode

        self.head_reduce_channels = head_reduce_channels

        self.minimum_scale = minimum_scale
        self.categorical_vocab_size_list = categorical_vocab_size_list

        self.patch_register_tokens = patch_register_tokens

        self.channel_register_tokens = channel_register_tokens

        self.full_patch_mask_percentage = full_patch_mask_percentage

        self.fft_time_add_forecasting_pt_loss = fft_time_add_forecasting_pt_loss
        self.fft_time_add_forecasting_pt_loss_weight = fft_time_add_forecasting_pt_loss_weight

        self.batch_aware_masking = batch_aware_masking

        self.fft_applied_on = fft_applied_on

        self.fft_time_consistent_masking = fft_time_consistent_masking
        self.hydra_class_head = hydra_class_head

        self.fft_mask_ratio = fft_mask_ratio
        self.fft_mask_strategy = fft_mask_strategy

        self.prediction_length = prediction_length

    def set_scale(self, scale_param, actual_param, base_param, no_levels):
        if self._check_one_or_none(getattr(self, scale_param)):
            setattr(self, scale_param, [1] * no_levels)
            setattr(self, actual_param, [getattr(self, base_param)] * no_levels)
        else:
            setattr(
                self,
                actual_param,
                [int(getattr(self, base_param) * float(i)) for i in getattr(self, scale_param)],
            )

        for i in getattr(self, actual_param):
            if i < self.min_allowed_range and i != getattr(self, base_param):
                raise ValueError(
                    "Too much compression beyond level %s for param: %s, %s"
                    % (self.min_allowed_range, scale_param, getattr(self, actual_param))
                )

    def _check_one_or_none(self, param):
        if param is None:
            return True
        elif all(element == 1 for element in param):
            return True
        return False

    def check_and_init_preprocessing(self, task="reconstruction"):
        self.init_processing = True
        # print("Number of channels:", self.num_input_channels)

        if self.data_actual_context_length is not None:
            raise Exception("Set data_actual_context_length to None. Variable length input is not allowed currently.")

        if self.channel_register_tokens is not None:
            raise Exception("channel_register_tokens is not allowed. Set channel_register_tokens to None")
        if self.fuse_fft and self.context_length % 2 != 0:
            raise Exception("context length needs to be divisible by 2 when fuse_fft is True")

        if self.patch_length != self.patch_stride:
            raise Exception("Overlapping patches are not allowed currently..")

        if self.mask_ratio and self.mask_ratio > 0:
            if self.mask_block_length != self.patch_length:
                logger.warning("Please be aware that Mask block length is set different from the patch length")

        # if self.total_embedding_size is None:
        #     raise Exception("total_embedding_size cannot be None")

        if self.patch_register_tokens is None:
            raise Exception("Patch register tokens cannot be None..")

        if not hasattr(self, "num_patches"):
            self.num_patches = (
                max(self.context_length, self.patch_length) - self.patch_length
            ) // self.patch_stride + 1

            if self.fuse_fft:
                self.num_patches = self.num_patches * 2

        if self.mode == "common_channel":
            if not self._check_one_or_none(self.num_channels_layerwise_scale) or not self._check_one_or_none(
                self.decoder_num_channels_layerwise_scale
            ):
                logger.warning("Channel Compression not allowed when mode is common_channel. Setting it not None")
                self.num_channels_layerwise_scale = None
                self.decoder_num_channels_layerwise_scale = None

        if task in ["classification"]:
            if not self._check_one_or_none(self.num_channels_layerwise_scale):
                logger.warning("Channel Compression is not allowed in encoder for classification. Setting it to None")
                self.num_channels_layerwise_scale = None

        self.set_scale(
            scale_param="num_patches_layerwise_scale",
            actual_param="num_patches_layerwise",
            base_param="num_patches",
            no_levels=self.num_layers,
        )

        self.set_scale(
            scale_param="num_channels_layerwise_scale",
            actual_param="num_channels_layerwise",
            base_param="num_input_channels",
            no_levels=self.num_layers,
        )

        self.set_scale(
            scale_param="d_model_layerwise_scale",
            actual_param="d_model_layerwise",
            base_param="d_model",
            no_levels=self.num_layers,
        )

        self.set_scale(
            scale_param="decoder_num_patches_layerwise_scale",
            actual_param="decoder_num_patches_layerwise",
            base_param="num_patches",
            no_levels=self.decoder_num_layers,
        )

        self.set_scale(
            scale_param="decoder_num_channels_layerwise_scale",
            actual_param="decoder_num_channels_layerwise",
            base_param="num_input_channels",
            no_levels=self.decoder_num_layers,
        )

        self.set_scale(
            scale_param="decoder_d_model_layerwise_scale",
            actual_param="decoder_d_model_layerwise",
            base_param="decoder_d_model",
            no_levels=self.decoder_num_layers,
        )
