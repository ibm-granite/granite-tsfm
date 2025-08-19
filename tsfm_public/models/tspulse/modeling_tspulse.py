# Copyright contributors to the TSFM project
#
"""
TSPulse model (Pytorch, HuggingFace-style)
TSPulse is ultra-compact pretrained-models that can be specialized for various tasks such as Anomaly Detection, Classification, Search and Imputation.
"""

import copy
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    ModelOutput,
    add_start_docstrings,
    logging,
    replace_return_docstrings,
)

from .configuration_tspulse import TSPulseConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "TSPulseConfig"


TSPULSE_PRETRAINED_MODEL_ARCHIVE_LIST = []


TSPULSE_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`TSPulseConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

TSPULSE_INPUTS_DOCSTRING = r"""
    Args:
        past_values (`torch.FloatTensor` of shape `(batch_size, seq_length, num_input_channels)`):
            Context values of the time series. This denotes the history/past time series values.
            For univariate time series, `num_input_channels` dimension should be 1. For multivariate time series, it is
            greater than 1.

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers.

        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class TSPulseGatedAttention(nn.Module):
    """
    Module that applies gated attention to input data.

    Args:
        in_size (`int`): The input size.
        out_size (`int`): The output size.
    """

    def __init__(self, in_size: int, out_size: int, attention_activation="softmax"):
        super().__init__()

        self.attn_layer = nn.Linear(in_size, out_size)
        if attention_activation == "softmax":
            self.attn_activation_layer = nn.Softmax(dim=-1)
        elif attention_activation == "sigmoid":
            self.attn_activation_layer = nn.Sigmoid()

    def _init_identity_weights(self):
        logger.info("Try identity init in Gated Attention.")
        nn.init.zeros_(self.attn_layer.weight)  # Zero weights to start with no influence
        nn.init.constant_(self.attn_layer.bias, 0)  # Bias to zero for neutral effect

    def forward(self, inputs):
        attn_weight = self.attn_activation_layer(self.attn_layer(inputs))
        inputs = inputs * attn_weight
        return inputs


class TSPulseBatchNorm(nn.Module):
    """
    Compute batch normalization over the sequence length (time) dimension.
    """

    def __init__(self, config: TSPulseConfig):
        super().__init__()
        self.batchnorm = nn.BatchNorm1d(config.d_model, eps=config.norm_eps)

    def forward(self, inputs: torch.Tensor):
        """
        Parameters:
            inputs (`torch.Tensor` of shape `(batch_size, sequence_length, d_model)`):
                input for Batch norm calculation
        Returns:
            `torch.Tensor` of shape `(batch_size, sequence_length, d_model)`
        """
        output = inputs.transpose(1, 2)  # output: (batch_size, d_model, sequence_length)
        output = self.batchnorm(output)
        return output.transpose(1, 2)


class TSPulsePositionalEncoding(nn.Module):
    """
    Class for positional encoding
    """

    def __init__(self, config: TSPulseConfig):
        super().__init__()
        # positional encoding: [num_patches x d_model]
        if config.use_positional_encoding:
            self.position_enc = self._init_pe(config)
        else:
            self.position_enc = nn.Parameter(torch.zeros(config.num_patches, config.d_model))

    @staticmethod
    def _init_pe(config: TSPulseConfig) -> nn.Parameter:
        # Positional encoding
        if config.positional_encoding_type == "random":
            position_enc = nn.Parameter(torch.randn(config.num_patches, config.d_model), requires_grad=True)
        elif config.positional_encoding_type == "sincos":
            position_enc = torch.zeros(config.num_patches, config.d_model)
            position = torch.arange(0, config.num_patches).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, config.d_model, 2) * -(math.log(10000.0) / config.d_model))
            position_enc[:, 0::2] = torch.sin(position * div_term)
            position_enc[:, 1::2] = torch.cos(position * div_term)
            position_enc = position_enc - position_enc.mean()
            position_enc = position_enc / (position_enc.std() * 10)
            position_enc = nn.Parameter(position_enc, requires_grad=False)
        else:
            raise ValueError(
                f"{config.positional_encoding_type} is not a valid positional encoder. Available types are 'random' and 'sincos'."
            )
        return position_enc

    def forward(self, patch_input: torch.Tensor):
        # hidden_state: [bs x num_channels x num_patches x d_model]
        hidden_state = patch_input + self.position_enc
        return hidden_state


class TSPulseNormLayer(nn.Module):
    """Normalization block

    Args:
        config (`TSPulseConfig`, *required*):
            Configuration.
    """

    def __init__(self, config: TSPulseConfig):
        super().__init__()

        self.norm_mlp = config.norm_mlp

        if "batch" in config.norm_mlp.lower():
            self.norm = TSPulseBatchNorm(config)
        else:
            self.norm = nn.LayerNorm(config.d_model, eps=config.norm_eps)

    def forward(self, inputs: torch.Tensor):
        """
        Args:
            inputs (`torch.Tensor` of shape `((batch_size, num_channels, num_patches, d_model))`):
                Input to the normalization layer.
        Returns:
            `torch.Tensor` of shape `((batch_size, num_channels, num_patches, d_model))`
        """
        if "batch" in self.norm_mlp.lower():
            # reshape the data
            inputs_reshaped = torch.reshape(
                inputs,
                (
                    inputs.shape[0] * inputs.shape[1],
                    inputs.shape[2],
                    inputs.shape[3],
                ),
            )  # inputs_reshaped: [batch_size*num_channels, num_patches, d_model]

            # inputs_reshaped: [batch_size*num_channels, num_patches, d_model]
            inputs_reshaped = self.norm(inputs_reshaped)

            # put back data to the original shape
            inputs = torch.reshape(inputs_reshaped, inputs.shape)

        else:
            inputs = self.norm(inputs)

        return inputs


class TSPulseMLP(nn.Module):
    def __init__(self, in_features, out_features, config):
        super().__init__()
        num_hidden = in_features * config.expansion_factor
        self.fc1 = nn.Linear(in_features, num_hidden)
        self.dropout1 = nn.Dropout(config.dropout)
        self.fc2 = nn.Linear(num_hidden, out_features)
        self.dropout2 = nn.Dropout(config.dropout)

        self.in_features = in_features
        self.out_features = out_features
        self.num_hidden = num_hidden

        self.config = config

    def _init_identity_weights(self):
        # recursive_init_identity_modules(self)

        if self.config.channel_mix_init == "zero":
            nn.init.zeros_(self.fc1.weight)
            nn.init.zeros_(self.fc1.bias)
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)
        else:
            # print("Try identity init in TSPulseMLP")
            if self.in_features == self.num_hidden:
                nn.init.eye_(self.fc1.weight)  # Identity matrix for weights
            else:
                nn.init.kaiming_uniform_(self.fc1.weight)  # Use a reasonable default for non-square matrices
            nn.init.zeros_(self.fc1.bias)  # Zero biases

            # Initialize fc2 to be identity (or as close as possible)
            if self.num_hidden == self.out_features:
                nn.init.eye_(self.fc2.weight)
            else:
                nn.init.kaiming_uniform_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

            self.fc1.weight.data *= 0.5
            self.fc2.weight.data *= 0.5

    def forward(self, inputs: torch.Tensor):
        """
        Args:
            inputs (`torch.Tensor` of shape `((batch_size, num_channels, num_patches, d_model))`):
                Input to the MLP layer.
        Returns:
            `torch.Tensor` of the same shape as `inputs`
        """
        inputs = self.dropout1(nn.functional.gelu(self.fc1(inputs)))
        inputs = self.fc2(inputs)
        inputs = self.dropout2(inputs)
        return inputs


class TSPulseChannelFeatureMixerBlock(nn.Module):
    """This module mixes the features in the channel dimension.

    Args:
        config (`TSPulseConfig`, *required*):
            Configuration.
    """

    def __init__(self, config: TSPulseConfig):
        super().__init__()

        self.norm = TSPulseNormLayer(config)
        self.gated_attn = config.gated_attn

        if config.free_channel_flow and config.channel_mix_init == "identity":
            self.config = copy.deepcopy(config)
            self.config.expansion_factor = 1
        else:
            self.config = config

        self.mlp = TSPulseMLP(
            in_features=config.num_input_channels,
            out_features=config.num_input_channels,
            config=self.config,
        )

        if config.gated_attn:
            self.gating_block = TSPulseGatedAttention(
                in_size=config.num_input_channels,
                out_size=config.num_input_channels,
                attention_activation=config.gated_attention_activation,
            )

    def _init_identity_weights(self):
        logger.info("Init identity weights for channel mixing")
        # recursive_init_identity_modules(self)
        self.mlp._init_identity_weights()
        if self.config.gated_attn:
            self.gating_block._init_identity_weights()

    def forward(self, inputs: torch.Tensor):
        """
        Args:
            inputs (`torch.Tensor` of shape `((batch_size, num_channels, num_patches, d_model))`):
                input to the MLP layer
        Returns:
            `torch.Tensor` of the same shape as `inputs`
        """
        residual = inputs
        inputs = self.norm(inputs)

        inputs = inputs.permute(0, 3, 2, 1)

        if self.gated_attn:
            inputs = self.gating_block(inputs)

        inputs = self.mlp(inputs)

        inputs = inputs.permute(0, 3, 2, 1)

        out = inputs + residual
        return out


class TSPulseAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[TSPulseConfig] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        # `past_key_value[0].shape[2] == key_value_states.shape[1]`
        # is checking that the `sequence_length` of the `past_key_value` is the same as
        # the provided `key_value_states` to support prefix tuning
        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class PatchMixerBlock(nn.Module):
    """This module mixes the patch dimension.

    Args:
        config (`TSPulseConfig`, *required*):
            Configuration.
    """

    def __init__(self, config: TSPulseConfig):
        super().__init__()

        self.norm = TSPulseNormLayer(config)

        self.self_attn = config.self_attn
        self.gated_attn = config.gated_attn

        self.mlp = TSPulseMLP(
            in_features=config.num_patches,
            out_features=config.num_patches,
            config=config,
        )

        if config.gated_attn:
            self.gating_block = TSPulseGatedAttention(
                in_size=config.num_patches,
                out_size=config.num_patches,
                attention_activation=config.gated_attention_activation,
            )

        if config.self_attn:
            self.self_attn_layer = TSPulseAttention(
                embed_dim=config.d_model,
                num_heads=config.self_attn_heads,
                dropout=config.dropout,
            )
            self.norm_attn = TSPulseNormLayer(config)

    def forward(self, hidden_state):
        """
        Args:
            hidden_state (`torch.Tensor`): Input tensor.

        Returns:
            `torch.Tensor`: Transformed tensor.
        """
        residual = hidden_state

        hidden_state = self.norm(hidden_state)

        if self.self_attn:
            batch_size, n_vars, num_patches, d_model = hidden_state.shape
            hidden_state_reshaped = hidden_state.reshape(batch_size * n_vars, num_patches, d_model)

            x_attn, _, _ = self.self_attn_layer(hidden_state_reshaped, output_attentions=False)
            x_attn = x_attn.reshape(batch_size, n_vars, num_patches, d_model)

        # Transpose so that num_patches is the last dimension
        hidden_state = hidden_state.transpose(2, 3)
        hidden_state = self.mlp(hidden_state)

        if self.gated_attn:
            hidden_state = self.gating_block(hidden_state)

        # Transpose back
        hidden_state = hidden_state.transpose(2, 3)

        if self.self_attn:
            hidden_state = self.norm_attn(hidden_state + x_attn)

        out = hidden_state + residual
        return out


class FeatureMixerBlock(nn.Module):
    """This module mixes the hidden feature dimension.

    Args:
        config (`TSPulseConfig`, *required*):
            Configuration.
    """

    def __init__(self, config: TSPulseConfig):
        super().__init__()

        self.norm = TSPulseNormLayer(config)

        self.gated_attn = config.gated_attn

        self.mlp = TSPulseMLP(
            in_features=config.d_model,
            out_features=config.d_model,
            config=config,
        )

        if config.gated_attn:
            self.gating_block = TSPulseGatedAttention(
                in_size=config.d_model,
                out_size=config.d_model,
                attention_activation=config.gated_attention_activation,
            )

    def forward(self, hidden: torch.Tensor):
        """
        Args:
            hidden (`torch.Tensor` of shape `(batch_size, num_patches, d_model)`):
                Input tensor to the layer.

        Returns:
            `torch.Tensor`: Transformed tensor.
        """
        residual = hidden
        hidden = self.norm(hidden)
        hidden = self.mlp(hidden)

        if self.gated_attn:
            hidden = self.gating_block(hidden)

        out = hidden + residual
        return out


class TSPulseLayer(nn.Module):
    """
    The `TSPulse` layer that does all three kinds of mixing.

    Args:
        config (`TSPulseConfig`, *required*):
            Configuration.

    """

    def __init__(self, config: TSPulseConfig):
        super().__init__()

        if config.num_patches > 1:
            self.patch_mixer = PatchMixerBlock(config=config)

        self.feature_mixer = FeatureMixerBlock(config=config)

        self.mode = config.mode
        self.num_patches = config.num_patches
        if config.mode == "mix_channel":
            self.channel_feature_mixer = TSPulseChannelFeatureMixerBlock(config=config)

    def forward(self, hidden: torch.Tensor):
        """
        Args:
            hidden (`torch.Tensor` of shape `(batch_size, num_patches, d_model)`):
                Input tensor to the layer.

        Returns:
            `torch.Tensor`: Transformed tensor.
        """
        if self.mode == "mix_channel":
            hidden = self.channel_feature_mixer(hidden)

        if self.num_patches > 1:
            hidden = self.patch_mixer(hidden)
        hidden = self.feature_mixer(hidden)  # hidden: (batch_size x num_patches x d_model)
        return hidden


class LTranspose(nn.Module):
    """Helper module to transpose"""

    def __init__(self, dim1, dim2):
        super(LTranspose, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        x = torch.transpose(x, self.dim1, self.dim2)  # Transpose dimensions 1 and 2
        return x


class TSPulseBlock(nn.Module):
    """The main computing framework of the `TSPulse` model.

    Args:
        config (`TSPulseConfig`, *required*):
            Configuration.
    """

    def __init__(self, config: TSPulseConfig, return_tuple=True):
        super().__init__()

        num_layers = config.num_layers

        self.mixers = nn.ModuleList()
        current_d_model = config.d_model
        current_num_patches = config.num_patches
        current_num_input_channels = config.num_input_channels
        self.return_tuple = return_tuple
        for i in range(num_layers):
            temp_config = copy.deepcopy(config)
            temp_config.num_patches = config.num_patches_layerwise[i]
            temp_config.num_input_channels = config.num_channels_layerwise[i]
            temp_config.d_model = config.d_model_layerwise[i]

            if current_d_model != temp_config.d_model:
                self.mixers.append(nn.Linear(current_d_model, temp_config.d_model))
                current_d_model = temp_config.d_model

            if current_num_input_channels != temp_config.num_input_channels:
                self.mixers.append(LTranspose(-1, -3))
                self.mixers.append(nn.Linear(current_num_input_channels, temp_config.num_input_channels))
                current_num_input_channels = temp_config.num_input_channels
                self.mixers.append(LTranspose(-1, -3))

            if current_num_patches != temp_config.num_patches:
                self.mixers.append(LTranspose(-1, -2))
                self.mixers.append(nn.Linear(current_num_patches, temp_config.num_patches))
                current_num_patches = temp_config.num_patches
                self.mixers.append(LTranspose(-1, -2))

            self.mixers.append(TSPulseLayer(config=temp_config))

    def forward(self, hidden_state, output_hidden_states: bool = False):
        """
        Args:
            hidden_state (`torch.Tensor`): The input tensor.
            output_hidden_states (`bool`, *optional*, defaults to False.):
                Whether to output the hidden states as well.

        Returns:
            `torch.Tensor`: The embedding. `list`: List of all hidden states if `output_hidden_states` is set to
            `True`.
        """
        all_hidden_states = []

        embedding = hidden_state

        for mod in self.mixers:
            embedding = mod(embedding)
            if output_hidden_states:
                all_hidden_states.append(embedding)

        if self.return_tuple:
            if output_hidden_states:
                return embedding, all_hidden_states
            else:
                return embedding, None
        else:
            return embedding


class TSPulseDecoder(nn.Module):
    """

    Args:
        config (`TSPulseConfig`, *required*):
            Configuration.
    """

    def __init__(self, config: TSPulseConfig):
        super().__init__()

        self.num_input_channels = config.num_input_channels

        decoder_config = copy.deepcopy(config)
        decoder_config.num_layers = config.decoder_num_layers
        decoder_config.d_model = config.d_model_layerwise[-1]
        decoder_config.num_patches = config.num_patches_layerwise[-1]
        decoder_config.num_input_channels = config.num_channels_layerwise[-1]

        decoder_config.dropout = config.head_dropout
        decoder_config.mode = config.decoder_mode
        decoder_config.gated_attention_activation = config.head_gated_attention_activation
        decoder_config.num_channels_layerwise_scale = config.decoder_num_channels_layerwise_scale
        decoder_config.num_patches_layerwise_scale = config.decoder_num_patches_layerwise_scale
        decoder_config.d_model_layerwise_scale = config.decoder_d_model_layerwise_scale

        decoder_config.num_channels_layerwise = config.decoder_num_channels_layerwise
        decoder_config.num_patches_layerwise = config.decoder_num_patches_layerwise
        decoder_config.d_model_layerwise = config.decoder_d_model_layerwise

        self.decoder_block = TSPulseBlock(decoder_config)
        self.decoder_config = decoder_config

    def forward(self, hidden_state, output_hidden_states: bool = False):
        """
        Args:
            hidden_state (`torch.Tensor` of shape `(batch_size x nvars x num_patch x d_model)`): The input tensor from backbone.
            output_hidden_states (`bool`, *optional*, defaults to False.):
                Whether to output the hidden states as well.

        Returns:
            `torch.Tensor`: The embedding. `list`: List of all hidden states if `output_hidden_states` is set to
            `True`.
        """
        if output_hidden_states:
            decoder_hidden_states = []
        else:
            decoder_hidden_states = None

        decoder_input = hidden_state

        decoder_output, hidden_states = self.decoder_block(
            hidden_state=decoder_input, output_hidden_states=output_hidden_states
        )  # bs x nvars x n_patches x d_model

        if output_hidden_states:
            decoder_hidden_states.extend(hidden_states)

        return decoder_output, decoder_hidden_states


class TSPulseForReconstructionHead(nn.Module):
    """Reconstruction Head for Reconstruction

    Args:
        config (`TSPulseConfig`, *required*): Configuration.
    """

    def __init__(self, config: TSPulseConfig):
        super().__init__()

        self.dropout_layer = nn.Dropout(config.head_dropout)

        head_d_model = config.decoder_d_model_layerwise[-1]

        self.reconstruction_type = config.reconstruction_type

        if config.reconstruction_type == "full":
            self.base_reconstruction_block = nn.Linear((config.num_patches * head_d_model), config.context_length)
        else:
            self.base_reconstruction_block = nn.Linear(head_d_model, config.patch_length)

        self.flatten = nn.Flatten(start_dim=-2)

    def forward(self, hidden_features):
        """
        Args:
            hidden_features `(batch_size, n_vars, num_patch, d_model)` in `common_channel`/`mix_channel` mode.): Input hidden
                features.
        Returns:
            `torch.Tensor` of shape `(batch_size, context_length, channels)`.
        """

        if self.reconstruction_type == "full":
            hidden_features = self.flatten(hidden_features)  # [batch_size x n_vars x num_patch * d_model]

        hidden_features = self.dropout_layer(hidden_features)
        reconstruction = self.base_reconstruction_block(
            hidden_features
        )  # [batch_size, n_vars, num_patch, patch_length] or [batch_size x n_vars x context_length]

        if self.reconstruction_type != "full":
            reconstruction = self.flatten(reconstruction)  # [batch_size x n_vars x num_patch*patch_length]

        reconstruction = reconstruction.transpose(-1, -2)  # [batch_size x context_length x n_vars]

        return reconstruction


class TSPulsePreTrainedModel(PreTrainedModel):
    # Weight initialization
    config_class = TSPulseConfig
    base_model_prefix = "model"
    main_input_name = "past_values"
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, TSPulsePositionalEncoding):
            # initialize positional encoding
            if self.config.positional_encoding_type == "random":
                nn.init.normal_(module.position_enc, mean=0.0, std=0.1)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        elif isinstance(module, TSPulseChannelFeatureMixerBlock) and self.config.free_channel_flow:
            logger.info(f"Identity Init in Module: , {module.__class__.__name__}")
            module._init_identity_weights()

        elif isinstance(module, TSPulseBatchNorm):
            module.batchnorm.bias.data.zero_()
            module.batchnorm.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            init.xavier_uniform_(module.weight)  # Xavier uniform initialization for weights
            # Initialize biases if they exist
            if module.bias is not None:
                init.zeros_(module.bias)  # Zero initialization for biases

        elif isinstance(module, nn.Linear):
            logger.info(f"Initializing Linear layers with method: {self.config.init_linear}")
            if self.config.init_linear == "normal":
                module.weight.data.normal_(mean=0.0, std=self.config.init_std)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif self.config.init_linear == "uniform":
                nn.init.uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif self.config.init_linear == "xavier_uniform":
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            else:
                module.reset_parameters()
        elif isinstance(module, nn.Embedding):
            logger.info(f"Initializing Embedding layers with method: {self.config.init_embed}")
            if self.config.init_embed == "normal":
                nn.init.normal_(module.weight)
            elif self.config.init_embed == "uniform":
                nn.init.uniform_(module.weight)
            elif self.config.init_embed == "xavier_uniform":
                nn.init.xavier_uniform_(module.weight)
            else:
                module.reset_parameters()
        # elif isinstance(module, nn.Linear):
        #     module.weight.data.normal_(mean=0.0, std=self.config.init_std)
        #     if module.bias is not None:
        #         module.bias.data.zero_()


class TSPulsePatchify(nn.Module):
    """
    A class to patchify the time series sequence into different patches

    Returns:
        `torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`
    """

    def __init__(self, config: TSPulseConfig):
        super().__init__()

        self.sequence_length = config.context_length
        self.patch_length = config.patch_length
        self.patch_stride = config.patch_stride

        if self.sequence_length <= self.patch_length:
            raise ValueError(
                f"Sequence length ({self.sequence_length}) has to be greater than the patch length ({self.patch_length})"
            )

        # get the number of patches
        self.num_patches = (max(self.sequence_length, self.patch_length) - self.patch_length) // self.patch_stride + 1
        new_sequence_length = self.patch_length + self.patch_stride * (self.num_patches - 1)
        self.sequence_start = self.sequence_length - new_sequence_length

    def forward(self, past_values: torch.Tensor):
        """
        Parameters:
            past_values (`torch.Tensor` of shape `(batch_size, sequence_length, num_channels)`, *required*):
                Input for patchification

        Returns:
            `torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`
        """
        sequence_length = past_values.shape[-2]
        if sequence_length != self.sequence_length:
            raise ValueError(
                f"Input sequence length ({sequence_length}) doesn't match model configuration ({self.sequence_length})."
            )
        # output: [bs x new_sequence_length x num_channels]
        output = past_values[:, self.sequence_start :, :]
        # output: [bs x num_patches x num_input_channels x patch_length]
        output = output.unfold(dimension=-2, size=self.patch_length, step=self.patch_stride)
        # output: [bs x num_input_channels x num_patches x patch_length]
        output = output.transpose(-2, -3).contiguous()
        return output


class TSPulseLearnableMaskRevIN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = getattr(config, "scaling_dim", 1)
        self.keepdim = getattr(config, "keepdim", True)
        self.minimum_scale = getattr(config, "minimum_scale", 1e-5)
        self.num_channels = getattr(config, "num_input_channels", 1)
        self.affine = getattr(config, "revin_affine", False)

        # Channel-wise affine only if decoder_mode == "mix_channel"
        self.decoder_mode = getattr(config, "decoder_mode", "common_channel")
        self.channel_wise_affine = self.decoder_mode == "mix_channel"

        if self.affine:
            if self.channel_wise_affine and False:
                # disable channel_wise_affine
                # One parameter per channel
                self.affine_weight = nn.Parameter(torch.ones(1, 1, self.num_channels))
                self.affine_bias = nn.Parameter(torch.zeros(1, 1, self.num_channels))
            else:
                # Shared across all channels
                self.affine_weight = nn.Parameter(torch.tensor(1.0).view(1, 1, 1))
                self.affine_bias = nn.Parameter(torch.tensor(0.0).view(1, 1, 1))

    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Normalize observed points, apply optional affine transform.
        Args:
            data: (B, T, C)
            observed_indicator: (B, T, C)
        Returns:
            normed_data, loc, scale
        """
        denominator = observed_indicator.sum(self.dim, keepdim=self.keepdim).clamp_min(1.0)
        loc = (data * observed_indicator).sum(self.dim, keepdim=self.keepdim) / denominator
        variance = (((data - loc) * observed_indicator) ** 2).sum(self.dim, keepdim=self.keepdim) / denominator
        scale = torch.sqrt(variance.clamp_min(self.minimum_scale))

        normalized = (data - loc) / scale
        # do not normalize mask_token as they are learnable..
        normalized = torch.where(observed_indicator.bool(), normalized, data)

        if self.affine:
            normalized = normalized * self.affine_weight + self.affine_bias

        return normalized, loc, scale

    def inverse(
        self,
        data: torch.Tensor,
        loc: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        """
        Undo normalization + affine transform on observed values.
        """
        if self.affine:
            data = (data - self.affine_bias) / self.affine_weight

        restored = data * scale + loc
        # restored = torch.where(observed_indicator.bool(), restored, data)
        return restored


class TSPulseStdScaler(nn.Module):
    """
    Standardize features by calculating the mean and scaling along the first dimension, and then normalizes it by
    subtracting from the mean and dividing by the standard deviation.
    """

    def __init__(self, config: TSPulseConfig):
        super().__init__()
        self.dim = config.scaling_dim if hasattr(config, "scaling_dim") else 1
        self.keepdim = config.keepdim if hasattr(config, "keepdim") else True
        self.minimum_scale = config.minimum_scale if hasattr(config, "minimum_scale") else 1e-5

    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters:
            data (`torch.Tensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                input for Batch norm calculation
            observed_indicator (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                Calculating the scale on the observed indicator.
        Returns:
            tuple of `torch.Tensor` of shapes
                (`(batch_size, sequence_length, num_input_channels)`,`(batch_size, 1, num_input_channels)`,
                `(batch_size, 1, num_input_channels)`)
        """
        denominator = observed_indicator.sum(self.dim, keepdim=self.keepdim)
        denominator = denominator.clamp_min(1.0)
        loc = (data * observed_indicator).sum(self.dim, keepdim=self.keepdim) / denominator

        variance = (((data - loc) * observed_indicator) ** 2).sum(self.dim, keepdim=self.keepdim) / denominator

        variance = variance + self.minimum_scale

        variance = variance.clamp_(min=self.minimum_scale)

        scale = torch.sqrt(variance)

        return (data - loc) / scale, loc, scale

    def inverse(
        self,
        data: torch.Tensor,
        loc: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        """
        Undo normalization transform on observed values.
        """
        restored = data * scale + loc
        return restored


class TSPulseMeanScaler(nn.Module):
    """
    Computes a scaling factor as the weighted average absolute value along the first dimension, and scales the data
    accordingly.
    """

    def __init__(self, config: TSPulseConfig):
        super().__init__()
        self.dim = config.scaling_dim if hasattr(config, "scaling_dim") else 1
        self.keepdim = config.keepdim if hasattr(config, "keepdim") else True
        self.minimum_scale = config.minimum_scale if hasattr(config, "minimum_scale") else 1e-10
        self.default_scale = config.default_scale if hasattr(config, "default_scale") else None

    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters:
            data (`torch.Tensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                input for Batch norm calculation
            observed_indicator (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                Calculating the scale on the observed indicator.
        Returns:
            tuple of `torch.Tensor` of shapes
                (`(batch_size, sequence_length, num_input_channels)`,`(batch_size, 1, num_input_channels)`,
                `(batch_size, 1, num_input_channels)`)
        """
        ts_sum = (data * observed_indicator).abs().sum(self.dim, keepdim=True)
        num_observed = observed_indicator.sum(self.dim, keepdim=True)

        scale = ts_sum / torch.clamp(num_observed, min=1)

        # If `default_scale` is provided, we use it, otherwise we use the scale
        # of the batch.
        if self.default_scale is None:
            batch_sum = ts_sum.sum(dim=0)
            batch_observations = torch.clamp(num_observed.sum(0), min=1)
            default_scale = torch.squeeze(batch_sum / batch_observations)
        else:
            default_scale = self.default_scale * torch.ones_like(scale)

        # apply default scale where there are no observations
        scale = torch.where(num_observed > 0, scale, default_scale)

        # ensure the scale is at least `self.minimum_scale`
        scale = torch.clamp(scale, min=self.minimum_scale)
        scaled_data = data / scale

        if not self.keepdim:
            scale = scale.squeeze(dim=self.dim)

        return scaled_data, torch.zeros_like(scale), scale

    def inverse(
        self,
        data: torch.Tensor,
        loc: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        """
        Undo normalization transform on observed values.
        """
        restored = data * scale + loc
        return restored


class TSPulseFFTMasker(nn.Module):
    def __init__(self, config: TSPulseConfig, batch_mode="full"):
        """
        Args:
            mask_ratio (float): Percentage of elements to mask (0.0 to 1.0).
            strategy (str): Masking strategy, either "random" or "magnitude".
        """
        super(TSPulseFFTMasker, self).__init__()

        assert config.fft_mask_strategy in [
            "random",
            "magnitude",
            "full_magnitude",
        ], "strategy must be 'random' or 'magnitude'."

        self.mask_ratio = config.fft_mask_ratio
        self.strategy = config.fft_mask_strategy
        self.batch_mode = batch_mode

    def forward(self, fft_tensor):
        """
        Args:
            fft_tensor (torch.Tensor): Input FFT tensor of shape (batch, seq_len, channels),
                                       where the first half is the real part and the second half is the imaginary part.

        Returns:
            masked_fft (torch.Tensor): FFT tensor with masked values set to 0.
            mask (torch.Tensor): Binary mask tensor of the same shape as fft_tensor.
        """
        batch, seq_len, channels = fft_tensor.shape
        half_seq_len = seq_len // 2  # First half: real part, Second half: imaginary part
        num_mask = int(half_seq_len * self.mask_ratio)  # Number of elements to mask per channel in the real part

        if self.batch_mode == "odd":
            batch_mask = torch.arange(batch, device=fft_tensor.device) % 2 == 1
        elif self.batch_mode == "even":
            batch_mask = torch.arange(batch, device=fft_tensor.device) % 2 == 0
        else:  # "full"
            batch_mask = torch.ones(batch, dtype=torch.bool, device=fft_tensor.device)

        # Generate the mask for the real part
        if self.strategy == "random":
            # Random masking
            mask = torch.ones(
                (batch, half_seq_len, channels),
                device=fft_tensor.device,
                dtype=torch.bool,
            )
            random_indices = torch.rand(batch, half_seq_len, channels, device=fft_tensor.device).argsort(dim=1)

            mask[batch_mask] = mask[batch_mask].scatter(1, random_indices[batch_mask, :num_mask, :], False)

            # mask.scatter_(
            #     1, random_indices[:, :num_mask, :], False
            # )  # Mask the first `num_mask` indices

        elif self.strategy == "magnitude":
            # Magnitude-based masking
            real_part = fft_tensor[:, :half_seq_len, :]  # Extract real part
            magnitudes = real_part.abs()  # Compute magnitudes
            mask = torch.ones(
                (batch, half_seq_len, channels),
                device=fft_tensor.device,
                dtype=torch.bool,
            )
            _, topk_indices = torch.topk(magnitudes, k=num_mask, dim=1, largest=True)
            # mask.scatter_(1, topk_indices, False)  # Mask the top `num_mask` magnitudes

            mask[batch_mask] = mask[batch_mask].scatter(1, topk_indices[batch_mask], False)

        elif self.strategy == "full_magnitude":
            # Full complex magnitude-based masking
            real_part = fft_tensor[:, :half_seq_len, :]
            imag_part = fft_tensor[:, half_seq_len:, :]
            complex_mag = torch.sqrt(real_part**2 + imag_part**2)

            mask = torch.ones(
                (batch, half_seq_len, channels),
                device=fft_tensor.device,
                dtype=torch.bool,
            )

            # Get top-k indices based on full magnitude
            _, topk_indices = torch.topk(complex_mag, k=num_mask, dim=1, largest=True)

            # Apply masking to selected samples only
            mask[batch_mask] = mask[batch_mask].scatter(1, topk_indices[batch_mask], False)

        else:
            raise Exception("Invalid fft_mask_strategy")

        # Replicate the mask for the imaginary part
        full_mask = torch.cat([mask, mask], dim=1)  # Apply the same mask to the second half (imaginary part)

        # Apply the mask to the FFT tensor
        masked_fft = fft_tensor * full_mask.float()

        return masked_fft, ~full_mask.type(torch.bool)


class TSPulseNOPScaler(nn.Module):
    """
    Assigns a scaling factor equal to 1 along the first dimension, and therefore applies no scaling to the input data.
    """

    def __init__(self, config: TSPulseConfig):
        super().__init__()
        self.dim = config.scaling_dim if hasattr(config, "scaling_dim") else 1
        self.keepdim = config.keepdim if hasattr(config, "keepdim") else True

    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters:
            data (`torch.Tensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                input for Batch norm calculation
        Returns:
            tuple of `torch.Tensor` of shapes
                (`(batch_size, sequence_length, num_input_channels)`,`(batch_size, 1, num_input_channels)`,
                `(batch_size, 1, num_input_channels)`)
        """
        scale = torch.ones_like(data, requires_grad=False).mean(dim=self.dim, keepdim=self.keepdim)
        loc = torch.zeros_like(data, requires_grad=False).mean(dim=self.dim, keepdim=self.keepdim)
        return data, loc, scale

    def inverse(
        self,
        data: torch.Tensor,
        loc: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        """
        Undo normalization transform on observed values.
        """
        restored = data * scale + loc
        return restored


@dataclass
class TSPulseEncoderOutput(ModelOutput):
    """
    Base class for `TSPulseEncoderOutput`, with potential hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, num_patches, d_model)`):
            Hidden-state at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden-states of the model at the output of each layer.
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class TSPulseEncoding(TSPulsePreTrainedModel):
    """
    Encoder for TSPulse which inputs patched time-series and outputs patched embeddings.

    Args:
        config (`TSPulseConfig`, *required*):
            Configuration.
    """

    def __init__(self, config: TSPulseConfig):
        super().__init__(config)
        self.patcher = nn.Linear(config.patch_length, config.d_model)
        if config.use_positional_encoding:
            self.positional_encoder = TSPulsePositionalEncoding(config=config)
        else:
            self.positional_encoder = None

    def forward(
        self,
        past_values: torch.Tensor,
    ):
        r"""
        Args:
            past_values (`torch.FloatTensor` of shape `(batch_size, num_input_channels, num_patches, patch_length)`):
                Context values of the time series.
                For univariate time series, `num_input_channels` dimension should be 1. For multivariate time series,
                it is greater than 1.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_input_channels, num_patches, d_model)`
        """

        patches = self.patcher(past_values)

        # add positional encoder
        if self.positional_encoder is not None:
            patches = self.positional_encoder(patches)

        return patches


class TSPulseEncoder(TSPulsePreTrainedModel):
    """
    Encoder for TSPulse which inputs patched time-series and outputs patched embeddings.

    Args:
        config (`TSPulseConfig`, *required*):
            Configuration.
    """

    def __init__(self, config: TSPulseConfig):
        super().__init__(config)
        self.use_return_dict = config.use_return_dict

        self.patcher = nn.Linear(config.patch_length, config.d_model)
        if config.use_positional_encoding:
            self.positional_encoder = TSPulsePositionalEncoding(config=config)
        else:
            self.positional_encoder = None
        self.mlp_mixer_encoder = TSPulseBlock(config=config)

        self.d_model = config.d_model
        # Initialize weights and apply final processing
        if config.post_init:
            self.post_init()

    @replace_return_docstrings(output_type=TSPulseEncoderOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        past_values: torch.Tensor,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TSPulseEncoderOutput]:
        r"""
        Args:
            past_values (`torch.FloatTensor` of shape `(batch_size, seq_length, num_input_channels)`):
                Context values of the time series.
                For univariate time series, `num_input_channels` dimension should be 1. For multivariate time series,
                it is greater than 1.

            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers.

            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, n_vars, num_patches, d_model)`
        """

        return_dict = return_dict if return_dict is not None else self.use_return_dict

        # flatten [bs x num_patch x d_model]. common_channel/mix_channel: [bs x n_vars x num_patch x d_model]
        patches = self.patcher(past_values)

        # add positional encoder
        if self.positional_encoder is not None:
            patches = self.positional_encoder(patches)

        last_hidden_state, hidden_states = self.mlp_mixer_encoder(patches, output_hidden_states=output_hidden_states)

        if not return_dict:
            return tuple(
                v
                for v in [
                    last_hidden_state,
                    hidden_states,
                ]
            )

        return TSPulseEncoderOutput(last_hidden_state=last_hidden_state, hidden_states=hidden_states)


class TSPulseAddLearnableRegisterTokens(nn.Module):
    def __init__(self, config: TSPulseConfig, device):
        super(TSPulseAddLearnableRegisterTokens, self).__init__()
        self.num_patch_tokens = config.patch_register_tokens
        self.num_channel_tokens = config.channel_register_tokens
        d_model = config.d_model

        # Learnable patch tokens (p): shape (num_patch_tokens x d_model)
        if self.num_patch_tokens is not None:
            self.patch_tokens = nn.Parameter(torch.randn(self.num_patch_tokens, d_model).to(device))

        # Learnable channel tokens (z): shape (num_channel_tokens x d_model)
        if self.num_channel_tokens is not None:
            self.channel_tokens = nn.Parameter(torch.randn(self.num_channel_tokens, d_model).to(device))

    def forward(self, x):
        # Input x shape: batch x num_channels x num_patches x d_model
        batch_size, num_channels, num_patches, d_model = x.size()

        if self.num_patch_tokens is not None:
            # Expand patch tokens along the batch and channel dimensions
            # Result shape: (1 x 1 x num_patch_tokens x d_model)
            patch_tokens_expanded = self.patch_tokens.unsqueeze(0).unsqueeze(0)

            # Add patch tokens to the num_patches dimension
            # Shape: (batch x num_channels x (num_patches + num_patch_tokens) x d_model)
            x = torch.cat(
                [x, patch_tokens_expanded.expand(batch_size, num_channels, -1, -1)],
                dim=2,
            )

        if self.num_channel_tokens is not None:
            # Expand channel tokens along the batch and patch dimensions
            # Result shape: (1 x num_channel_tokens x 1 x d_model)
            channel_tokens_expanded = self.channel_tokens.unsqueeze(0).unsqueeze(2)

            # Add channel tokens to the num_channels dimension
            # Shape: (batch x (num_channels + num_channel_tokens) x (num_patches + num_patch_tokens) x d_model)
            x = torch.cat(
                [x, channel_tokens_expanded.expand(batch_size, -1, x.size(2), -1)],
                dim=1,
            )

        return x


@dataclass
class TSPulseModelOutput(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, num_patches, d_model)`):
            Final hidden states from the backbone output layer. num_patches wil be 2*input_num_patches + patch_register_tokens
        last_hidden_flatten_state (`torch.FloatTensor` of shape `(batch_size, total_embedding_size)`):
            Flattened version of `last_hidden_state`, where `total_embedding_size = num_channels * num_patches * d_model`.
            num_patches wil be 2*input_num_patches + patch_register_tokens
        masked_past_values (`torch.FloatTensor` of shape `(batch_size, context_length, num_channels)`):
            Input time-series after masking has been applied.
        original_past_values_fft (`torch.FloatTensor` of shape `(batch_size, fft_length, num_channels)`):
            Original FFT-transformed input values withony any masking.
        past_values_fft (`torch.FloatTensor` of shape `(batch_size, fft_length, num_channels)`):
            FFT-transformed representation after processing like masking.
        loc (`torch.FloatTensor` of shape `(batch_size, 1, num_channels)`, *optional*):
            Mean of the input sequence used for normalization (e.g., in RevIN).
        scale (`torch.FloatTensor` of shape `(batch_size, 1, num_channels)`, *optional*):
            Standard deviation of the input sequence used for normalization.
        mask (`torch.FloatTensor` of shape `(batch_size, context_length, num_channels)`, *optional*):
            Binary mask indicating which input time points were observed (1) vs. masked (0).
        fft_base_component (`torch.FloatTensor` of shape `(batch_size, 1, num_channels)`, *optional*):
            FFT base component removed from the input. Either dc or last component
        fft_mask (`torch.FloatTensor` of shape `(batch_size, context_length, num_channels)`, *optional*):
            Binary mask indicating masked FFT frequency components.
        fft_real_max (`torch.FloatTensor` of shape `(batch_size, 1, num_channels)`, *optional*):
            Maximum real component value in the FFT representation (used for scaling or normalization).
        fft_imag_max (`torch.FloatTensor` of shape `(batch_size, 1, num_channels)`, *optional*):
            Maximum imaginary component value in the FFT representation.
        original_fft_softmax (`torch.FloatTensor` of shape `(batch_size, fft_length, num_channels)`, *optional*):
            Softmax-transformed FFT input from the original unmaked input used for FFT-based loss computation.

    """

    last_hidden_state: torch.FloatTensor = None
    last_hidden_flatten_state: torch.FloatTensor = None
    masked_past_values: torch.FloatTensor = None
    original_past_values_fft: torch.FloatTensor = None
    past_values_fft: torch.FloatTensor = None
    loc: Optional[torch.FloatTensor] = None
    scale: Optional[torch.FloatTensor] = None
    mask: Optional[torch.FloatTensor] = None
    fft_base_component: Optional[torch.FloatTensor] = None
    fft_mask: Optional[torch.FloatTensor] = None
    fft_real_max: Optional[torch.FloatTensor] = None
    fft_imag_max: Optional[torch.FloatTensor] = None
    original_fft_softmax: Optional[torch.FloatTensor] = None


@add_start_docstrings(
    "The TSPulse Model for time-series forecasting.",
    TSPULSE_START_DOCSTRING,
)
class TSPulseModel(TSPulsePreTrainedModel):
    def __init__(self, config: TSPulseConfig):
        if not config.init_processing:
            config.check_and_init_preprocessing()
        super().__init__(config)

        self.use_return_dict = config.use_return_dict

        self.encoder_block = TSPulseBlock(config)

        self.patching = TSPulsePatchify(config)

        self.time_encoding = TSPulseEncoding(config)

        self.time_masker = None
        self.fft_masker = None

        if config.batch_aware_masking:
            time_batch_mode = "odd"
            fft_batch_mode = "even"
        else:
            time_batch_mode = "full"
            fft_batch_mode = "full"

        if self.config.fuse_fft:
            self.fft_encoding = TSPulseEncoding(config)
            if config.fft_mask_ratio is not None and config.fft_mask_ratio > 0:
                self.fft_masker = TSPulseFFTMasker(config, batch_mode=fft_batch_mode)

        device = next(self.parameters()).device

        if (config.mask_ratio is not None and config.mask_ratio > 0) or config.mask_type == "user":
            self.time_masker = TSPulseMasking(config, device, batch_mode=time_batch_mode)

        if config.scaling == "mean":
            self.scaler = TSPulseMeanScaler(config)
        elif config.scaling == "std" or config.scaling is True:
            self.scaler = TSPulseStdScaler(config)
        elif config.scaling == "revin":
            self.scaler = TSPulseLearnableMaskRevIN(config)
        else:
            self.scaler = TSPulseNOPScaler(config)

        self.d_model = config.d_model
        self.mode = config.mode

        if self.config.fft_applied_on == "scaled_ts" and self.config.fuse_fft:
            self.groundtruth_scaler_for_fft = TSPulseStdScaler(config)

        self.fft_time_consistent_masking = config.fft_time_consistent_masking
        self.embedding_size = config.d_model_layerwise[-1] * config.num_patches_layerwise[-1]

        self.add_tokens = TSPulseAddLearnableRegisterTokens(config, device=device)

        # if self.mode == "mix_channel":
        #     self.embedding_size = (
        #         self.embedding_size * config.num_channels_layerwise[-1]
        #     )

        self.base_norm = nn.LayerNorm(self.config.num_patches * self.config.d_model, eps=config.norm_eps)

        if config.post_init:
            self.post_init()

    def normalize_fft(self, tensor):
        min_val = tensor.min(dim=1, keepdim=True).values
        max_val = tensor.max(dim=1, keepdim=True).values
        normalized = (tensor - min_val) / (max_val - min_val)
        return normalized, min_val, max_val

    def mean_scale_fft(self, tensor):
        mean_val = tensor.mean(dim=1, keepdim=True)
        scaled = tensor - mean_val  # / std_val
        return scaled, mean_val  # , std_val

    # @add_start_docstrings_to_model_forward(TSPULSE_INPUTS_DOCSTRING)
    # @replace_return_docstrings(
    #     output_type=TSPulseModelOutput, config_class=_CONFIG_FOR_DOC
    # )
    def forward(
        self,
        past_values: torch.Tensor,
        past_observed_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = None,
        enable_masking: Optional[bool] = True,
    ) -> TSPulseModelOutput:
        """
        Args:
            past_values (`torch.FloatTensor` of shape `(batch_size, context_length, num_input_channels)`):
                Input past time-series values to be encoded by the model.
            past_observed_mask (`torch.FloatTensor` of shape `(batch_size, context_length, num_input_channels)`, *optional*):
                Binary mask indicating observed (1.0) vs. missing (0.0) values in `past_values`.
                Missing values are assumed to have been imputed (e.g., with zeros).
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether to return the hidden states from all layers in the backbone.
            return_dict (`bool`, *optional*):
                If `True`, returns a [`TSPulseModelOutput`] dictionary. If `False`, returns a tuple instead.
            enable_masking (`bool`, *optional*, defaults to `True`):
                Whether masking to be enabled or not. This is set to False for classification task to disable masking during eval or inference.

        Returns: `TSPulseModelOutput` or `tuple`

        """
        return_dict = return_dict if return_dict is not None else self.use_return_dict

        original_past_values = past_values

        if self.config.fft_applied_on == "scaled_ts" and self.config.fuse_fft:
            original_scaled_past_values, _, _ = self.groundtruth_scaler_for_fft(
                original_past_values,
                observed_indicator=torch.ones_like(original_past_values),
            )

        # set zero tensor as default
        original_past_values_fft = torch.zeros(size=(1,)).to(past_values.device)
        past_values_fft = torch.zeros(size=(1,)).to(past_values.device)
        # do time masking
        mask = torch.zeros(size=(1,)).to(past_values.device)

        if self.time_masker is not None and enable_masking:
            past_values, mask = self.time_masker(
                original_past_values,
                past_observed_mask=past_observed_mask,
            )

            if past_observed_mask is not None and self.config.mask_type == "user":
                # sanity check to avoid errors
                assert torch.equal(past_observed_mask, ~mask)

            past_observed_mask = ~mask

        if past_observed_mask is None:
            past_observed_mask = torch.ones_like(past_values)

        scaled_past_values, loc, scale = self.scaler(past_values, past_observed_mask)
        masked_scaled_past_values = scaled_past_values

        if self.config.channel_virtual_expand_scale and self.config.channel_virtual_expand_scale > 1:
            scaled_past_values = scaled_past_values.repeat(1, 1, self.config.channel_virtual_expand_scale)

        patched_time = self.patching(scaled_past_values)  # [batch_size x num_input_channels x num_patch x patch_length

        patched_encoding = self.time_encoding(patched_time)

        fft_base_component = torch.zeros(size=(1,)).to(past_values.device)
        fft_real_max = torch.zeros(size=(1,)).to(past_values.device)
        fft_imag_max = torch.zeros(size=(1,)).to(past_values.device)
        fft_mask = torch.zeros(size=(1,)).to(past_values.device)
        original_fft_softmax = torch.zeros(size=(1,)).to(past_values.device)
        if self.config.fuse_fft:
            if self.fft_time_consistent_masking is True:
                if self.config.fft_applied_on == "scaled_ts":
                    fft_input = masked_scaled_past_values  # take masked scaled input
                else:
                    fft_input = past_values  # take masked input
            else:
                if self.config.fft_applied_on == "scaled_ts":
                    fft_input = original_scaled_past_values  # take original scaled input
                else:
                    fft_input = original_past_values  # take original input

            (
                original_past_values_fft,
                fft_base_component,
                fft_real_max,
                fft_imag_max,
                original_fft_softmax,
            ) = get_fft(
                fft_input,
                epsilon=self.config.minimum_scale,
                fft_remove_component=self.config.fft_remove_component,
                fft_prob_mode=self.config.fft_prob_mode,
                fft_prob_length=self.config.fft_prob_length,
            )

            if self.fft_masker is not None and enable_masking:
                past_values_fft, fft_mask = self.fft_masker(original_past_values_fft)
            else:
                past_values_fft = original_past_values_fft

            if self.fft_time_consistent_masking is True:
                # get orignal groudtruth without time masking

                if self.config.fft_applied_on == "scaled_ts":
                    fft_input = original_scaled_past_values  # take scaled input
                else:
                    fft_input = original_past_values  # take original input

                (
                    original_past_values_fft,
                    _,
                    _,
                    _,
                    original_fft_softmax,
                ) = get_fft(
                    fft_input,
                    epsilon=self.config.minimum_scale,
                    fft_remove_component=self.config.fft_remove_component,
                    fft_prob_mode=self.config.fft_prob_mode,
                    fft_prob_length=self.config.fft_prob_length,
                )

            if self.config.channel_virtual_expand_scale and self.config.channel_virtual_expand_scale > 1:
                past_values_fft = past_values_fft.repeat(1, 1, self.config.channel_virtual_expand_scale)

            patched_fft = self.patching(past_values_fft)  # [batch_size x num_input_channels x num_patch x patch_length

            patched_fft_encoding = self.fft_encoding(patched_fft)

            patched_encoding = torch.concat([patched_encoding, patched_fft_encoding], dim=-2)

        patched_encoding = self.add_tokens(patched_encoding)
        #  [batch_size x num_input_channels+channel_register_tokens x num_patch+patch_register_tokens x patch_length

        B, C, P, D = patched_encoding.shape

        patched_encoding = patched_encoding.reshape(B, C, P * D)
        patched_encoding = self.base_norm(patched_encoding)
        patched_encoding = patched_encoding.reshape(B, C, P, D)

        encoder_output, _ = self.encoder_block(
            patched_encoding,
        )

        flatten_start_dim = 2

        encoder_output_flatten = encoder_output.flatten(start_dim=flatten_start_dim)

        if not return_dict:
            return tuple(
                v
                for v in [
                    encoder_output,
                    encoder_output_flatten,
                    past_values,
                    original_past_values_fft,
                    past_values_fft,
                    loc,
                    scale,
                    mask,
                    fft_base_component,
                    fft_mask,
                    fft_real_max,
                    fft_imag_max,
                    original_fft_softmax,
                ]
            )

        return TSPulseModelOutput(
            last_hidden_state=encoder_output,
            last_hidden_flatten_state=encoder_output_flatten,
            masked_past_values=past_values,
            original_past_values_fft=original_past_values_fft,
            past_values_fft=past_values_fft,
            loc=loc,
            scale=scale,
            mask=mask,
            fft_base_component=fft_base_component,
            fft_mask=fft_mask,
            fft_real_max=fft_real_max,
            fft_imag_max=fft_imag_max,
            original_fft_softmax=original_fft_softmax,
        )


@dataclass
class TSPulseForReconstructionOutput(ModelOutput):
    """
    Output type of [`TSPulseForReconstructionOutput`].

    Args:
        loss (`torch.FloatTensor`, *optional*):
            Total loss computed during training, including all contributing loss terms.
        reconstruction_outputs (`torch.FloatTensor` of shape `(batch_size, context_length, num_input_channels)`):
            Reconstructed time-series output.
        backbone_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_input_channels, num_patches, d_model)`):
            Backbone encoder representations before decoding. num_patches wil be 2*input_num_patches + patch_register_tokens
        decoder_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_input_channels, num_patches, d_model)`):
            Decoder representations before the reconstruction. num_patches wil be 2*input_num_patches + patch_register_tokens
        fft_reconstruction_outputs (`torch.FloatTensor` of shape `(batch_size, context_length, num_input_channels)`, *optional*):
            FFT Output reconstructed.
        original_past_values_fft (`torch.FloatTensor` of shape `(batch_size, context_length, num_input_channels)`, *optional*):
            Ground-truth FFT signal
        reconstructed_ts_from_fft (`torch.FloatTensor` of shape `(batch_size, context_length, num_input_channels)`, *optional*):
            Time-series reconstructed from FFT space.
        future_values (`tuple(torch.FloatTensor)` of shape `(batch_size, prediction_length, num_input_channels)`, *optional*):
            Ground-truth forecast values (used when forecast head is enabled).
        forecast_output (`tuple(torch.FloatTensor)` of shape `(batch_size, prediction_length, num_input_channels)`, *optional*):
            Forecasted future outputs from the model.
        original_fft_softmax (`tuple(torch.FloatTensor)` of shape `(batch_size, context_length//2, num_input_channels)`, *optional*):
            Ground-truth of fft-prob output (used in probabilistic FFT loss).
        fft_softmax_preds (`tuple(torch.FloatTensor)` of shape `(batch_size, context_length//2, num_input_channels)`, *optional*):
            Predicted fft-prob output
        fft_loss (`tuple(torch.FloatTensor)`, *optional*):
            Loss values computed in the frequency domain.
        reconstruction_loss (`tuple(torch.FloatTensor)`, *optional*):
            Pointwise reconstruction loss (e.g., MSE) computed in time domain.
        forecast_loss (`tuple(torch.FloatTensor)`, *optional*):
            Loss on forecasted values, when applicable.
        reconstructed_ts_from_fft_loss (`tuple(torch.FloatTensor)`, *optional*):
            Loss between original time series and inverse-FFT-reconstructed time series.
        masked_reconstruction_loss (`tuple(torch.FloatTensor)`, *optional*):
            Loss computed only over the masked regions of the input sequence.
        fft_prob_loss (`tuple(torch.FloatTensor)`, *optional*):
            Probabilistic loss in the FFT softmax space.
        masked_past_values (`torch.FloatTensor` of shape `(batch_size, context_length, num_input_channels)`):
            Input sequence after applying masking (masked values are replaced with mask_token).
        past_values (`torch.FloatTensor` of shape `(batch_size, context_length, num_input_channels)`):
            Original input sequence before masking.
        fft_mask (`torch.FloatTensor` of shape `(batch_size, context_length, num_input_channels)`):
            Mask applied in frequency domain, if applicable.
        mask (`torch.FloatTensor` of shape `(batch_size, context_length, num_input_channels)`):
            Boolean mask indicating masked positions in the time domain (1 = mask, 0 = un-masked).
        loc (`torch.FloatTensor` of shape `(batch_size, 1, num_input_channels)`, *optional*):
            Mean used for input normalization (e.g., in RevIN).
        scale (`torch.FloatTensor` of shape `(batch_size, 1, num_input_channels)`, *optional*):
            Standard deviation used for input normalization (e.g., in RevIN).
    """

    loss: Optional[torch.FloatTensor] = None
    reconstruction_outputs: torch.FloatTensor = None
    backbone_hidden_state: torch.FloatTensor = None
    fft_reconstruction_outputs: torch.FloatTensor = None
    original_past_values_fft: torch.FloatTensor = None
    reconstructed_ts_from_fft: torch.FloatTensor = None
    future_values: Optional[Tuple[torch.FloatTensor]] = None
    forecast_output: Optional[Tuple[torch.FloatTensor]] = None
    original_fft_softmax: Optional[Tuple[torch.FloatTensor]] = None
    fft_softmax_preds: Optional[Tuple[torch.FloatTensor]] = None
    fft_loss: Optional[Tuple[torch.FloatTensor]] = None
    reconstruction_loss: Optional[Tuple[torch.FloatTensor]] = None
    forecast_loss: Optional[Tuple[torch.FloatTensor]] = None
    reconstructed_ts_from_fft_loss: Optional[Tuple[torch.FloatTensor]] = None
    masked_reconstruction_loss: Optional[Tuple[torch.FloatTensor]] = None
    fft_prob_loss: Optional[Tuple[torch.FloatTensor]] = None
    masked_past_values: torch.FloatTensor = None
    past_values: torch.FloatTensor = None
    decoder_hidden_state: torch.FloatTensor = None
    fft_mask: torch.FloatTensor = None
    mask: torch.FloatTensor = None
    loc: torch.FloatTensor = None
    scale: torch.FloatTensor = None


@dataclass
class TSPulseForClassificationOutput(ModelOutput):
    """
    Output type of [`TSPulseForReconstructionOutput`].

    Args:
        loss (`torch.FloatTensor`, *optional*, shape `()`):
            Total loss computed during training.
        prediction_outputs (`torch.FloatTensor` of shape `(batch_size, num_targets)`):
            Output predictions from the classification head.
        backbone_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_input_channels, num_patches, d_model)`):
            Backbone embeddings from the encoder before being passed to the decoder.
        decoder_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_input_channels, num_patches, d_model)`):
            Decoder embeddings before being passed to the prediction head.
        loc (`torch.FloatTensor`, *optional*, of shape `(batch_size, 1, num_input_channels)`):
            Mean used for input normalization (e.g., in RevIN).
        scale (`torch.FloatTensor`, *optional*, of shape `(batch_size, 1, num_input_channels)`):
            Standard deviation used for input normalization (e.g., in RevIN).

    """

    loss: Optional[torch.FloatTensor] = None
    prediction_outputs: torch.FloatTensor = None
    backbone_hidden_state: torch.FloatTensor = None
    decoder_hidden_state: torch.FloatTensor = None
    loc: torch.FloatTensor = None
    scale: torch.FloatTensor = None


@dataclass
class TSPulseDecoderWithReconstructionHeadOutput(ModelOutput):
    """
    Output type of [`TSPulseDecoderWithReconstructionHeadOutput`].

    Args:
        reconstruction_outputs (`torch.FloatTensor` of shape `(batch_size, context_length, num_input_channels)`):
            Reconstructed output in the time domain from the decoder head.
        reconstruction_fft_outputs (`torch.FloatTensor` of shape `(batch_size, context_length, num_input_channels)`, *optional*):
            FFT Reconstructed output.
        forecast_output (`torch.FloatTensor` of shape `(batch_size, prediction_length, num_input_channels)`, *optional*):
            Predicted future values if forecasting is enabled.
        reconstructed_ts_from_fft (`torch.FloatTensor` of shape `(batch_size, context_length, num_input_channels)`, *optional*):
            Time-series reconstructed by applying inverse FFT to frequency-domain predictions.
        decoder_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_input_channels, num_patches, d_model)`):
            Final hidden representations from the decoder before passing to the reconstruction head. num_patches wil be 2*input_num_patches + patch_register_tokens
        fft_softmax_preds (`torch.FloatTensor` of shape `(batch_size, context_length//2, num_input_channels)`, *optional*):
            Predicted softmax scores over FFT bins from the decoder.

    """

    reconstruction_outputs: torch.FloatTensor = None
    reconstruction_fft_outputs: torch.FloatTensor = None
    forecast_output: torch.FloatTensor = None
    reconstructed_ts_from_fft: torch.FloatTensor = None
    decoder_hidden_state: torch.FloatTensor = None
    fft_softmax_preds: torch.FloatTensor = None


class TSPulseCategoricalEmbeddingLayer(nn.Module):
    """ """

    def __init__(self, config: TSPulseConfig):
        super().__init__()
        self.categorical_vocab_size_list = config.categorical_vocab_size_list
        self.embedding_layers = nn.ModuleList(
            [nn.Embedding(vocab, config.d_model) for vocab in self.categorical_vocab_size_list]
        )
        self.number_of_categorical_variables = len(self.categorical_vocab_size_list)
        self.num_patches = config.num_patches

    def forward(self, static_categorical_values: torch.Tensor):
        # static_categorical_values [bs x number_of_categorical_variables]
        embedded_tensors = []

        for i in range(self.number_of_categorical_variables):
            embedded_tensor = self.embedding_layers[i](static_categorical_values[:, i].long())
            embedded_tensors.append(embedded_tensor)

        output_tensor = torch.stack(embedded_tensors, dim=1)  # bs x number_of_categorical_variables x d_model
        output_tensor = output_tensor.unsqueeze(2).repeat(
            1, 1, self.num_patches, 1
        )  # bs x number_of_categorical_variables x num_patches x d_model

        return output_tensor


@dataclass
class TSPulseDecoderWithClassificationHeadOutput(ModelOutput):
    """
    Output type of [`TSPulseDecoderWithClassificationHeadOutput`].

    Args:
        prediction_outputs (`torch.FloatTensor` of shape `(batch_size, num_targets)`):
            Output logits from the classification head.
        decoder_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_input_channels, num_patches, d_model)`):
            Final hidden states from the decoder prior to being passed to the head. num_patches wil be 2*input_num_patches + patch_register_tokens
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple containing the hidden states from all decoder layers, including the initial embedding (if applicable).
        decoder_input (`torch.FloatTensor` of shape `(batch_size, num_input_channels, num_patches, d_model)`):
            Input provided to the decoder, typically the encoder output.

    """

    prediction_outputs: torch.FloatTensor = None
    decoder_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_input: torch.FloatTensor = None


def register_token_config_update(config):
    temp_config = copy.deepcopy(config)

    if temp_config.patch_register_tokens is not None:
        temp_config.num_patches += temp_config.patch_register_tokens
        temp_config.num_patches_layerwise = [
            i + temp_config.patch_register_tokens for i in temp_config.num_patches_layerwise
        ]
        temp_config.decoder_num_patches_layerwise = [
            i + temp_config.patch_register_tokens for i in temp_config.decoder_num_patches_layerwise
        ]

    if temp_config.channel_register_tokens is not None:
        temp_config.num_input_channels += temp_config.channel_register_tokens
        temp_config.num_channels_layerwise = [
            i + temp_config.channel_register_tokens for i in temp_config.num_channels_layerwise
        ]
        temp_config.decoder_num_channels_layerwise = [
            i + temp_config.channel_register_tokens for i in temp_config.decoder_num_channels_layerwise
        ]

    if temp_config.channel_virtual_expand_scale and temp_config.channel_virtual_expand_scale > 1:
        temp_config.num_input_channels *= temp_config.channel_virtual_expand_scale
        temp_config.num_channels_layerwise = [
            i * temp_config.channel_virtual_expand_scale for i in temp_config.num_channels_layerwise
        ]
        temp_config.decoder_num_channels_layerwise = [
            i * temp_config.channel_virtual_expand_scale for i in temp_config.decoder_num_channels_layerwise
        ]

    return temp_config


class CrossEntropySoftmaxLoss(nn.Module):
    """
    A PyTorch loss module that computes the categorical cross-entropy loss
    between two softmax probability distributions.
    """

    def __init__(self, epsilon=1e-15):
        """
        Initializes the loss module.

        Args:
            epsilon (float): Small value to prevent log(0) issues.
        """
        super().__init__()
        self.epsilon = epsilon

    def forward(self, softmax_output1, softmax_output2):
        """
        Computes the categorical cross-entropy loss between two softmax probability distributions.

        Args:
            softmax_output1 (torch.Tensor): The true probability distribution.
            softmax_output2 (torch.Tensor): The predicted probability distribution.

        Returns:
            torch.Tensor: Cross-entropy loss.
        """
        # Clip softmax outputs to avoid log(0) errors for numerical stability.
        softmax_output2 = torch.clip(softmax_output2, self.epsilon, 1 - self.epsilon)

        # Compute the cross-entropy loss
        log_softmax_output2 = torch.log(softmax_output2)
        loss = -torch.sum(softmax_output1 * log_softmax_output2, dim=-1)

        return loss.mean()  # Average over batch


class TSPulseForReconstruction(TSPulsePreTrainedModel):
    """
    `TSPulse` for forecasting application.

    Args:
        config (`TSPulseConfig`, *required*):
            Configuration.

    Returns:
        `None`.
    """

    def __init__(self, config: TSPulseConfig):
        super().__init__(config)
        config.check_and_init_preprocessing()

        self.config = config

        if config.loss == "mse":
            self.loss = nn.MSELoss(reduction="mean")
        elif config.loss == "mae":
            self.loss = nn.L1Loss(reduction="mean")
        else:
            raise ValueError("Invalid loss function: Allowed values: mse, mae")

        self.use_return_dict = config.use_return_dict

        config = register_token_config_update(config)

        self.backbone = TSPulseModel(config)

        self.num_input_channels = config.num_input_channels

        self.decoder_with_head = TSPulseDecoderWithReconstructionHead(config)

        if config.decoder_d_model != config.d_model:
            raise Exception("decoder_d_model should be same as d_model for reconstruction tasks")

        if config.decoder_d_model_layerwise[-1] != config.d_model:
            raise ValueError("decoder_d_model_layerwise[-1] should be same as config.d_model")

        if config.decoder_num_patches_layerwise[-1] != config.num_patches:
            raise ValueError("decoder_num_patches_layerwise[-1] should be same as config.num_patches")

        if config.decoder_num_channels_layerwise[-1] != config.num_input_channels:
            raise ValueError("decoder_num_channels_layerwise[-1] should be same as config.num_input_channels")

        self.ces_loss = CrossEntropySoftmaxLoss()

        # Initialize weights and apply final processing
        if config.post_init:
            self.post_init()

    # @add_start_docstrings_to_model_forward(TSPULSE_INPUTS_DOCSTRING)
    # @replace_return_docstrings(
    #     output_type=TSPulseForReconstructionOutput, config_class=_CONFIG_FOR_DOC
    # )
    def forward(
        self,
        past_values: torch.Tensor,
        future_values: Optional[torch.Tensor] = None,
        past_observed_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = False,
        return_loss: bool = True,
        return_dict: Optional[bool] = None,
    ) -> TSPulseForReconstructionOutput:
        """

        Args:
        past_values (`torch.FloatTensor` of shape `(batch_size, context_length, num_input_channels)`):
            Input time series values from the past window. This is the primary input to be reconstructed.
        future_values (`torch.FloatTensor` of shape `(batch_size, prediction_length, num_input_channels)`, *optional*):
            Ground-truth future values. Used to compute forecast loss if applicable when forecast auxillary head enabled.
        past_observed_mask (`torch.FloatTensor` of shape `(batch_size, context_length, num_input_channels)`, *optional*):
            Binary mask indicating observed (1.0) vs. missing (0.0) values in `past_values`. Missing values are assumed
            to have been imputed (e.g., with zeros) and the model is expected to reconstruct them. past_observed_mask is used only when
            mask_type = `"user`"
        output_hidden_states (`bool`, *optional*, defaults to `False`):
            Whether to return the hidden states from each encoder/decoder layer.
        return_loss (`bool`, *optional*, defaults to `True`):
            If `True`, returns the total loss.
        return_dict (`bool`, *optional*):
            If `True` or None, returns a [`TSPulseForReconstructionOutput`] object. If `False`, returns a tuple.


        Returns: `TSPulseForReconstructionOutput` or `tuple`

        """

        return_dict = return_dict if return_dict is not None else self.use_return_dict
        backbone_input = past_values
        # past_values: tensor [batch_size x context_length x num_input_channels]
        model_output = self.backbone(
            backbone_input,
            past_observed_mask=past_observed_mask,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )  # model_output: [batch_size x nvars x num_patch x d_model]

        mask = model_output.mask
        fft_mask = model_output.fft_mask
        fft_base_component = model_output.fft_base_component
        fft_real_max = model_output.fft_real_max
        fft_imag_max = model_output.fft_imag_max

        if isinstance(model_output, tuple):
            model_output = TSPulseModelOutput(*model_output)

        decoder_input = model_output.last_hidden_flatten_state

        loc = model_output.loc
        scale = model_output.scale

        decoder_with_head_output = self.decoder_with_head(
            decoder_input=decoder_input,
            loc=loc,
            scale=scale,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            fft_base_component=fft_base_component,
            fft_real_max=fft_real_max,
            fft_imag_max=fft_imag_max,
        )

        if isinstance(decoder_with_head_output, tuple):
            decoder_with_head_output = TSPulseDecoderWithReconstructionHeadOutput(*decoder_with_head_output)

        forecast_output = decoder_with_head_output.forecast_output
        reconstruction_outputs = decoder_with_head_output.reconstruction_outputs
        reconstructed_ts_from_fft = decoder_with_head_output.reconstructed_ts_from_fft

        if self.config.fft_time_add_forecasting_pt_loss:
            forecast_output = self.backbone.scaler.inverse(data=forecast_output, loc=loc, scale=scale)

        reconstruction_outputs = self.backbone.scaler.inverse(data=reconstruction_outputs, loc=loc, scale=scale)

        if (
            self.config.fft_applied_on == "scaled_ts"
            and self.config.fuse_fft
            and self.config.fft_original_signal_loss_weight > 0
        ):
            reconstructed_ts_from_fft = self.backbone.scaler.inverse(
                data=reconstructed_ts_from_fft, loc=loc, scale=scale
            )

        loss_val = torch.zeros(size=(1,)).to(loc.device)

        fft_loss = torch.zeros(size=(1,)).to(loc.device)

        reconstruction_loss = torch.zeros(size=(1,)).to(loc.device)

        masked_reconstruction_loss = torch.zeros(size=(1,)).to(loc.device)

        forecast_loss = torch.zeros(size=(1,)).to(loc.device)

        reconstructed_ts_from_fft_loss = torch.zeros(size=(1,)).to(loc.device)

        fft_prob_loss = torch.zeros(size=(1,)).to(loc.device)

        if return_loss is True:
            if self.config.loss_apply_mode == "full" or (
                (self.config.mask_ratio is None or self.config.mask_ratio == 0) and self.config.mask_type != "user"
            ):
                reconstruction_loss = self.config.reconstruction_loss_weight * self.loss(
                    reconstruction_outputs, past_values
                )
                loss_val = reconstruction_loss

            else:
                if self.config.loss_apply_mode == "mask":
                    bool_mask = mask.type(torch.bool)

                    masked_reconstruction_loss = self.config.masked_reconstruction_loss_weight * self.loss(
                        reconstruction_outputs[bool_mask],
                        past_values[bool_mask],
                    )

                    loss_val = masked_reconstruction_loss
                elif self.config.loss_apply_mode == "mask_and_full":
                    bool_mask = mask.type(torch.bool)
                    masked_reconstruction_loss = self.config.masked_reconstruction_loss_weight * self.loss(
                        reconstruction_outputs[bool_mask],
                        past_values[bool_mask],
                    )
                    reconstruction_loss = self.config.reconstruction_loss_weight * self.loss(
                        reconstruction_outputs, past_values
                    )
                    loss_val = reconstruction_loss + masked_reconstruction_loss
                else:
                    raise Exception("Invalid loss_apply_mode")

            if self.config.fuse_fft and self.config.fft_weight > 0:
                if (
                    self.config.fft_mask_ratio is None
                    or self.config.fft_mask_ratio == 0
                    or self.config.loss_apply_mode == "full"
                ):
                    fft_loss = self.config.fft_weight * self.loss(
                        decoder_with_head_output.reconstruction_fft_outputs,
                        model_output.original_past_values_fft,
                    )

                else:
                    fft_loss = self.config.fft_weight * self.loss(
                        decoder_with_head_output.reconstruction_fft_outputs[fft_mask],
                        model_output.original_past_values_fft[fft_mask],
                    )

                loss_val = loss_val + fft_loss

            if self.config.fuse_fft and self.config.fft_original_signal_loss_weight > 0:
                bool_mask = mask.type(torch.bool)
                if self.config.loss_apply_mode == "mask" and bool_mask.any():
                    reconstructed_ts_from_fft_loss = self.config.fft_original_signal_loss_weight * self.loss(
                        reconstructed_ts_from_fft[bool_mask],
                        past_values[bool_mask],
                    )
                else:
                    reconstructed_ts_from_fft_loss = self.config.fft_original_signal_loss_weight * self.loss(
                        reconstructed_ts_from_fft, past_values
                    )
                loss_val = loss_val + reconstructed_ts_from_fft_loss

            if self.config.fuse_fft and self.config.enable_fft_prob_loss:
                fft_prob_loss = self.ces_loss(
                    model_output.original_fft_softmax,
                    decoder_with_head_output.fft_softmax_preds,
                )
                fft_prob_loss = self.config.fft_prob_weight * fft_prob_loss

                loss_val = loss_val + fft_prob_loss

            if self.config.fft_time_add_forecasting_pt_loss and future_values is not None:
                forecast_loss = self.config.fft_time_add_forecasting_pt_loss_weight * self.loss(
                    forecast_output, future_values
                )

                loss_val += forecast_loss

        decoder_hidden_state = decoder_with_head_output.decoder_hidden_state
        if not return_dict:
            return tuple(
                v
                for v in [
                    loss_val,
                    reconstruction_outputs,
                    model_output.last_hidden_state,
                    decoder_with_head_output.reconstruction_fft_outputs,
                    model_output.original_past_values_fft,
                    reconstructed_ts_from_fft,
                    future_values,
                    forecast_output,
                    model_output.original_fft_softmax,
                    decoder_with_head_output.fft_softmax_preds,
                    fft_loss,
                    reconstruction_loss,
                    forecast_loss,
                    reconstructed_ts_from_fft_loss,
                    masked_reconstruction_loss,
                    fft_prob_loss,
                    model_output.masked_past_values,
                    past_values,
                    decoder_hidden_state,
                    fft_mask,
                    mask,
                    loc,
                    scale,
                ]
            )

        return TSPulseForReconstructionOutput(
            loss=loss_val,
            reconstruction_outputs=reconstruction_outputs,  # tensor [batch_size x context_length x num_input_channels]
            backbone_hidden_state=model_output.last_hidden_flatten_state,  # x: [batch_size x nvars x num_patch x d_model]
            fft_reconstruction_outputs=decoder_with_head_output.reconstruction_fft_outputs,
            original_past_values_fft=model_output.original_past_values_fft,
            reconstructed_ts_from_fft=reconstructed_ts_from_fft,
            future_values=future_values,
            forecast_output=forecast_output,
            original_fft_softmax=model_output.original_fft_softmax,
            fft_softmax_preds=decoder_with_head_output.fft_softmax_preds,
            fft_loss=fft_loss,
            reconstruction_loss=reconstruction_loss,
            forecast_loss=forecast_loss,
            reconstructed_ts_from_fft_loss=reconstructed_ts_from_fft_loss,
            masked_reconstruction_loss=masked_reconstruction_loss,
            fft_prob_loss=fft_prob_loss,
            masked_past_values=model_output.masked_past_values,
            past_values=past_values,
            decoder_hidden_state=decoder_hidden_state,
            fft_mask=fft_mask,
            mask=mask,
            loc=loc,
            scale=scale,
        )


def get_fft(
    inputs,
    epsilon: float = 1e-5,
    fft_remove_component="dc",
    fft_prob_mode="log",
    fft_prob_length=None,
):
    # input shape [B x seq_len x channels]
    rfft_result = torch.fft.rfft(inputs, dim=1)  # (n+1)/2

    if fft_prob_mode == "log":
        rfft_mag = torch.abs(rfft_result[:, 1:, :])
        rfft_log_mag = torch.log1p(rfft_mag)
        if fft_prob_length is not None:
            rfft_log_mag = rfft_log_mag[:, :fft_prob_length, :]
        rfft_softmax_without_dc = torch.softmax(rfft_log_mag, dim=1)
    elif fft_prob_mode == "plain":
        rfft_mag = torch.abs(rfft_result[:, 1:, :])
        if fft_prob_length is not None:
            rfft_mag = rfft_mag[:, :fft_prob_length, :]
        rfft_softmax_without_dc = torch.softmax(rfft_mag, dim=1)
    else:
        rfft_softmax_without_dc = torch.softmax(rfft_result[:, 1:, :].real, dim=1)  # remove DC component

    # rfft_mag = torch.abs(rfft_result[:, 1:, :])
    # if fft_prob_mode == "log":
    #     rfft_mag = torch.abs(rfft_result[:, 1:, :])
    #     rfft_log_mag = torch.log1p(rfft_mag)
    #     if fft_prob_length is not None:
    #         rfft_log_mag = rfft_log_mag[:, :fft_prob_length, :]
    #     rfft_softmax_without_dc = torch.softmax(rfft_log_mag, dim=1)

    # else:
    #     rfft_softmax_without_dc = torch.softmax(rfft_result[:, 1:, :].real, dim=1)  # remove DC component

    if fft_remove_component == "dc":
        base_component = rfft_result[:, 0:1, :]
        rfft_result = rfft_result[:, 1:, :]  # remove DC component
    else:
        base_component = rfft_result[:, -1:, :]  # Select the last component
        rfft_result = rfft_result[:, :-1, :]  # Remove the last component

    real_part = rfft_result.real
    imag_part = rfft_result.imag

    real_max = real_part.abs().max(dim=1, keepdim=True)[0] + epsilon
    imag_max = imag_part.abs().max(dim=1, keepdim=True)[0] + epsilon

    real_part = real_part / real_max
    imag_part = imag_part / imag_max

    fft_output = torch.cat([real_part, imag_part], dim=1)

    # if fft_prob_length is not None:
    #     rfft_softmax_without_dc = rfft_softmax_without_dc[:, :fft_prob_length, :]

    return fft_output, base_component, real_max, imag_max, rfft_softmax_without_dc


class TSPulseMasking(nn.Module):
    def __init__(self, config: TSPulseConfig, device, batch_mode="full"):
        super().__init__()
        self.config = config

        if self.config.use_learnable_mask_token:
            self.mask_token = torch.nn.Parameter(torch.randn(self.config.mask_block_length).to(device))
        else:
            self.mask_token = None

        self.batch_mode = batch_mode

        self.channel_consistent_masking = config.channel_consistent_masking

    def variable_length_hybrid_masking_with_token(
        self, tensor, mask_percentage, patch_size, full_patch_mask_percentage=0.5
    ):
        """
        Args:
            tensor: [B, T, C]
            mask_percentage: float between 0 and 1
            patch_size: int
            full_patch_mask_percentage: float between 0 and 1 — portion of masked tokens to come from full patch masking

        Returns:
            masked_tensor: [B, T, C]
            mask: [B, T, C] — True where masked
        """
        B, T, C = tensor.shape
        device = tensor.device

        # === Per-sample mask percentage
        sampled_mask_percentages = torch.rand(B, device=device) * mask_percentage

        total_masks = (sampled_mask_percentages * T).long()  # [B]

        if mask_percentage > 0:
            total_masks = total_masks.clamp(min=1)

        if self.mask_token is None:
            self.mask_token = torch.full((patch_size,), 0, dtype=tensor.dtype).to(device)

        # === Patch info
        patch_ids_full = torch.arange(T, device=device) // patch_size  # [T]
        patch_ids_relative = torch.arange(T, device=device) % patch_size  # [T]
        num_patches = T // patch_size
        if self.channel_consistent_masking:
            patch_ids_batched = patch_ids_full.unsqueeze(0).expand(B, T)  # [B, T]
            rand_vals = torch.rand(B, num_patches, device=device)  # [B, P]

            # === Compute number of patches per sample
            full_patch_counts = (
                (full_patch_mask_percentage * total_masks / patch_size).clamp(max=num_patches).long()
            )  # [B]
            sorted_ids = torch.argsort(rand_vals, dim=1)  # [B, P]
            patch_range = torch.arange(num_patches, device=device).unsqueeze(0).expand(B, -1)  # [B, P]
            mask_patch_ids = patch_range < full_patch_counts.unsqueeze(1)  # [B, P]
            selected_patch_ids = torch.gather(sorted_ids, 1, patch_range)  # [B, P]
            top_patch_ids = selected_patch_ids.masked_fill(~mask_patch_ids, -1)  # [B, P]

            # Broadcast comparison to get full patch mask
            full_patch_mask = (patch_ids_batched.unsqueeze(-1) == top_patch_ids.unsqueeze(1)).any(-1)  # [B, T]

            full_patch_count_actual = full_patch_mask.sum(dim=1, keepdim=True)  # [B, 1]
            remaining_to_mask = (total_masks.view(B, 1) - full_patch_count_actual).clamp(min=0)  # [B, 1]

            rand_scores = torch.rand(B, T, device=device)
            rand_scores[full_patch_mask] = float("inf")
            sorted_scores, sorted_indices = torch.sort(rand_scores, dim=1)
            topk_mask = torch.arange(T, device=device).unsqueeze(0) < remaining_to_mask  # [B, T]
            additional_mask = torch.zeros_like(topk_mask)
            additional_mask.scatter_(1, sorted_indices, topk_mask)
            combined_mask = full_patch_mask | additional_mask
            mask = combined_mask.unsqueeze(-1).expand(B, T, C)

        else:
            # Channel-wise masking
            patch_ids_batched = patch_ids_full.view(1, T, 1).expand(B, T, C)
            rand_vals = torch.rand(B, C, num_patches, device=device)  # [B, C, P]

            full_patch_counts = (
                (full_patch_mask_percentage * total_masks / patch_size).clamp(max=num_patches).long()
            )  # [B]
            full_patch_counts_bc = full_patch_counts.view(B, 1).expand(B, C)  # [B, C]
            sorted_ids = torch.argsort(rand_vals, dim=2)  # [B, C, P]
            patch_range = torch.arange(num_patches, device=device).view(1, 1, -1).expand(B, C, -1)  # [B, C, P]
            mask_patch_ids = patch_range < full_patch_counts_bc.unsqueeze(-1)  # [B, C, P]
            selected_patch_ids = torch.gather(sorted_ids, 2, patch_range)  # [B, C, P]
            top_patch_ids = selected_patch_ids.masked_fill(~mask_patch_ids, -1)  # [B, C, P]

            patch_ids_exp = patch_ids_full.view(1, T, 1, 1).expand(B, T, C, num_patches)
            top_patch_ids_exp = top_patch_ids.view(B, 1, C, num_patches).expand(B, T, C, num_patches)
            full_patch_mask = (patch_ids_exp == top_patch_ids_exp).any(-1)  # [B, T, C]

            full_patch_counts_actual = full_patch_mask.sum(dim=1)  # [B, C]
            total_masks_per_channel = total_masks.view(B, 1).expand(B, C)
            remaining_masks = (total_masks_per_channel - full_patch_counts_actual).clamp(min=0)

            rand_scores = torch.rand(B, T, C, device=device)
            rand_scores[full_patch_mask] = float("inf")
            sorted_scores, sorted_indices = torch.sort(rand_scores, dim=1)
            time_range = torch.arange(T, device=device).view(1, T, 1).expand(B, T, C)
            topk_mask = time_range < remaining_masks.view(B, 1, C)
            additional_mask = torch.zeros_like(topk_mask)
            additional_mask.scatter_(1, sorted_indices, topk_mask)
            mask = full_patch_mask | additional_mask

        # === Apply mask token
        patch_pos = patch_ids_relative.unsqueeze(0).expand(B, T)
        selected_token = self.mask_token[patch_pos]
        selected_token = selected_token.unsqueeze(-1).expand(-1, -1, C)
        masked_tensor = torch.where(mask, selected_token, tensor)  # [B, T, C]

        return masked_tensor, mask

    def hybrid_masking_with_token(self, tensor, mask_percentage, patch_size, num_full_patches_to_mask=1):
        """
        Args:
            tensor: [B, T, C]
            mask_percentage: float between 0 and 1
            patch_size: int
            num_full_patches_to_mask: int — how many full patches to mask per sample

        Returns:
            masked_tensor: [B, T, C]
            mask: [B, T, C] — True where masked
        """
        B, T, C = tensor.shape
        device = tensor.device
        total_masks = int(mask_percentage * T)

        if self.mask_token is None:
            self.mask_token = torch.full((patch_size,), 0, dtype=tensor.dtype).to(tensor.device)  # set to default zero

        if num_full_patches_to_mask * patch_size > total_masks:
            logger.warning(
                f"[hybrid_masking_with_token] num_full_patches_to_mask={num_full_patches_to_mask} "
                f"× patch_size={patch_size} > total_masks={total_masks}. Setting to 0."
            )
            num_full_patches_to_mask = 0

        # === Patch info ===
        patch_ids_full = torch.arange(T, device=device) // patch_size  # [T]
        patch_ids_relative = torch.arange(T, device=device) % patch_size  # [T]
        num_patches = T // patch_size

        if self.channel_consistent_masking:
            # -----------------------------
            # Channel-Consistent Masking
            # -----------------------------
            patch_ids_batched = patch_ids_full.unsqueeze(0).expand(B, T)  # [B, T]
            rand_vals = torch.rand(B, num_patches, device=device)
            _, top_patch_ids = torch.topk(rand_vals, k=num_full_patches_to_mask, dim=1)  # [B, K]
            full_patch_mask = (patch_ids_batched.unsqueeze(-1) == top_patch_ids.unsqueeze(1)).any(-1)  # [B, T]

            full_patch_counts = full_patch_mask.sum(dim=1, keepdim=True)  # [B, 1]
            remaining_to_mask = (total_masks - full_patch_counts).clamp(min=0)  # [B, 1]

            rand_scores = torch.rand(B, T, device=device)
            rand_scores[full_patch_mask] = float("inf")
            sorted_scores, sorted_indices = torch.sort(rand_scores, dim=1)
            topk_mask = torch.arange(T, device=device).unsqueeze(0).expand(B, T) < remaining_to_mask
            additional_mask = torch.zeros_like(topk_mask)
            additional_mask.scatter_(1, sorted_indices, topk_mask)
            combined_mask = full_patch_mask | additional_mask  # [B, T]
            mask = combined_mask.unsqueeze(-1).expand(B, T, C)  # [B, T, C]

        else:
            # -----------------------------
            # Channel-Random Masking
            # -----------------------------

            patch_ids_batched = patch_ids_full.view(1, T, 1).expand(B, T, C)  # [B, T, C]
            rand_vals = torch.rand(B, C, num_patches, device=device)
            _, top_patch_ids = torch.topk(rand_vals, k=num_full_patches_to_mask, dim=2)  # [B, C, K]
            selected_patch_ids = top_patch_ids.unsqueeze(1).expand(B, T, C, num_full_patches_to_mask)
            patch_id_exp = patch_ids_full.view(1, T, 1, 1).expand(B, T, C, num_full_patches_to_mask)
            full_patch_mask = (patch_id_exp == selected_patch_ids).any(-1)  # [B, T, C]

            full_patch_counts = full_patch_mask.sum(dim=1)  # [B, C]
            total_masks_per_channel = int(mask_percentage * T)
            remaining_masks = (total_masks_per_channel - full_patch_counts).clamp(min=0)  # [B, C]

            rand_scores = torch.rand(B, T, C, device=device)
            rand_scores[full_patch_mask] = float("inf")
            sorted_scores, sorted_indices = torch.sort(rand_scores, dim=1)
            time_range = torch.arange(T, device=device).view(1, T, 1).expand(B, T, C)
            topk_mask = time_range < remaining_masks.view(B, 1, C)
            additional_mask = torch.zeros_like(topk_mask)
            additional_mask.scatter_(1, sorted_indices, topk_mask)
            mask = full_patch_mask | additional_mask  # [B, T, C]

        # === Mask token selection (per sample) ===
        # self.mask_token: [B, patch_size]
        # self.mask_token: [patch_size]
        patch_pos = patch_ids_relative.unsqueeze(0).expand(B, T)  # [B, T]
        selected_token = self.mask_token[patch_pos]  # [B, T]
        selected_token = selected_token.unsqueeze(-1).expand(-1, -1, C)  # [B, T, C]

        # === Apply mask
        masked_tensor = torch.where(mask, selected_token, tensor)  # [B, T, C]

        return masked_tensor, mask

    def mask_random_percentage(self, tensor, mask_percentage):
        B, T, N = tensor.shape
        mask = torch.rand((B, T, N)).to(tensor.device)
        mask[mask <= mask_percentage] = 0  # masked
        mask[mask > mask_percentage] = 1  # remained
        masked_tensor = tensor.masked_fill(mask == 0, 0)
        mask = ~mask.type(torch.bool)
        return masked_tensor, mask

    def mask_contiguous_with_token(
        self,
        tensor,
        mask_percentage,
        patch_length,
    ):
        channel_consistent = self.channel_consistent_masking

        if self.mask_token is None:
            self.mask_token = torch.full((patch_length,), 0).to(tensor.device)  # set to default zero

        b, s, c = tensor.shape

        # Split into patches: (b, s // patch_length, patch_length, c)
        num_patches = s // patch_length
        # Ensure the sequence length is divisible by the patch length
        if s % patch_length != 0:
            raise ValueError("Sequence length (s) must be divisible by the patch length (K).")

        tensor = tensor.transpose(-1, -2)  # b, c, s
        tensor_patches = tensor.reshape(b, c, num_patches, patch_length)  # b, c, n, l

        # Calculate the number of patches to mask per sample
        num_patches_to_mask_per_sample = int(num_patches * mask_percentage)

        if channel_consistent:
            # Generate a random permutation for each sample in the batch
            rand_indices = torch.rand(b, num_patches, device=tensor.device).argsort(dim=1)
            # Expand dimensions to match the tensor shape for consistent masking across channels
            rand_indices = rand_indices.unsqueeze(1).expand(b, c, num_patches)
        else:
            # Generate a random permutation for each channel in each sample in the batch
            rand_indices = torch.rand(b, c, num_patches, device=tensor.device).argsort(dim=2)

        # Create a mask by setting the first num_patches_to_mask_per_sample entries of each sample to False
        mask_patches = torch.ones((b, c, num_patches), dtype=torch.bool, device=tensor.device)

        if self.batch_mode == "odd":
            batch_mask = torch.arange(b, device=tensor.device) % 2 == 1
        elif self.batch_mode == "even":
            batch_mask = torch.arange(b, device=tensor.device) % 2 == 0
        else:  # "full"
            batch_mask = torch.ones(b, dtype=torch.bool, device=tensor.device)

        mask_indices = rand_indices[:, :, :num_patches_to_mask_per_sample]
        # mask_patches.scatter_(-1, mask_indices, False)
        mask_patches[batch_mask] = mask_patches[batch_mask].scatter(-1, mask_indices[batch_mask], False)

        # Expand the patch mask to match the patch size: (b, c, num_patches, patch_length)
        patch_mask = mask_patches.unsqueeze(-1).expand(-1, -1, -1, patch_length)

        mask_token_expanded = self.mask_token.view(1, 1, 1, patch_length)  # 1,1,1,l
        mask_patches_expanded = mask_patches.unsqueeze(-1)  # b, c, n, 1

        masked_data = torch.where(mask_patches_expanded, tensor_patches, mask_token_expanded)  # b, c, n, l

        masked_data = masked_data.permute(0, 2, 3, 1)  # b, n, l, c
        masked_data = masked_data.reshape(b, s, c)  # b, s, c

        # Convert the patch mask back to the original shape (b, s, c)
        mask = ~patch_mask.reshape(b, c, s // patch_length, patch_length).permute(0, 2, 3, 1).reshape(b, s, c)

        return masked_data, mask

    def mask_with_past_observed(self, tensor: torch.Tensor, past_observed_mask: torch.Tensor, patch_length):
        """
        Args:
            tensor: [B, T, C]
            past_observed_mask: [B, T, C] where 1/True = observed, 0/False = missing

        Returns:
            masked_tensor: [B, T, C]
            mask: [B, T, C] — True where masked
        """
        B, T, C = tensor.shape
        device = tensor.device

        if self.mask_token is None:
            self.mask_token = torch.full((patch_length,), 0).to(tensor.device)  # set to default zero

        if past_observed_mask is None:
            mask = torch.zeros_like(tensor, dtype=torch.bool, device=device)
            return tensor, mask
        else:
            # Invert: we want True where data is missing
            mask = ~past_observed_mask.bool()  # [B, T, C]

            # Determine patch-relative position
            patch_ids_relative = torch.arange(T, device=device) % patch_length  # [T]
            patch_pos = patch_ids_relative.view(1, T, 1).expand(B, T, C)  # [B, T, C]

            selected_token = self.mask_token[patch_pos]  # [B, T, C]

            # Apply mask
            masked_tensor = torch.where(mask, selected_token, tensor)

            return masked_tensor, mask

    def forward(
        self,
        inputs,
        past_observed_mask: Optional[torch.Tensor] = None,
    ):
        if self.config.mask_type == "user":
            masked_inputs, mask = self.mask_with_past_observed(
                tensor=inputs,
                past_observed_mask=past_observed_mask,
                patch_length=self.config.mask_block_length,
            )

        elif self.config.mask_type == "block":
            masked_inputs, mask = self.mask_contiguous_with_token(
                tensor=inputs,
                mask_percentage=self.config.mask_ratio,
                patch_length=self.config.mask_block_length,
            )
        elif self.config.mask_type == "hybrid":
            masked_inputs, mask = self.hybrid_masking_with_token(
                tensor=inputs,
                mask_percentage=self.config.mask_ratio,
                patch_size=self.config.mask_block_length,
                num_full_patches_to_mask=self.config.num_full_patches_for_hybrid_mask,
            )

        elif self.config.mask_type == "var_hybrid":
            masked_inputs, mask = self.variable_length_hybrid_masking_with_token(
                tensor=inputs,
                mask_percentage=self.config.mask_ratio,
                patch_size=self.config.mask_block_length,
                full_patch_mask_percentage=self.config.full_patch_mask_percentage,
            )

        elif self.config.mask_type == "random":
            masked_inputs, mask = self.mask_random_percentage(tensor=inputs, mask_percentage=self.config.mask_ratio)
        else:
            raise Exception("Invalid mask type")

        return masked_inputs, mask


class TSPulseForClassification(TSPulsePreTrainedModel):
    r"""
    `TSPulse` for Classification application. Based on the loss function, we apply
    classification.

    Args:
        config (`TSPulseConfig`, *required*):
            Configuration.

    Returns:
        `None`.
    """

    def __init__(self, config: TSPulseConfig):
        super().__init__(config)

        config.check_and_init_preprocessing(task="classification")

        self.config = config

        if config.loss == "mse":
            self.loss = nn.MSELoss(reduction="mean")
        elif config.loss == "mae":
            self.loss = nn.L1Loss(reduction="mean")
        elif config.loss == "cross_entropy":
            self.loss = nn.CrossEntropyLoss()

        # config.mask_ratio = None

        self.use_return_dict = config.use_return_dict

        config = register_token_config_update(config)

        self.backbone = TSPulseModel(config)

        self.decoder_with_head = TSPulseDecoderWithClassificationHead(config)

        # Initialize weights and apply final processing
        if config.post_init:
            self.post_init()

    # @add_start_docstrings_to_model_forward(TSPULSE_INPUTS_DOCSTRING)
    # @replace_return_docstrings(
    #     output_type=TSPulseForClassificationOutput,
    #     config_class=_CONFIG_FOR_DOC,
    # )
    def forward(
        self,
        past_values: torch.Tensor,
        target_values: torch.Tensor = None,
        past_observed_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = False,
        return_loss: bool = True,
        return_dict: Optional[bool] = None,
        static_categorical_values: Optional[torch.Tensor] = None,
        class_weights: torch.Tensor = None,
    ) -> TSPulseForClassificationOutput:
        """
        Args:
        past_values (`torch.FloatTensor` of shape `(batch_size, context_length, num_input_channels)`):
            Input time series values from the past window.
        target_values (`torch.FloatTensor` of shape `(batch_size,)` for classification:
            Target values or class labels used for computing the loss. For classification, this should be class indices.
        past_observed_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
            Binary mask indicating observed (1.0) vs. missing (0.0) values in `past_values`. Missing values are assumed
            to have been imputed (e.g., with zeros). Used only when mask_type = user.
        output_hidden_states (`bool`, *optional*, defaults to `False`):
            Whether to return the hidden states from each encoder/decoder layer.
        return_loss (`bool`, *optional*, defaults to `True`):
            Whether to compute and return the loss in the output.
        return_dict (`bool`, *optional*):
            If `True`, returns a [`TSPulseForClassificationOutput`] dictionary.
            If `False`, returns a tuple of outputs instead.
        static_categorical_values (`torch.FloatTensor` of shape `(batch_size, num_categorical_features)`, *optional*):
            Tokenized categorical variables associated with each instance. Must match the order of
            `categorical_vocab_size_list` in the config. Not allowed currently.
        class_weights (`torch.FloatTensor` of shape `(num_classes,)`, *optional*):
            Optional weights to apply to each class when computing classification loss. Not allowed currently

        Returns: TSPulseForClassificationOutput or Tuple

        """
        if self.config.loss == "cross_entropy" and class_weights is not None and return_loss is True:
            self.loss = nn.CrossEntropyLoss(weight=class_weights[0])

        # loss = torch.nn.CrossEntropyLoss()
        return_dict = return_dict if return_dict is not None else self.use_return_dict

        enable_masking = True
        if self.config.disable_mask_in_classification_eval:
            enable_masking = self.training

        # past_values: tensor [batch_size x context_length x num_input_channels]
        model_output = self.backbone(
            past_values,
            past_observed_mask=past_observed_mask,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            enable_masking=enable_masking,
        )  # model_output: [batch_size x nvars x num_patch x d_model]

        if isinstance(model_output, tuple):
            model_output = TSPulseModelOutput(*model_output)

        decoder_input = model_output.last_hidden_flatten_state
        # hidden_states = model_output.hidden_states

        loc = model_output.loc
        scale = model_output.scale

        decoder_with_head_output = self.decoder_with_head(
            decoder_input=decoder_input,
            loc=loc,
            scale=scale,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            static_categorical_values=static_categorical_values,
        )

        if isinstance(decoder_with_head_output, tuple):
            decoder_with_head_output = TSPulseDecoderWithClassificationHeadOutput(*decoder_with_head_output)

        # if output_hidden_states:
        #     hidden_states.extend(decoder_with_head_output.hidden_states)

        loss_val = torch.zeros(size=(1,)).to(loc.device)

        if return_loss is True and target_values is not None:
            loss_val = self.loss(decoder_with_head_output.prediction_outputs, target_values)

        decoder_hidden_state = decoder_with_head_output.decoder_hidden_state.flatten(start_dim=2)
        if not return_dict:
            return tuple(
                v
                for v in [
                    loss_val,
                    decoder_with_head_output.prediction_outputs,
                    model_output.last_hidden_flatten_state,
                    decoder_hidden_state,
                    loc,
                    scale,
                ]
            )

        return TSPulseForClassificationOutput(
            loss=loss_val,
            prediction_outputs=decoder_with_head_output.prediction_outputs,  # tensor [batch_size x num_targets]
            backbone_hidden_state=model_output.last_hidden_flatten_state,  # x: [batch_size x nvars x num_patch * d_model]
            decoder_hidden_state=decoder_hidden_state,  # x: [batch_size x nvars x num_patch * decoder_d_model]
            loc=loc,
            scale=scale,
        )


class TSPulseLinearHead(nn.Module):
    """Linear head for Classification.

    Args:
        config (`PatchTSMixerConfig`, *required*):

    """

    def __init__(self, config: TSPulseConfig):
        super().__init__()

        self.config = config
        self.head_aggregation = config.head_aggregation
        self.output_range = config.output_range
        self.actual_local_patches = None
        self.num_patches = config.num_patches

        if config.data_actual_context_length is not None and config.data_actual_context_length < config.context_length:
            self.actual_local_patches = math.ceil(config.data_actual_context_length / config.patch_length)
            self.num_patches = 2 * self.actual_local_patches + config.patch_register_tokens

        head_channels = config.num_input_channels
        if config.head_reduce_channels is not None:
            self.reduce_channel_proj = nn.Linear(config.num_input_channels, config.head_reduce_channels)
            # self.reduce_channel_norm = nn.LayerNorm(
            #     config.head_reduce_channels, eps=config.norm_eps
            # )

            head_channels = config.head_reduce_channels

        d_model_size = config.d_model

        if config.head_reduce_d_model is not None:
            # self.reduce_proj = nn.Sequential(
            #     nn.LayerNorm(config.d_model, eps=config.norm_eps),
            #     nn.Linear(config.d_model, config.head_reduce_d_model),
            # )
            self.reduce_proj = nn.Linear(config.d_model, config.head_reduce_d_model)
            d_model_size = config.head_reduce_d_model

        if config.head_aggregation is None:
            mul_factor = self.num_patches
        else:
            mul_factor = 1

        if config.head_aggregation_dim == "patch" or config.head_aggregation is None:
            dim_size = head_channels
        else:
            dim_size = self.num_patches

        projection_dim = (d_model_size * dim_size * mul_factor) + 2 * config.loc_num_input_channels
        self.projection = nn.Linear(
            projection_dim,
            config.num_targets,
        )
        # self.projection = nn.utils.spectral_norm(self.projection)

        self.loc_scale_norm = nn.Linear(2 * config.loc_num_input_channels, 2 * config.loc_num_input_channels)

        self.flatten = nn.Flatten(start_dim=1)

        # if config.head_aggregation is None:
        #     self.flatten = nn.Flatten(start_dim=-3)
        # else:
        #     self.flatten = nn.Flatten(start_dim=-2)

        self.dropout = nn.Dropout(config.head_dropout)

        config_temp = copy.deepcopy(config)
        config_temp.d_model = projection_dim
        self.head_norm = TSPulseNormLayer(config_temp)

        if self.config.head_attention:
            self.head_global_attention = TSPulseGatedAttention(
                in_size=projection_dim,
                out_size=projection_dim,
                # attention_activation="config.head_gated_attention_activation",
                attention_activation="sigmoid",
            )

    def forward(self, hidden_features, loc, scale):
        """
        Args:
            hidden_features `(batch_size x n_vars x num_patch x d_model)` in `common_channel`/`mix_channel` mode.): Input hidden
                features.

        Returns:
            `torch.Tensor` of shape `(batch_size x num_targets)`.
        """
        if self.actual_local_patches is not None:
            temp_patches = self.config.context_length // self.config.patch_length
            selected_patches = []

            if self.config.classification_mode in [
                "fft_embedding",
                "time_embedding",
                "time_with_short_fft_embedding",
            ]:
                raise Exception(
                    "Pad filtering not allowed for fft_embedding or time_embedding or time_with_short_fft_embedding"
                )

            if self.config.classification_mode in [
                "long_embedding",
                "full_embedding",
            ]:
                selected_patches.append(hidden_features[:, :, : self.actual_local_patches, :])

                if self.config.fuse_fft:
                    selected_patches.append(
                        hidden_features[
                            :,
                            :,
                            temp_patches : temp_patches + self.actual_local_patches,
                            :,
                        ]
                    )

            if self.config.classification_mode in ["short_embedding", "full_embedding"]:
                selected_patches.append(hidden_features[:, :, -self.config.patch_register_tokens :, :])

            hidden_features = torch.cat(selected_patches, dim=2)

        if self.config.head_reduce_channels is not None:
            hidden_features = hidden_features.permute(0, 2, 3, 1)  # batch_size x num_patch x d_model x n_vars
            hidden_features = self.reduce_channel_proj(hidden_features)
            # hidden_features = self.reduce_channel_norm(hidden_features)
            hidden_features = hidden_features.permute(0, 3, 1, 2)

        if self.config.head_reduce_d_model is not None:
            hidden_features = self.reduce_proj(hidden_features)
            # hidden_features = self.reduce_proj_norm(hidden_features)

        if self.head_aggregation is not None:
            if self.config.head_aggregation_dim == "patch" or self.head_aggregation == "use_last":
                hidden_features = hidden_features.transpose(-1, -2)  # batch_size x n_vars x d_model x num_patch
            else:
                hidden_features = hidden_features.transpose(-1, -3)  # batch_size x d_model x num_patches x n_vars

        if self.head_aggregation == "use_last":
            # batch_size x d_model (flatten) or # batch_size x n_vars x d_model (common_channel)
            hidden_features = hidden_features[..., -1]
        elif self.head_aggregation == "max_pool":
            # batch_size x n_vars x d_model or batch_size x d_model
            hidden_features = hidden_features.max(dim=-1).values
        elif self.head_aggregation == "avg_pool":
            # batch_size x n_vars x d_model or batch_size x d_model
            hidden_features = hidden_features.mean(dim=-1)

        hidden_features = self.flatten(hidden_features)

        loc_scale = torch.cat((loc, scale), dim=1)
        loc_scale = loc_scale.reshape(loc.shape[0], -1)
        loc_scale = self.loc_scale_norm(loc_scale)

        hidden_features = torch.cat((hidden_features, loc_scale), dim=-1)
        hidden_features = self.head_norm(hidden_features)
        hidden_features = self.dropout(hidden_features)
        hidden_features = nn.functional.gelu(hidden_features)
        if self.config.head_attention:
            hidden_features = self.head_global_attention(hidden_features)
        hidden_features = self.projection(hidden_features)  # batch_size x num_targets

        if self.output_range is not None:
            hidden_features = (
                torch.sigmoid(hidden_features) * (self.output_range[1] - self.output_range[0]) + self.output_range[0]
            )
        return hidden_features


class TSPulseDecoderWithReconstructionHead(TSPulsePreTrainedModel):
    """
    Decoder + Head

    Args:
        config (`TSPulseConfig`, *required*):
            Configuration.
    """

    def __init__(self, config: TSPulseConfig):
        if not config.init_processing:
            config.check_and_init_preprocessing()
        super().__init__(config)
        self.use_return_dict = config.use_return_dict

        self.config = config

        # decoder_config = copy.deepcopy(config)

        # backbone_num_channels = config.num_channels_layerwise[-1]
        # backbone_num_patches = config.num_patches_layerwise[-1]

        # if config.patch_register_tokens is not None:
        #     decoder_config.num_patches += config.patch_register_tokens
        #     backbone_num_channels += config.patch_register_tokens
        #     decoder_config.decoder_num_patches_layerwise = [
        #         decoder_config.num_patches
        #     ] * decoder_config.decoder_num_layers

        # if config.channel_register_tokens is not None:
        #     decoder_config.num_input_channels += config.channel_register_tokens
        #     backbone_num_patches += config.channel_register_tokens
        #     decoder_config.decoder_num_channels_layerwise = [
        #         decoder_config.num_input_channels
        #     ] * decoder_config.decoder_num_layers

        self.decoder = TSPulseDecoder(config)

        self.reshape_dims = (
            -1,
            config.num_channels_layerwise[-1],
            config.num_patches_layerwise[-1],
            config.d_model_layerwise[-1],
        )

        self.base_num_input_channels = self.config.num_input_channels
        self.base_num_patches = self.config.num_patches
        self.base_backbone_last_num_patches = config.num_patches_layerwise[-1]

        if self.config.channel_register_tokens is not None:
            self.base_num_input_channels -= self.config.channel_register_tokens

        if self.config.channel_virtual_expand_scale and self.config.channel_virtual_expand_scale > 1:
            self.base_num_input_channels = self.base_num_input_channels // self.config.channel_virtual_expand_scale

        if self.config.patch_register_tokens is not None:
            self.base_num_patches -= self.config.patch_register_tokens
            self.base_backbone_last_num_patches -= self.config.patch_register_tokens

        head_config = copy.deepcopy(config)
        head_config.num_patches = self.base_num_patches // 2
        head_config.num_input_channels = self.base_num_input_channels
        head_config.gated_attention_activation = config.head_gated_attention_activation

        self.time_head = TSPulseForReconstructionHead(head_config)

        self.fft_head = None
        if self.config.fuse_fft and (self.config.fft_weight > 0 or self.config.fft_original_signal_loss_weight > 0):
            self.fft_head = TSPulseForReconstructionHead(head_config)

        self.backbone_mode = config.mode

        self.patch_register_embedding_len = config.patch_register_tokens * config.d_model

        if self.config.register_mixer_layers is not None:
            self.register_mix_config = copy.deepcopy(self.decoder.decoder_config)
            self.register_mix_config.num_layers = self.config.register_mixer_layers

            # set patches
            self.register_mix_config.num_patches = self.config.patch_register_tokens
            self.register_mix_config.num_patches_layerwise = [
                self.register_mix_config.num_patches
            ] * self.register_mix_config.num_layers
            self.register_mix_config.num_patches_layerwise_scale = [1] * self.register_mix_config.num_layers

            # set d_model
            self.register_mix_config.d_model = self.register_mix_config.decoder_d_model_layerwise[-1]
            self.register_mix_config.d_model_layerwise = [
                self.register_mix_config.d_model
            ] * self.register_mix_config.num_layers
            self.register_mix_config.d_model_layerwise_scale = [1] * self.register_mix_config.num_layers

            # set num channels
            # self.register_mix_config.num_input_channels = (
            #     self.register_mix_config.decoder_num_channels_layerwise[-1]
            # )
            self.register_mix_config.num_input_channels = self.base_num_input_channels
            self.register_mix_config.num_channels_layerwise = [
                self.register_mix_config.num_input_channels
            ] * self.register_mix_config.num_layers
            self.register_mix_config.num_channels_layerwise_scale = [1] * self.register_mix_config.num_layers

        if self.config.fft_time_add_forecasting_pt_loss is True:
            # self.forecast_mapping = nn.Linear(
            #     config.d_model_layerwise[-1] * self.config.patch_register_tokens,
            #     self.config.prediction_length,
            # )

            if self.config.register_mixer_layers is not None:
                self.forecast_mapping = nn.Sequential(
                    TSPulseBlock(config=self.register_mix_config, return_tuple=False),
                    nn.Flatten(start_dim=2),
                    nn.LayerNorm(self.patch_register_embedding_len, eps=config.norm_eps),
                    nn.Linear(
                        self.patch_register_embedding_len,
                        self.config.prediction_length,
                    ),
                )
            else:
                self.forecast_mapping = nn.Sequential(
                    nn.Flatten(start_dim=2),
                    nn.LayerNorm(self.patch_register_embedding_len, eps=config.norm_eps),
                    TSPulseMLP(
                        in_features=self.patch_register_embedding_len,
                        out_features=self.config.prediction_length,
                        config=config,
                    ),
                )

            # nn.Linear(
            #     config.d_model_layerwise[-1] * self.config.patch_register_tokens,
            #     self.config.prediction_length,
            # )
            # self.forecast_mapping = nn.Linear(
            #     5 * self.base_backbone_last_num_patches, self.config.prediction_length
            # )

        # self.fft_softmax_mapping = nn.Linear(
        #     config.d_model_layerwise[-1] * self.config.patch_register_tokens,
        #     self.config.context_length // 2,
        # )
        self.fft_softmax_mapping = None

        if self.config.enable_fft_prob_loss is True and self.config.fuse_fft:
            if self.config.fft_prob_length is None:
                fft_prob_length = self.config.context_length // 2
            else:
                fft_prob_length = self.config.fft_prob_length

            if self.config.register_mixer_layers is not None:
                self.fft_softmax_mapping = nn.Sequential(
                    TSPulseBlock(config=self.register_mix_config, return_tuple=False),
                    nn.Flatten(start_dim=2),
                    nn.LayerNorm(self.patch_register_embedding_len, eps=config.norm_eps),
                    nn.Linear(
                        self.patch_register_embedding_len,
                        fft_prob_length,
                    ),
                )
            else:
                self.fft_softmax_mapping = nn.Sequential(
                    nn.Flatten(start_dim=2),
                    nn.LayerNorm(self.patch_register_embedding_len, eps=config.norm_eps),
                    TSPulseMLP(
                        in_features=self.patch_register_embedding_len,
                        out_features=fft_prob_length,
                        config=config,
                    ),
                )

    @replace_return_docstrings(
        output_type=TSPulseDecoderWithReconstructionHeadOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        decoder_input: torch.Tensor,
        loc: torch.Tensor,
        scale: torch.Tensor,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = None,
        fft_base_component: Optional[torch.Tensor] = None,
        fft_real_max: Optional[torch.Tensor] = None,
        fft_imag_max: Optional[torch.Tensor] = None,
    ) -> TSPulseDecoderWithReconstructionHeadOutput:
        r"""
        Args:
            decoder_input `torch.Tensor` of shape `(batch_size x emb_size)`): The input tensor from backbone.

            loc (`torch.FloatTensor`, *optional* of shape `(batch_size, 1, num_input_channels)`):
            Input mean

            scale (`torch.FloatTensor`, *optional* of shape `(batch_size, 1, num_input_channels)`):
                Input std dev

            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers.

            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Returns:

        """
        return_dict = return_dict if return_dict is not None else self.use_return_dict

        decoder_input = decoder_input.reshape(self.reshape_dims)

        decoder_output, decoder_hidden_states = self.decoder(decoder_input, output_hidden_states)

        patch_register_output = decoder_output[
            :, : self.base_num_input_channels, -self.config.patch_register_tokens :, :
        ]

        # patch_register_output = patch_register_output.flatten(
        #     start_dim=2
        # )  # bs x c x n*p
        # # filter in case of register tokens

        # patch_register_output = self.register_norm(patch_register_output)
        # breakpoint()

        decoder_hidden_state = decoder_output[:, : self.base_num_input_channels, :, :].flatten(start_dim=2)

        decoder_output = decoder_output[:, : self.base_num_input_channels, : self.base_num_patches, :]

        forecast_output = torch.zeros(size=(1,)).to(decoder_output.device)
        fft_softmax_preds = torch.zeros(size=(1,)).to(decoder_output.device)

        if self.config.fft_time_add_forecasting_pt_loss is True:
            forecast_output = self.forecast_mapping(patch_register_output).transpose(-1, -2)  # bs x c x forecast

            # moved to TSPulseForReconstruction class
            # if loc is not None and scale is not None:
            #     forecast_output = forecast_output * scale + loc

        if self.fft_softmax_mapping is not None:
            fft_softmax_preds = torch.softmax(self.fft_softmax_mapping(patch_register_output).transpose(-1, -2), dim=1)

        reconstruction_fft_outputs = torch.zeros(size=(1,)).to(decoder_output.device)
        reconstructed_ts_from_fft = torch.zeros(size=(1,)).to(decoder_output.device)

        if self.config.fuse_fft:
            decoder_output_time = decoder_output[:, :, 0 : self.base_num_patches // 2, :]
            decoder_output_fft = decoder_output[:, :, -self.base_num_patches // 2 :, :]

            if self.fft_head is not None:
                reconstruction_fft_outputs = self.fft_head(decoder_output_fft)

            reconstruction_outputs = self.time_head(decoder_output_time)

            if self.config.fft_original_signal_loss_weight > 0:
                half_pos = reconstruction_fft_outputs.shape[1] // 2

                real_part = reconstruction_fft_outputs[:, :half_pos, :] * fft_real_max
                imag_part = reconstruction_fft_outputs[:, half_pos:, :] * fft_imag_max

                rfft_result = torch.complex(real_part, imag_part)

                if self.config.fft_remove_component == "dc":
                    # Add back the DC component
                    rfft_result = torch.cat([fft_base_component, rfft_result], dim=1)
                else:
                    rfft_result = torch.cat([rfft_result, fft_base_component], dim=1)

                # Apply inverse FFT to recover the original input
                reconstructed_ts_from_fft = torch.fft.irfft(rfft_result, n=(rfft_result.shape[1] - 1) * 2, dim=1)

                # mag = reconstruction_fft_outputs[:, :half_pos, :] * fft_real_max
                # phase = reconstruction_fft_outputs[:, half_pos:, :] * fft_imag_max

                # # Reconstruct complex FFT coefficients
                # real_part = mag * torch.cos(phase)
                # imag_part = mag * torch.sin(phase)

                # rfft_result = torch.complex(real_part, imag_part)

                # if self.config.fft_remove_component == "dc":
                #     # Add back the DC component
                #     rfft_result = torch.cat([fft_base_component, rfft_result], dim=1)
                # else:
                #     rfft_result = torch.cat([rfft_result, fft_base_component], dim=1)

                # # Apply inverse FFT to recover the original input
                # reconstructed_ts_from_fft = torch.fft.irfft(
                #     rfft_result, n=(rfft_result.shape[1] - 1) * 2, dim=1
                # )

        else:
            reconstruction_outputs = self.time_head(decoder_output)

        # moved to TSPulseForReconstruction class
        # if loc is not None and scale is not None:
        #     reconstruction_outputs = reconstruction_outputs * scale + loc

        if output_hidden_states:
            decoder_hidden_states.append(reconstruction_outputs)

        if not return_dict:
            return tuple(
                v
                for v in [
                    reconstruction_outputs,
                    reconstruction_fft_outputs,
                    forecast_output,
                    reconstructed_ts_from_fft,
                    decoder_hidden_state,
                    fft_softmax_preds,
                ]
            )

        return TSPulseDecoderWithReconstructionHeadOutput(
            reconstruction_outputs=reconstruction_outputs,
            reconstruction_fft_outputs=reconstruction_fft_outputs,
            forecast_output=forecast_output,
            reconstructed_ts_from_fft=reconstructed_ts_from_fft,
            decoder_hidden_state=decoder_hidden_state,
            fft_softmax_preds=fft_softmax_preds,
        )


class TSPulseDecoderWithClassificationHead(TSPulsePreTrainedModel):
    """
    Decoder + Head

    Args:
        config (`TSPulseConfig`, *required*):
            Configuration.
    """

    def __init__(self, config: TSPulseConfig):
        if not config.init_processing:
            config.check_and_init_preprocessing()
        super().__init__(config)
        self.use_return_dict = config.use_return_dict
        self.config = config
        self.backbone_mode = config.mode

        self.reshape_dims = (
            -1,
            config.num_channels_layerwise[-1],
            config.num_patches_layerwise[-1],
            config.d_model_layerwise[-1],
        )

        self.base_num_input_channels = self.config.num_input_channels
        self.base_num_patches = self.config.num_patches
        self.base_backbone_last_num_patches = config.num_patches_layerwise[-1]
        self.base_decoder_last_num_patches = config.decoder_num_patches_layerwise[-1]
        self.base_decoder_last_num_input_channels = config.decoder_num_channels_layerwise[-1]

        if self.config.channel_register_tokens is not None:
            self.base_num_input_channels -= self.config.channel_register_tokens
            self.base_decoder_last_num_input_channels -= self.config.channel_register_tokens

        if self.config.channel_virtual_expand_scale and self.config.channel_virtual_expand_scale > 1:
            self.base_num_input_channels = self.base_num_input_channels // self.config.channel_virtual_expand_scale
            self.base_decoder_last_num_input_channels = (
                self.base_decoder_last_num_input_channels // self.config.channel_virtual_expand_scale
            )

        decoder_config = copy.deepcopy(config)
        decoder_config.gated_attention_activation = config.head_gated_attention_activation
        if config.categorical_vocab_size_list is not None:
            raise Exception("categorical code to be updated... not ready....")
            if config.decoder_mode == "common_channel":
                # logger.warning("Setting decoder_mode to mix_channel as static categorical variables is available")
                # config.decoder_mode = "mix_channel"
                raise ValueError("set decoder_mode to mix_channel when using static categorical variables")

            cat_config = copy.deepcopy(decoder_config)
            cat_config.num_patches = cat_config.num_patches_layerwise[-1]
            cat_config.d_model = cat_config.d_model_layerwise[-1]

            # decoder_config.num_input_channels += len(config.categorical_vocab_size_list)
            self.decoder_cat_embedding_layer = TSPulseCategoricalEmbeddingLayer(cat_config)
            decoder_config.num_channels_layerwise[-1] += len(config.categorical_vocab_size_list)
            decoder_config.decoder_num_channels_layerwise = [
                x + len(config.categorical_vocab_size_list) for x in decoder_config.decoder_num_channels_layerwise
            ]

            self.base_decoder_last_num_input_channels = decoder_config.decoder_num_channels_layerwise[-1]

        else:
            self.decoder_cat_embedding_layer = None

        self.decoder = TSPulseDecoder(decoder_config)

        temp_config = copy.deepcopy(decoder_config)
        # temp_config.num_patches = decoder_config.decoder_num_patches_layerwise[-1]
        # temp_config.num_input_channels = decoder_config.decoder_num_channels_layerwise[
        #     -1
        # ]

        if self.config.classification_mode == "long_embedding":
            self.base_num_patches -= self.config.patch_register_tokens
            self.base_backbone_last_num_patches -= self.config.patch_register_tokens
            self.base_decoder_last_num_patches -= self.config.patch_register_tokens
            temp_config.num_patches = self.base_decoder_last_num_patches
        elif self.config.classification_mode == "short_embedding":
            temp_config.num_patches = self.config.patch_register_tokens
        elif self.config.classification_mode in ["time_embedding", "fft_embedding"]:
            self.base_num_patches -= self.config.patch_register_tokens
            self.base_backbone_last_num_patches -= self.config.patch_register_tokens
            self.base_decoder_last_num_patches -= self.config.patch_register_tokens
            self.base_num_patches = self.base_num_patches // 2
            self.base_backbone_last_num_patches = self.base_backbone_last_num_patches // 2
            self.base_decoder_last_num_patches = self.base_decoder_last_num_patches // 2
            temp_config.num_patches = self.base_decoder_last_num_patches
        elif self.config.classification_mode == "time_with_short_fft_embedding":
            time_patches = (self.base_num_patches - self.config.patch_register_tokens) // 2
            total_patches = time_patches + self.config.patch_register_tokens
            self.time_patches = time_patches
            self.base_num_patches = total_patches
            self.base_backbone_last_num_patches = total_patches
            self.base_decoder_last_num_patches = total_patches
            temp_config.num_patches = self.base_decoder_last_num_patches

        temp_config.num_input_channels = self.base_decoder_last_num_input_channels
        temp_config.loc_num_input_channels = self.base_num_input_channels
        temp_config.d_model = decoder_config.decoder_d_model_layerwise[-1]
        # self.head = ROCKETHead(
        #     input_channels=temp_config.num_input_channels,
        #     num_filters=10000,
        #     num_classes=temp_config.num_targets,
        # )

        self.temp_config = temp_config
        if config.hydra_class_head is None:
            self.head = TSPulseLinearHead(temp_config)
        else:
            self.hydra_heads = nn.ModuleList(
                [
                    TSPulseLinearHead(self._get_updated_config(temp_config, params))
                    for params in config.hydra_class_head
                ]
            )
            # Create a linear layer to compute weights based on x
            if config.hydra_class_attention:
                head_input_size = temp_config.d_model
                # 2 * (
                #     temp_config.num_input_channels * 1 + temp_config.num_patches * 1
                # )

                self.hydra_head_weight_layer = nn.Linear(head_input_size, len(config.hydra_class_head))

        # self.head = TSPulseLinearAttentionHead(temp_config)
        # # Initialize weights and apply final processing
        # if config.post_init:
        #     self.post_init()

    def _get_updated_config(self, base_config, params):
        new_config = copy.deepcopy(base_config)
        new_config.update(params)  # Update the copied config with params
        return new_config

    @replace_return_docstrings(
        output_type=TSPulseDecoderWithClassificationHeadOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        decoder_input: torch.Tensor,
        loc: torch.Tensor,
        scale: torch.Tensor,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = None,
        static_categorical_values: Optional[torch.Tensor] = None,
    ) -> TSPulseDecoderWithClassificationHeadOutput:
        r"""
        Args:
            decoder_input `torch.Tensor` of shape `(batch_size x emb_size)`): The input tensor from backbone.

            loc (`torch.FloatTensor`, *optional* of shape `(batch_size, 1, num_input_channels)`):
            Input mean

            scale (`torch.FloatTensor`, *optional* of shape `(batch_size, 1, num_input_channels)`):
                Input std dev

            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers.

            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

            static_categorical_values (`torch.FloatTensor` of shape `(batch_size, number_of_categorical_variables)`, *optional*):
                Tokenized categorical values can be passed here. Ensure to pass in the same order as the vocab size list used in the
                TSPulseConfig param `categorical_vocab_size_list`

        Returns:

        """

        return_dict = return_dict if return_dict is not None else self.use_return_dict

        # if self.embed_conversion is not None:
        #     decoder_input = self.embed_conversion(
        #         decoder_input
        #     )  # b x total_emb_size or b x c x total_emb_size
        # breakpoint()
        decoder_input = decoder_input.reshape(self.reshape_dims)

        if self.decoder_cat_embedding_layer is not None:
            cat_embeddings = self.decoder_cat_embedding_layer(static_categorical_values)
            decoder_input = torch.concat(
                (decoder_input, cat_embeddings), dim=1
            )  # bs x nvars+n_cat x n_patches x d_model
        decoder_output, decoder_hidden_states = self.decoder(decoder_input, output_hidden_states)

        # # filter in case of register tokens

        if self.config.classification_mode == "long_embedding":
            decoder_output = decoder_output[:, : self.base_num_input_channels, : self.base_num_patches, :]
        elif self.config.classification_mode == "short_embedding":
            decoder_output = decoder_output[
                :,
                : self.base_num_input_channels,
                -self.config.patch_register_tokens :,
                :,
            ]
        elif self.config.classification_mode == "time_embedding":
            decoder_output = decoder_output[:, : self.base_num_input_channels, : self.base_num_patches, :]
        elif self.config.classification_mode == "fft_embedding":
            decoder_output = decoder_output[
                :,
                : self.base_num_input_channels,
                self.base_num_patches : 2 * self.base_num_patches,
                :,
            ]
        elif self.config.classification_mode == "time_with_short_fft_embedding":
            temp_time = decoder_output[
                :,
                : self.base_num_input_channels,
                : self.time_patches,
                :,
            ]
            temp_short = decoder_output[
                :,
                : self.base_num_input_channels,
                -self.config.patch_register_tokens :,
                :,
            ]

            decoder_output = torch.cat([temp_time, temp_short], dim=2)

        else:
            decoder_output = decoder_output[
                :,
                : self.base_num_input_channels,
                :,
                :,
            ]

        if self.config.hydra_class_head is None:
            prediction_outputs = self.head(decoder_output, loc, scale)
        else:
            head_outputs = torch.stack(
                [head(decoder_output, loc, scale) for head in self.hydra_heads],
                dim=1,
            )  # Shape: [batch, num_heads, num_targets]

            if self.config.hydra_class_attention:
                # Compute weights based on x
                # breakpoint()
                E = decoder_output.mean(dim=(1, 2))

                raw_weights = self.hydra_head_weight_layer(E)  # Shape: [batch, num_heads]
                weights = F.softmax(raw_weights, dim=-1)  # Apply softmax across the heads

                # Apply weights and aggregate outputs
                prediction_outputs = torch.einsum("bht,bh->bt", head_outputs, weights)  # Shape: [batch, num_targets]
            else:
                prediction_outputs = head_outputs.mean(dim=1)  # Shape: [batch, num_targets]

        if output_hidden_states:
            decoder_hidden_states.append(prediction_outputs)

        if not return_dict:
            return tuple(
                v
                for v in [
                    prediction_outputs,
                    decoder_output,
                    decoder_hidden_states,
                ]
            )

        return TSPulseDecoderWithClassificationHeadOutput(
            prediction_outputs=prediction_outputs,
            decoder_hidden_state=decoder_output,
            hidden_states=decoder_hidden_states,
            decoder_input=decoder_input,
        )
