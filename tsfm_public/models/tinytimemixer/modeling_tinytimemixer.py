# Copyright contributors to the TSFM project
#
# This code is based on layers and components from the PatchTSMixer model in the HuggingFace Transformers
# Library: https://github.com/huggingface/transformers/blob/main/src/transformers/models/patchtsmixer/modeling_patchtsmixer.py
"""PyTorch TinyTimeMixer model."""

import copy
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel
from transformers.time_series_utils import (
    NegativeBinomialOutput,
    NormalOutput,
    StudentTOutput,
)
from transformers.utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

from .configuration_tinytimemixer import TinyTimeMixerConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "TinyTimeMixerConfig"


TINYTIMEMIXER_PRETRAINED_MODEL_ARCHIVE_LIST = []


TINYTIMEMIXER_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`TinyTimeMixerConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

TINYTIMEMIXER_INPUTS_DOCSTRING = r"""
    Args:
        past_values (`torch.FloatTensor` of shape `(batch_size, seq_length, num_input_channels)`):
            Context values of the time series. For a forecasting task, this denotes the history/past time series values.
            For univariate time series, `num_input_channels` dimension should be 1. For multivariate time series, it is
            greater than 1.

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers.

        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class TinyTimeMixerGatedAttention(nn.Module):
    """
    Module that applies gated attention to input data.

    Args:
        in_size (`int`): The input size.
        out_size (`int`): The output size.
    """

    def __init__(self, in_size: int, out_size: int):
        super().__init__()
        self.attn_layer = nn.Linear(in_size, out_size)
        self.attn_softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        attn_weight = self.attn_softmax(self.attn_layer(inputs))
        inputs = inputs * attn_weight
        return inputs


class TinyTimeMixerCategoricalEmbeddingLayer(nn.Module):
    """ """

    def __init__(self, config: TinyTimeMixerConfig):
        super().__init__()
        self.categorical_vocab_size_list = config.categorical_vocab_size_list
        self.embedding_layers = nn.ModuleList(
            [nn.Embedding(vocab, config.d_model) for vocab in self.categorical_vocab_size_list]
        )
        self.number_of_categorical_variables = len(self.categorical_vocab_size_list)
        self.num_patches = config.num_patches

    def forward(self, static_categorical_values: torch.Tensor):
        """
        Parameters:
            static_categorical_values (`torch.FloatTensor` of shape `(batch_size, number_of_categorical_variables)`):
            Tokenized categorical values can be passed here. Ensure to pass in the same order as the vocab size list used in the
            TinyTimeMixerConfig param `categorical_vocab_size_list`
        Returns:
            `torch.Tensor` of shape `(batch_size, number_of_categorical_variables, num_patches, d_model)`
        """
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


class TinyTimeMixerBatchNorm(nn.Module):
    """
    Compute batch normalization over the sequence length (time) dimension.
    """

    def __init__(self, config: TinyTimeMixerConfig):
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


class TinyTimeMixerPositionalEncoding(nn.Module):
    """
    Class for positional encoding
    """

    def __init__(self, config: TinyTimeMixerConfig):
        super().__init__()
        # positional encoding: [num_patches x d_model]
        if config.use_positional_encoding:
            self.position_enc = self._init_pe(config)
        else:
            self.position_enc = nn.Parameter(torch.zeros(config.num_patches, config.d_model))

    @staticmethod
    def _init_pe(config: TinyTimeMixerConfig) -> nn.Parameter:
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


class TinyTimeMixerNormLayer(nn.Module):
    """Normalization block

    Args:
        config (`TinyTimeMixerConfig`, *required*):
            Configuration.
    """

    def __init__(self, config: TinyTimeMixerConfig):
        super().__init__()

        self.norm_mlp = config.norm_mlp

        if "batch" in config.norm_mlp.lower():
            self.norm = TinyTimeMixerBatchNorm(config)
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


class TinyTimeMixerMLP(nn.Module):
    def __init__(self, in_features, out_features, config):
        super().__init__()
        num_hidden = in_features * config.expansion_factor
        self.fc1 = nn.Linear(in_features, num_hidden)
        self.dropout1 = nn.Dropout(config.dropout)
        self.fc2 = nn.Linear(num_hidden, out_features)
        self.dropout2 = nn.Dropout(config.dropout)

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


class TinyTimeMixerChannelFeatureMixerBlock(nn.Module):
    """This module mixes the features in the channel dimension.

    Args:
        config (`TinyTimeMixerConfig`, *required*):
            Configuration.
    """

    def __init__(self, config: TinyTimeMixerConfig):
        super().__init__()

        self.norm = TinyTimeMixerNormLayer(config)
        self.gated_attn = config.gated_attn
        self.mlp = TinyTimeMixerMLP(
            in_features=config.num_input_channels,
            out_features=config.num_input_channels,
            config=config,
        )

        if config.gated_attn:
            self.gating_block = TinyTimeMixerGatedAttention(
                in_size=config.num_input_channels, out_size=config.num_input_channels
            )

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


class TinyTimeMixerAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[TinyTimeMixerConfig] = None,
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
        config (`TinyTimeMixerConfig`, *required*):
            Configuration.
    """

    def __init__(self, config: TinyTimeMixerConfig):
        super().__init__()

        self.norm = TinyTimeMixerNormLayer(config)

        self.self_attn = config.self_attn
        self.gated_attn = config.gated_attn

        self.mlp = TinyTimeMixerMLP(
            in_features=config.num_patches,
            out_features=config.num_patches,
            config=config,
        )

        if config.gated_attn:
            self.gating_block = TinyTimeMixerGatedAttention(in_size=config.num_patches, out_size=config.num_patches)

        if config.self_attn:
            self.self_attn_layer = TinyTimeMixerAttention(
                embed_dim=config.d_model,
                num_heads=config.self_attn_heads,
                dropout=config.dropout,
            )
            self.norm_attn = TinyTimeMixerNormLayer(config)

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
        config (`TinyTimeMixerConfig`, *required*):
            Configuration.

    """

    def __init__(self, config: TinyTimeMixerConfig):
        super().__init__()

        self.norm = TinyTimeMixerNormLayer(config)

        self.gated_attn = config.gated_attn

        self.mlp = TinyTimeMixerMLP(
            in_features=config.d_model,
            out_features=config.d_model,
            config=config,
        )

        if config.gated_attn:
            self.gating_block = TinyTimeMixerGatedAttention(in_size=config.d_model, out_size=config.d_model)

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


class ForecastChannelHeadMixer(nn.Module):
    """ForecastChannelMixer Module to reconcile forecasts across channels with exogenous support.

    When channel_context_length is positive this mode creates a patch for every multi-variate forecast point with its surronding context
    it then flattens it and applies MLP to it.
    By this way, every forecast point learn from its pre and post surrounding context in a channel mixed way.
    Residual is added to ensure noise reduction with initial forecasts.
    """

    def __init__(self, config: TinyTimeMixerConfig):
        super().__init__()

        self.fcm_context_length = config.fcm_context_length
        self.scl = 2 * self.fcm_context_length + 1

        if config.prediction_channel_indices is not None:
            self.prediction_channel_count = len(config.prediction_channel_indices)
        else:
            self.prediction_channel_count = config.num_input_channels

        if config.exogenous_channel_indices is not None:
            self.exogenous_channel_count = len(config.exogenous_channel_indices)
        else:
            self.exogenous_channel_count = 0

        self.total_channel_count = self.prediction_channel_count + self.exogenous_channel_count

        self.fcm_use_mixer = config.fcm_use_mixer

        self.exogenous_channel_indices = config.exogenous_channel_indices
        self.prediction_channel_indices = config.prediction_channel_indices
        scl_features = self.scl

        if self.fcm_use_mixer:
            # model mixer considering channel dim as patch dim for lag computation
            temp_config = copy.deepcopy(config)
            temp_config.num_patches = self.total_channel_count
            temp_config.patch_length = self.scl
            temp_config.num_input_channels = config.prediction_length
            temp_config.d_model = self.scl * 2
            temp_config.patch_stride = 1
            temp_config.num_layers = config.fcm_mix_layers
            temp_config.dropout = config.head_dropout
            temp_config.mode = "common_channel"
            temp_config.gated_attn = config.fcm_gated_attn
            temp_config.adaptive_patching_levels = 0
            self.exog_mixer = TinyTimeMixerBlock(temp_config)
            scl_features = self.scl * 2
            self.fcm_embedding = nn.Linear(temp_config.patch_length, temp_config.d_model)

        self.mlp = nn.Linear(
            self.total_channel_count * (scl_features),
            self.prediction_channel_count,
        )
        if config.fcm_gated_attn:
            self.fcm_gating_block = TinyTimeMixerGatedAttention(
                in_size=self.total_channel_count * (scl_features),
                out_size=self.total_channel_count * (scl_features),
            )
        if self.fcm_context_length > 0:
            patch_config = copy.deepcopy(config)
            patch_config.context_length = config.prediction_length + (2 * config.fcm_context_length)
            patch_config.patch_length = self.scl
            patch_config.patch_stride = 1
            self.fcm_patch_block = TinyTimeMixerPatchify(patch_config)

        self.fcm_gated_attn = config.fcm_gated_attn
        self.prediction_length = config.prediction_length
        self.fcm_prepend_past = config.fcm_prepend_past

        self.fcm_prepend_past_offset = (
            config.fcm_prepend_past_offset
        )  # Number of items to skip in the context window from the end

        if self.fcm_prepend_past_offset is None:
            self.fcm_prepend_slicing_indices = slice(-self.fcm_context_length, None)
        else:
            self.fcm_prepend_slicing_indices = slice(
                -(self.fcm_prepend_past_offset + self.fcm_context_length),
                -self.fcm_prepend_past_offset,
            )

    def forward(
        self,
        base_forecasts: torch.Tensor,
        past_values: Optional[torch.Tensor],
        future_values: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            base_forecasts (`torch.Tensor` of shape `(batch_size, prediction length, forecast_channels)`):
                Base Forecasts to reconcile

            past_values (`torch.FloatTensor` of shape `(batch_size, seq_length, num_input_channels)`):
            Context values of the time series. For a forecasting task, this denotes the history/past time series values.
            For univariate time series, `num_input_channels` dimension should be 1. For multivariate time series, it is
            greater than 1.

            future_values (`torch.Tensor` of shape `(batch_size, prediction length, input_channels)`, *optional*, Defaults to None):
                Actual groundtruths of the forecasts. Pass dummy values (say 0) for forecast channels, if groundtruth is unknown.
                Pass the correct values for Exogenous channels where the forecast values are known.

        Returns:
            `torch.Tensor`: Updated forecasts of shape `(batch_size, prediction length, forecast_channels)`
        """
        # base_forecasts.shape == (batch_size x forecast_len x n_vars)

        if self.prediction_channel_indices is not None:
            past_prepend_values = past_values[
                :, self.fcm_prepend_slicing_indices, self.prediction_channel_indices
            ]  # bs x context_len x forecast_channels
        else:
            past_prepend_values = past_values[
                :, self.fcm_prepend_slicing_indices, :
            ]  # bs x fcm_context_len x forecast_channels

        if self.exogenous_channel_count > 0 and future_values is None:
            raise ValueError("future_values cannot be none when we have exogenous channels.")

        if self.exogenous_channel_count > 0:
            exog_values = future_values[..., self.exogenous_channel_indices]  # bs x prediction len x exog_channels
            past_exog_values = past_values[
                :, self.fcm_prepend_slicing_indices, self.exogenous_channel_indices
            ]  # bs x context_len x exog_channels

            past_prepend_values = torch.cat(
                (past_prepend_values, past_exog_values), dim=-1
            )  # bs x fcm_context_len x (forecast_channels+exog_channels)

        else:
            exog_values = None

        residual = base_forecasts

        if exog_values is not None:
            base_forecasts = torch.cat(
                (base_forecasts, exog_values), dim=-1
            )  # x.shape == (batch_size x forecast_len x (forecast_channels+exog_channels))

        if self.fcm_context_length > 0:
            # this mode creates a patch for every multi-variate forecast point with its surronding context
            # it then flattens it and applies MLP to it.
            # By this way, every forecast point learn from its pre and post surrounding context in a channel mixed way.
            # Residual is added to ensure noise reduction with initial forecasts.

            # prefill and postfill zeros to enable patching for every forecast point with surrounding context

            dummy = torch.zeros(
                base_forecasts.shape[0],
                self.fcm_context_length,
                base_forecasts.shape[2],
                device=base_forecasts.device,
            )  # bs x fcm_context_length x n_vars

            if self.fcm_prepend_past:
                # add prefill and postfill
                extend_forecasts = torch.concat(
                    (past_prepend_values, base_forecasts, dummy), dim=1
                )  # bs x forecast_len + 2*fcm_context_length x n_vars
            else:
                # add prefill and postfill
                extend_forecasts = torch.concat(
                    (dummy, base_forecasts, dummy), dim=1
                )  # bs x forecast_len + 2*fcm_context_length x n_vars

            # create patch
            extend_forecasts = self.fcm_patch_block(extend_forecasts)  # xb: [bs x n_vars x forecast_len  x scl]

            extend_forecasts = extend_forecasts.transpose(1, 2)  # [bs x forecast_len  x n_vars  x scl]

            if extend_forecasts.shape[1] != self.prediction_length:
                raise ValueError("out_patches should match to forecast length")

            if self.fcm_use_mixer:
                extend_forecasts = self.fcm_embedding(extend_forecasts)
                extend_forecasts, _ = self.exog_mixer(extend_forecasts)

            extend_forecasts = extend_forecasts.flatten(start_dim=2)  # xb: [bs x forecast_len x n_vars * scl]

            if self.fcm_gated_attn:
                extend_forecasts = self.fcm_gating_block(extend_forecasts)  # xb: [bs x forecast_len x n_vars * scl]

            extend_forecasts = self.mlp(extend_forecasts)  # xb: [bs x forecast_len x n_vars]

        else:
            if self.fcm_gated_attn:
                extend_forecasts = self.fcm_gating_block(base_forecasts)

            extend_forecasts = self.mlp(extend_forecasts)

        new_forecast = extend_forecasts + residual

        return new_forecast


class TinyTimeMixerLayer(nn.Module):
    """
    The `TinyTimeMixer` layer that does all three kinds of mixing.

    Args:
        config (`TinyTimeMixerConfig`, *required*):
            Configuration.

    """

    def __init__(self, config: TinyTimeMixerConfig):
        super().__init__()

        if config.num_patches > 1:
            self.patch_mixer = PatchMixerBlock(config=config)

        self.feature_mixer = FeatureMixerBlock(config=config)

        self.mode = config.mode
        self.num_patches = config.num_patches
        if config.mode == "mix_channel":
            self.channel_feature_mixer = TinyTimeMixerChannelFeatureMixerBlock(config=config)

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


class TinyTimeMixerAdaptivePatchingBlock(nn.Module):
    """
    The `TinyTimeMixer` layer that does all three kinds of mixing.

    Args:
        config (`TinyTimeMixerConfig`, *required*):
            Configuration.

    """

    def __init__(self, config: TinyTimeMixerConfig, adapt_patch_level: int):
        super().__init__()
        temp_config = copy.deepcopy(config)
        self.adapt_patch_level = adapt_patch_level
        adaptive_patch_factor = 2**adapt_patch_level
        self.adaptive_patch_factor = adaptive_patch_factor

        if config.d_model // self.adaptive_patch_factor <= 4:
            # do not allow reduction beyond d_model less than 4
            logger.warning(
                "Disabling adaptive patching at level %s. Either increase d_model or reduce adaptive_patching_levels"
                % (adapt_patch_level)
            )
            self.adaptive_patch_factor = 1

        if config.d_model % self.adaptive_patch_factor != 0:
            raise ValueError("d_model should be divisible by 2^i, where i varies from 0 to adaptive_patching_levels.")
        temp_config.num_patches = temp_config.num_patches * self.adaptive_patch_factor
        temp_config.d_model = temp_config.d_model // self.adaptive_patch_factor

        self.mixer_layers = nn.ModuleList([TinyTimeMixerLayer(temp_config) for i in range(temp_config.num_layers)])

    def forward(self, hidden: torch.Tensor):
        """
        Args:
            hidden (`torch.Tensor` of shape `(batch_size x nvars x num_patch x d_model)`):
                Input tensor to the layer.

        Returns:
            `torch.Tensor`: Transformed tensor.
        """
        all_hidden_states = []
        all_hidden_states.append(hidden)
        hidden = torch.reshape(
            hidden,
            (
                hidden.shape[0],
                hidden.shape[1],
                hidden.shape[2] * self.adaptive_patch_factor,
                hidden.shape[3] // self.adaptive_patch_factor,
            ),
        )
        all_hidden_states.append(hidden)

        for mod in self.mixer_layers:
            hidden = mod(hidden)
            all_hidden_states.append(hidden)

        hidden = torch.reshape(
            hidden,
            (
                hidden.shape[0],
                hidden.shape[1],
                hidden.shape[2] // self.adaptive_patch_factor,
                hidden.shape[3] * self.adaptive_patch_factor,
            ),
        )
        all_hidden_states.append(hidden)

        return hidden, all_hidden_states


class TinyTimeMixerBlock(nn.Module):
    """The main computing framework of the `TinyTimeMixer` model.

    Args:
        config (`TinyTimeMixerConfig`, *required*):
            Configuration.
    """

    def __init__(self, config: TinyTimeMixerConfig):
        super().__init__()

        num_layers = config.num_layers

        self.adaptive_patching_levels = config.adaptive_patching_levels

        if self.adaptive_patching_levels > 0:
            self.mixers = nn.ModuleList(
                [
                    TinyTimeMixerAdaptivePatchingBlock(config=config, adapt_patch_level=i)
                    for i in reversed(range(config.adaptive_patching_levels))
                ]
            )

        else:
            self.mixers = nn.ModuleList([TinyTimeMixerLayer(config=config) for _ in range(num_layers)])

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
            if self.adaptive_patching_levels > 0:
                embedding, hidden_states = mod(embedding)
                all_hidden_states.extend(hidden_states)
            else:
                embedding = mod(embedding)
                if output_hidden_states:
                    all_hidden_states.append(embedding)

        if output_hidden_states:
            return embedding, all_hidden_states
        else:
            return embedding, None


class TinyTimeMixerDecoder(nn.Module):
    """Decoder for tiny time mixer

    Args:
        config (`TinyTimeMixerConfig`, *required*):
            Configuration.
    """

    def __init__(self, config: TinyTimeMixerConfig):
        super().__init__()

        if config.d_model != config.decoder_d_model:
            self.adapter = nn.Linear(config.d_model, config.decoder_d_model)
        else:
            self.adapter = None

        self.decoder_raw_residual = config.decoder_raw_residual
        self.num_input_channels = config.num_input_channels

        if config.decoder_raw_residual:
            self.decoder_raw_embedding = nn.Linear(config.patch_length, config.decoder_d_model)
            # nn.init.zeros_(self.decoder_raw_embedding.weight)
            # nn.init.zeros_(self.decoder_raw_embedding.bias)

        decoder_config = copy.deepcopy(config)
        decoder_config.num_layers = config.decoder_num_layers
        decoder_config.d_model = config.decoder_d_model
        decoder_config.dropout = config.head_dropout
        decoder_config.adaptive_patching_levels = config.decoder_adaptive_patching_levels
        decoder_config.mode = config.decoder_mode

        if config.categorical_vocab_size_list is not None:
            if config.decoder_mode == "common_channel":
                # logger.warning("Setting decoder_mode to mix_channel as static categorical variables is available")
                # config.decoder_mode = "mix_channel"
                raise ValueError("set decoder_mode to mix_channel when using static categorical variables")

            decoder_config.num_input_channels += len(config.categorical_vocab_size_list)
            self.decoder_cat_embedding_layer = TinyTimeMixerCategoricalEmbeddingLayer(decoder_config)
        else:
            self.decoder_cat_embedding_layer = None

        self.decoder_block = TinyTimeMixerBlock(decoder_config)

        self.resolution_prefix_tuning = config.resolution_prefix_tuning

    def forward(
        self,
        hidden_state,
        patch_input,
        output_hidden_states: bool = False,
        static_categorical_values: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            hidden_state (`torch.Tensor` of shape `(batch_size x nvars x num_patch x d_model)`): The input tensor from backbone.
            output_hidden_states (`bool`, *optional*, defaults to False.):
                Whether to output the hidden states as well.

            static_categorical_values (`torch.FloatTensor` of shape `(batch_size, number_of_categorical_variables)`, *optional*):
            Tokenized categorical values can be passed here. Ensure to pass in the same order as the vocab size list used in the
            TinyTimeMixerConfig param `categorical_vocab_size_list`

        Returns:
            `torch.Tensor`: The embedding. `list`: List of all hidden states if `output_hidden_states` is set to
            `True`.
        """
        if output_hidden_states:
            decoder_hidden_states = []
        else:
            decoder_hidden_states = None

        decoder_input = hidden_state

        if self.adapter is not None:
            decoder_input = self.adapter(
                hidden_state
            )  # model_output: [batch_size x nvars x num_patch x decoder_d_model]
            if output_hidden_states:
                decoder_hidden_states.append(decoder_input)

        if self.decoder_raw_residual:
            if self.resolution_prefix_tuning:
                if patch_input.shape[-2] == decoder_input.shape[-2] - 1:
                    temp_shape = list(patch_input.shape)
                    temp_shape[-2] = 1
                    temp_zeros = torch.zeros(*temp_shape).to(patch_input.device)
                    patch_input = torch.cat([temp_zeros, patch_input], dim=-2)

            decoder_input = decoder_input + self.decoder_raw_embedding(
                patch_input
            )  # [batch_size x nvars x num_patch x decoder_d_model]
            if output_hidden_states:
                decoder_hidden_states.append(decoder_input)

        if self.decoder_cat_embedding_layer is not None:
            if static_categorical_values is None:
                raise ValueError("Missing static_categorical_values tensor in forward call")
            cat_embeddings = self.decoder_cat_embedding_layer(
                static_categorical_values
            )  # bs x n_cat x n_patches x d_model

            decoder_input = torch.concat(
                (decoder_input, cat_embeddings), dim=1
            )  # bs x nvars+n_cat x n_patches x d_model

        decoder_output, hidden_states = self.decoder_block(
            hidden_state=decoder_input, output_hidden_states=output_hidden_states
        )  # bs x nvars+n_cat x n_patches x d_model

        if output_hidden_states:
            decoder_hidden_states.extend(hidden_states)

        if self.decoder_cat_embedding_layer is not None:
            decoder_output = decoder_output[:, : self.num_input_channels, :, :]  # bs x nvars x n_patches x d_model
            if output_hidden_states:
                decoder_hidden_states.append(decoder_output)

        return decoder_output, decoder_hidden_states


class TinyTimeMixerForPredictionHead(nn.Module):
    """Prediction Head for Forecasting

    Args:
        config (`TinyTimeMixerConfig`, *required*): Configuration.
    """

    def __init__(self, config: TinyTimeMixerConfig, distribution_output=None):
        super().__init__()

        self.prediction_channel_indices = config.prediction_channel_indices

        if self.prediction_channel_indices is not None:
            self.prediction_channel_indices.sort()

        self.prediction_filter_length = config.prediction_filter_length

        self.dropout_layer = nn.Dropout(config.head_dropout)
        self.enable_forecast_channel_mixing = config.enable_forecast_channel_mixing
        if config.use_decoder:
            head_d_model = config.decoder_d_model
        else:
            head_d_model = config.d_model

        if distribution_output is None:
            self.base_forecast_block = nn.Linear((config.num_patches * head_d_model), config.prediction_length)
        else:
            self.base_forecast_block = distribution_output.get_parameter_projection(config.num_patches * head_d_model)

        self.flatten = nn.Flatten(start_dim=-2)

        if self.enable_forecast_channel_mixing:
            temp_config = copy.deepcopy(config)
            if self.prediction_filter_length is not None:
                temp_config.prediction_length = self.prediction_filter_length

            self.fcm_block = ForecastChannelHeadMixer(config=temp_config)

    def forward(self, hidden_features, past_values, future_values=None):
        """

        Args:
            hidden_features `(batch_size, n_vars, num_patch, d_model)` in `common_channel`/`mix_channel` mode.): Input hidden
                features.

            past_values (`torch.FloatTensor` of shape `(batch_size, seq_length, num_input_channels)`):
            Context values of the time series. For a forecasting task, this denotes the history/past time series values.
            For univariate time series, `num_input_channels` dimension should be 1. For multivariate time series, it is
            greater than 1.

            future_values (`torch.Tensor` of shape `(batch_size, prediction length, input_channels)`, *optional*, Defaults to None):
                Actual groundtruths of the forecasts. Pass dummy values (say 0) for forecast channels, if groundtruth is unknown.
                Pass the correct values for Exogenous channels where the forecast values are known.


        Returns:
            `torch.Tensor` of shape `(batch_size, prediction_length, forecast_channels)`.

        """

        hidden_features = self.flatten(hidden_features)  # [batch_size x n_vars x num_patch * d_model]
        hidden_features = self.dropout_layer(hidden_features)  # [batch_size x n_vars x num_patch * d_model]
        forecast = self.base_forecast_block(hidden_features)  # [batch_size x n_vars x prediction_length]
        if isinstance(forecast, tuple):
            forecast = tuple(z.transpose(-1, -2) for z in forecast)
        else:
            forecast = forecast.transpose(-1, -2)  # [batch_size x prediction_length x n_vars]

        if self.prediction_channel_indices is not None:
            if isinstance(forecast, tuple):
                forecast = tuple(z[..., self.prediction_channel_indices] for z in forecast)
            else:
                forecast = forecast[
                    ..., self.prediction_channel_indices
                ]  # [batch_size x prediction_length x prediction_n_vars]

        if self.prediction_filter_length is not None:
            if isinstance(forecast, tuple):
                forecast = tuple(z[:, : self.prediction_filter_length, :] for z in forecast)
            else:
                forecast = forecast[
                    :, : self.prediction_filter_length, :
                ]  # [batch_size x prediction_filter_length x prediction_n_vars]

        if (
            self.prediction_filter_length is not None
            and future_values is not None
            and future_values.shape[1] != self.prediction_filter_length
        ):
            future_values = future_values[
                :, : self.prediction_filter_length, :
            ]  # [batch_size x prediction_filter_length x n_vars]

        if self.enable_forecast_channel_mixing:
            if isinstance(forecast, tuple):
                raise ValueError("Forecast channel mixing is not enabled for distribution head")
            else:
                forecast = self.fcm_block(forecast, past_values=past_values, future_values=future_values)
                # [batch_size x prediction_length x prediction_n_vars]

        return forecast


class TinyTimeMixerPreTrainedModel(PreTrainedModel):
    # Weight initialization
    config_class = TinyTimeMixerConfig
    base_model_prefix = "model"
    main_input_name = "past_values"
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        """Initialize weights"""

        if isinstance(module, TinyTimeMixerPositionalEncoding):
            # initialize positional encoding
            if self.config.positional_encoding_type == "random":
                nn.init.normal_(module.position_enc, mean=0.0, std=0.1)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, TinyTimeMixerBatchNorm):
            module.batchnorm.bias.data.zero_()
            module.batchnorm.weight.data.fill_(1.0)
        elif isinstance(module, nn.Linear):
            # print(f"Initializing Linear layers with method: {self.config.init_linear}")
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
            # print(f"Initializing Embedding layers with method: {self.config.init_embed}")
            if self.config.init_embed == "normal":
                nn.init.normal_(module.weight)
            elif self.config.init_embed == "uniform":
                nn.init.uniform_(module.weight)
            elif self.config.init_embed == "xavier_uniform":
                nn.init.xavier_uniform_(module.weight)
            else:
                module.reset_parameters()


class TinyTimeMixerPatchify(nn.Module):
    """
    A class to patchify the time series sequence into different patches

    Returns:
        `torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`
    """

    def __init__(self, config: TinyTimeMixerConfig):
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


class TinyTimeMixerStdScaler(nn.Module):
    """
    Standardize features by calculating the mean and scaling along the first dimension, and then normalizes it by
    subtracting from the mean and dividing by the standard deviation.
    """

    def __init__(self, config: TinyTimeMixerConfig):
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
        denominator = denominator.clamp_min(torch.tensor(1, device=denominator.device))
        loc = (data * observed_indicator).sum(self.dim, keepdim=self.keepdim) / denominator

        variance = (((data - loc) * observed_indicator) ** 2).sum(self.dim, keepdim=self.keepdim) / denominator
        scale = torch.sqrt(variance + self.minimum_scale)
        return (data - loc) / scale, loc, scale


class TinyTimeMixerMeanScaler(nn.Module):
    """
    Computes a scaling factor as the weighted average absolute value along the first dimension, and scales the data
    accordingly.
    """

    def __init__(self, config: TinyTimeMixerConfig):
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


class TinyTimeMixerNOPScaler(nn.Module):
    """
    Assigns a scaling factor equal to 1 along the first dimension, and therefore applies no scaling to the input data.
    """

    def __init__(self, config: TinyTimeMixerConfig):
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


@dataclass
class TinyTimeMixerEncoderOutput(ModelOutput):
    """
    Base class for `TinyTimeMixerEncoderOutput`, with potential hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, num_patches, d_model)`):
            Hidden-state at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden-states of the model at the output of each layer.
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class TinyTimeMixerEncoder(TinyTimeMixerPreTrainedModel):
    """
    Encoder for TinyTimeMixer which inputs patched time-series and outputs patched embeddings.

    Args:
        config (`TinyTimeMixerConfig`, *required*):
            Configuration.
    """

    def __init__(self, config: TinyTimeMixerConfig):
        if config.init_processing is False:
            config.check_and_init_preprocessing()

        super().__init__(config)

        self.use_return_dict = config.use_return_dict

        self.patcher = nn.Linear(config.patch_length, config.d_model)
        if config.use_positional_encoding:
            self.positional_encoder = TinyTimeMixerPositionalEncoding(config=config)
        else:
            self.positional_encoder = None
        self.mlp_mixer_encoder = TinyTimeMixerBlock(config=config)

        if config.resolution_prefix_tuning:
            mid_dim = (config.patch_length + config.d_model) // 2

            self.freq_mod = nn.Sequential(
                nn.Embedding(config.frequency_token_vocab_size, config.patch_length),
                nn.Linear(config.patch_length, mid_dim),
                nn.GELU(),
                nn.Linear(mid_dim, config.d_model),
            )
        self.resolution_prefix_tuning = config.resolution_prefix_tuning
        self.d_model = config.d_model

        # # Initialize weights and apply final processing
        # if config.post_init:
        #     self.post_init()

    @replace_return_docstrings(output_type=TinyTimeMixerEncoderOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        past_values: torch.Tensor,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = None,
        freq_token: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, TinyTimeMixerEncoderOutput]:
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

        if self.resolution_prefix_tuning:
            if freq_token is not None:
                freq_embedding = self.freq_mod(freq_token.long())  # bs x d_model

                freq_embedding = freq_embedding.view(patches.shape[0], 1, 1, self.d_model)
                freq_embedding = freq_embedding.expand(
                    patches.shape[0],
                    patches.shape[1],
                    1,
                    self.d_model,
                )  # bs x channels x 1 x num_features

                patches = torch.cat((freq_embedding, patches), dim=-2)  # bs x channels x num_patch+1 x num_features

            else:
                raise Exception("Expecting freq_token in forward")

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

        return TinyTimeMixerEncoderOutput(last_hidden_state=last_hidden_state, hidden_states=hidden_states)


@dataclass
class TinyTimeMixerModelOutput(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor`  of shape `(batch_size, num_channels, num_patches, d_model)`):
            Hidden-state at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden-states of the model at the output of each layer.
        patch_input (`torch.FloatTensor` of shape `(batch_size, num_channels, num_patches, patch_length)`):
            Patched input data to the model.
        loc: (`torch.FloatTensor` of shape `(batch_size, 1, num_channels)`,*optional*):
            Gives the mean of the context window per channel. Used for revin denorm outside the model, if revin
            enabled.
        scale: (`torch.FloatTensor` of shape `(batch_size, 1, num_channels)`,*optional*):
            Gives the std dev of the context window per channel. Used for revin denorm outside the model, if revin
            enabled.
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    patch_input: torch.FloatTensor = None
    loc: Optional[torch.FloatTensor] = None
    scale: Optional[torch.FloatTensor] = None


@add_start_docstrings(
    "The TinyTimeMixer Model for time-series forecasting.",
    TINYTIMEMIXER_START_DOCSTRING,
)
class TinyTimeMixerModel(TinyTimeMixerPreTrainedModel):
    def __init__(self, config: TinyTimeMixerConfig):
        if config.init_processing is False:
            config.check_and_init_preprocessing()

        super().__init__(config)

        self.use_return_dict = config.use_return_dict
        self.encoder = TinyTimeMixerEncoder(config)
        self.patching = TinyTimeMixerPatchify(config)

        if config.scaling == "mean":
            self.scaler = TinyTimeMixerMeanScaler(config)
        elif config.scaling == "std" or config.scaling is True:
            self.scaler = TinyTimeMixerStdScaler(config)
        else:
            self.scaler = TinyTimeMixerNOPScaler(config)

        self.d_model = config.d_model

        # # Initialize weights and apply final processing
        # if config.post_init:
        #     self.post_init()

    @add_start_docstrings_to_model_forward(TINYTIMEMIXER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TinyTimeMixerModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        past_values: torch.Tensor,
        past_observed_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = None,
        freq_token: Optional[torch.Tensor] = None,
    ) -> TinyTimeMixerModelOutput:
        r"""
        past_observed_mask (`torch.Tensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
            Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
            in `[0, 1]` or `[False, True]`:
                - 1 or True for values that are **observed**,
                - 0 or False for values that are **missing** (i.e. NaNs that were replaced by zeros).

        Returns:

        """
        return_dict = return_dict if return_dict is not None else self.use_return_dict

        if past_observed_mask is None:
            past_observed_mask = torch.ones_like(past_values)
        scaled_past_values, loc, scale = self.scaler(past_values, past_observed_mask)

        patched_x = self.patching(scaled_past_values)  # [batch_size x num_input_channels x num_patch x patch_length

        enc_input = patched_x

        encoder_output = self.encoder(
            enc_input,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            freq_token=freq_token,
        )

        if isinstance(encoder_output, tuple):
            encoder_output = TinyTimeMixerEncoderOutput(*encoder_output)

        if not return_dict:
            return tuple(
                v
                for v in [
                    encoder_output.last_hidden_state,
                    encoder_output.hidden_states,
                    patched_x,
                    loc,
                    scale,
                ]
            )

        return TinyTimeMixerModelOutput(
            last_hidden_state=encoder_output.last_hidden_state,
            hidden_states=encoder_output.hidden_states,
            patch_input=patched_x,
            loc=loc,
            scale=scale,
        )


@dataclass
class TinyTimeMixerForPredictionOutput(ModelOutput):
    """
    Output type of [`TinyTimeMixerForPredictionOutput`].

    Args:
        prediction_outputs (`torch.FloatTensor` of shape `(batch_size, prediction_length, num_input_channels)`):
            Prediction output from the forecast head.
        backbone_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_input_channels, num_patches, d_model)`):
            Backbone embeddings before passing through the decoder
        decoder_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_input_channels, num_patches, d_model)`):
            Decoder embeddings before passing through the head.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        loss (*optional*, returned when `y` is provided, `torch.FloatTensor` of shape `()`):
            Total loss.
        loc (`torch.FloatTensor`, *optional* of shape `(batch_size, 1, num_input_channels)`):
            Input mean
        scale (`torch.FloatTensor`, *optional* of shape `(batch_size, 1, num_input_channels)`):
            Input std dev

    """

    loss: Optional[torch.FloatTensor] = None
    prediction_outputs: torch.FloatTensor = None
    backbone_hidden_state: torch.FloatTensor = None
    decoder_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    loc: torch.FloatTensor = None
    scale: torch.FloatTensor = None


@dataclass
class SampleTinyTimeMixerPredictionOutput(ModelOutput):
    """
    Base class for time series model's predictions outputs that contains the sampled values from the chosen
    distribution.

    Args:
        sequences (`torch.FloatTensor` of shape `(batch_size, num_samples, prediction_length, number_channels)`):
            Sampled values from the chosen distribution.
    """

    sequences: torch.FloatTensor = None


def nll(input: torch.distributions.Distribution, target: torch.Tensor) -> torch.Tensor:
    """
    Computes the negative log likelihood loss from input distribution with respect to target.
    """
    return -input.log_prob(target)


def weighted_average(input_tensor: torch.Tensor, weights: Optional[torch.Tensor] = None, dim=None) -> torch.Tensor:
    """
    Computes the weighted average of a given tensor across a given `dim`, masking values associated with weight zero,
    meaning instead of `nan * 0 = nan` you will get `0 * 0 = 0`.

    Args:
        input_tensor (`torch.FloatTensor`):
            Input tensor, of which the average must be computed.
        weights (`torch.FloatTensor`, *optional*):
            Weights tensor, of the same shape as `input_tensor`.
        dim (`int`, *optional*):
            The dim along which to average `input_tensor`.

    Returns:
        `torch.FloatTensor`: The tensor with values averaged along the specified `dim`.
    """
    if weights is not None:
        weighted_tensor = torch.where(weights != 0, input_tensor * weights, torch.zeros_like(input_tensor))
        sum_weights = torch.clamp(weights.sum(dim=dim) if dim else weights.sum(), min=1.0)
        return (weighted_tensor.sum(dim=dim) if dim else weighted_tensor.sum()) / sum_weights
    else:
        return input_tensor.mean(dim=dim)


class TinyTimeMixerForPrediction(TinyTimeMixerPreTrainedModel):
    r"""
    `TinyTimeMixer` for forecasting application.

    Args:
        config (`TinyTimeMixerConfig`, *required*):
            Configuration.

    Returns:
        `None`.
    """

    def __init__(self, config: TinyTimeMixerConfig):
        config.check_and_init_preprocessing()
        super().__init__(config)

        self.config = config

        self.loss = config.loss

        self.use_return_dict = config.use_return_dict

        self.prediction_channel_indices = config.prediction_channel_indices
        self.num_parallel_samples = config.num_parallel_samples

        self.num_input_channels = config.num_input_channels

        self.prediction_filter_length = config.prediction_filter_length

        if config.loss in ["mse", "mae"] or config.loss is None:
            self.distribution_output = None
        elif config.loss == "nll":
            if self.prediction_filter_length is None:
                dim = config.prediction_length
            else:
                dim = config.prediction_filter_length

            distribution_output_map = {
                "student_t": StudentTOutput,
                "normal": NormalOutput,
                "negative_binomial": NegativeBinomialOutput,
            }
            output_class = distribution_output_map.get(config.distribution_output, None)
            if output_class is not None:
                self.distribution_output = output_class(dim=dim)
            else:
                raise ValueError(f"Unknown distribution output {config.distribution_output}")

        self.backbone = TinyTimeMixerModel(config)

        self.use_decoder = config.use_decoder

        if config.use_decoder:
            self.decoder = TinyTimeMixerDecoder(config)

        self.head = TinyTimeMixerForPredictionHead(
            config=config,
            distribution_output=self.distribution_output,
        )

        # Initialize weights and apply final processing
        if config.post_init:
            self.post_init()

    @add_start_docstrings_to_model_forward(TINYTIMEMIXER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TinyTimeMixerForPredictionOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        past_values: torch.Tensor,
        future_values: Optional[torch.Tensor] = None,
        past_observed_mask: Optional[torch.Tensor] = None,
        future_observed_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = False,
        return_loss: bool = True,
        return_dict: Optional[bool] = None,
        freq_token: Optional[torch.Tensor] = None,
        static_categorical_values: Optional[torch.Tensor] = None,
    ) -> TinyTimeMixerForPredictionOutput:
        r"""
        past_observed_mask (`torch.Tensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
            Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
            in `[0, 1]` or `[False, True]`:
                - 1 or True for values that are **observed**,
                - 0 or False for values that are **missing** (i.e. NaNs that were replaced by zeros).
        future_values (`torch.FloatTensor` of shape `(batch_size, target_len, num_input_channels)` for forecasting,:
            `(batch_size, num_targets)` for regression, or `(batch_size,)` for classification, *optional*): Target
            values of the time series, that serve as labels for the model. The `future_values` is what the
            Transformer needs during training to learn to output, given the `past_values`. Note that, this is NOT
            required for a pretraining task.

            For a forecasting task, the shape is be `(batch_size, target_len, num_input_channels)`. Even if we want
            to forecast only specific channels by setting the indices in `prediction_channel_indices` parameter,
            pass the target data with all channels, as channel Filtering for both prediction and target will be
            manually applied before the loss computation.
        future_observed_mask (`torch.Tensor` of shape `(batch_size, prediction_length, num_targets)`, *optional*):
            Boolean mask to indicate which `future_values` were observed and which were missing. Mask values selected
            in `[0, 1]` or `[False, True]`:
                - 1 or True for values that are **observed**,
                - 0 or False for values that are **missing** (i.e. NaNs that were replaced by zeros).
        return_loss (`bool`,  *optional*):
            Whether to return the loss in the `forward` call.
        static_categorical_values (`torch.FloatTensor` of shape `(batch_size, number_of_categorical_variables)`, *optional*):
            Tokenized categorical values can be passed here. Ensure to pass in the same order as the vocab size list used in the
            TinyTimeMixerConfig param `categorical_vocab_size_list`

        Returns:

        """
        if self.loss == "mse":
            loss = nn.MSELoss(reduction="mean")
        elif self.loss == "mae":
            loss = nn.L1Loss(reduction="mean")
        elif self.loss == "nll":
            raise Exception(
                "NLL loss and Distribution heads are currently not allowed. Use mse or mae as loss functions."
            )
            loss = nll
        elif self.loss is None:
            loss = None
        else:
            raise ValueError("Invalid loss function: Allowed values: mse, mae and nll")

        return_dict = return_dict if return_dict is not None else self.use_return_dict

        # past_values: tensor [batch_size x context_length x num_input_channels]
        model_output = self.backbone(
            past_values,
            past_observed_mask=past_observed_mask,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            freq_token=freq_token,
        )  # model_output: [batch_size x nvars x num_patch x d_model]

        if isinstance(model_output, tuple):
            model_output = TinyTimeMixerModelOutput(*model_output)

        decoder_input = model_output.last_hidden_state
        hidden_states = model_output.hidden_states

        if self.use_decoder:
            decoder_output, decoder_hidden_states = self.decoder(
                hidden_state=decoder_input,
                patch_input=model_output.patch_input,
                output_hidden_states=output_hidden_states,
                static_categorical_values=static_categorical_values,
            )  # [batch_size x nvars x num_patch x decoder_d_model]

            if decoder_hidden_states:
                hidden_states.extend(decoder_hidden_states)

        else:
            decoder_output = decoder_input

        # tensor [batch_size x prediction_length x num_input_channels]

        # head should take future mask
        y_hat = self.head(decoder_output, past_values=past_values, future_values=future_values)

        if (
            self.prediction_filter_length is not None
            and future_values is not None
            and future_values.shape[1] != self.prediction_filter_length
        ):
            future_values = future_values[:, : self.prediction_filter_length, :]

            if future_observed_mask is not None:
                future_observed_mask = future_observed_mask[:, : self.prediction_filter_length, :]

        if (
            self.prediction_channel_indices is not None
            and future_values is not None
            and future_values.shape[2] != len(self.prediction_channel_indices)
            and future_values.shape[2] == self.num_input_channels
        ):
            future_values = future_values[..., self.prediction_channel_indices]

            if future_observed_mask is not None:
                future_observed_mask = future_observed_mask[..., self.prediction_channel_indices]

        if self.prediction_channel_indices is not None:
            loc = model_output.loc[..., self.prediction_channel_indices]
            scale = model_output.scale[..., self.prediction_channel_indices]
        else:
            loc = model_output.loc
            scale = model_output.scale
        # loc/scale: batch_size x 1 x prediction_channel_indices or num_targets

        loss_val = None

        if future_observed_mask is not None:
            fut_mask_bool = future_observed_mask.type(torch.bool)

        if self.distribution_output:
            distribution = self.distribution_output.distribution(y_hat, loc=loc, scale=scale)
            if future_values is not None and return_loss is True and loss is not None:
                if future_observed_mask is not None and (~fut_mask_bool).any():
                    if (~fut_mask_bool).all():
                        # no valid observed values
                        print(future_observed_mask)
                        raise ValueError("Loss computation failed due to too many missing values")
                    loss_val = loss(distribution, future_values)
                    # select only values of loss where entire timepoint is observed
                    loss_val = loss_val[fut_mask_bool.all(dim=-1)]
                else:
                    loss_val = loss(distribution, future_values)
                loss_val = weighted_average(loss_val)
        else:
            y_hat = y_hat * scale + loc
            if future_values is not None and return_loss is True and loss is not None:
                if future_observed_mask is not None:
                    loss_val = loss(y_hat[fut_mask_bool], future_values[fut_mask_bool])
                else:
                    # avoiding mask operations for performance benefits on normal scenarios.
                    loss_val = loss(y_hat, future_values)

        if not return_dict:
            return tuple(
                v
                for v in [
                    loss_val,
                    y_hat,
                    model_output.last_hidden_state,
                    decoder_output,
                    hidden_states,
                    loc,
                    scale,
                ]
            )

        return TinyTimeMixerForPredictionOutput(
            loss=loss_val,
            prediction_outputs=y_hat,  # tensor [batch_size x prediction_length x num_input_channels]
            backbone_hidden_state=model_output.last_hidden_state,  # x: [batch_size x nvars x num_patch x d_model]
            decoder_hidden_state=decoder_output,  # x: [batch_size x nvars x num_patch x decoder_d_model]
            hidden_states=hidden_states,
            loc=loc,
            scale=scale,
        )

    def generate(
        self,
        past_values: torch.Tensor,
        past_observed_mask: Optional[torch.Tensor] = None,
    ) -> SampleTinyTimeMixerPredictionOutput:
        """
        Generate sequences of sample predictions from a model with a probability distribution head.

        Args:
            past_values (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                Past values of the time series that serves as context in order to predict the future.

            past_observed_mask (`torch.Tensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
                Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
                in `[0, 1]` or `[False, True]`:
                    - 1 or True for values that are **observed**,
                    - 0 or False for values that are **missing** (i.e. NaNs that were replaced by zeros).

        Return:
            [`SampleTinyTimeMixerPredictionOutput`] where the outputs `sequences` tensor will have shape `(batch_size,
            number of samples, prediction_length, num_input_channels)`.
        """
        # get number of samples
        num_parallel_samples = self.num_parallel_samples

        # get model output
        outputs = self(
            past_values=past_values,
            future_values=None,
            past_observed_mask=past_observed_mask,
            output_hidden_states=False,
        )

        # get distribution

        distribution = self.distribution_output.distribution(
            outputs.prediction_outputs, loc=outputs.loc, scale=outputs.scale
        )

        # get samples: list of [batch_size x prediction_length x num_channels]
        samples = [distribution.sample() for _ in range(num_parallel_samples)]

        # stack tensors
        samples = torch.stack(samples, dim=1)  # [batch_size x num_samples x prediction_length x num_channels]
        return SampleTinyTimeMixerPredictionOutput(sequences=samples)
