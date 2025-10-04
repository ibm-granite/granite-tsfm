# Copyright contributors to the TSFM project
#
# This file implements the model from "FlowState: Sampling Rate Invariant Time Series Forecasting"
# Lars Graf, Thomas Ortner, Stanisław Woźniak and Angeliki Pantazi
#

"""PyTorch FlowState model."""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

from tsfm_public.models.flowstate.configuration_flowstate import FlowStateConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "FlowStateConfig"


FLOWSTATE_PRETRAINED_MODEL_ARCHIVE_LIST = []


FLOWSTATE_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.
    It implements FlowState as presented in "FlowState: Sampling Rate Invariant Time Series Forecasting",
    Lars Graf, Thomas Ortner, Stanisław Woźniak and Angeliki Pantazi

    Parameters:
        config ([`FlowStateConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

FLOWSTATE_INPUTS_DOCSTRING = r"""

    Args:
        past_values (`torch.FloatTensor` of shape `(batch_size, seq_length, num_input_channels)`):
            Context values of the time series. For a forecasting task, this denotes the history/past time series values.
            For univariate time series, `num_input_channels` dimension should be 1. For multivariate time series, it is
            greater than 1.
            If `batch_first=False`, the shape of `past_values` is `(seq_length, batch_size, num_input_channels)`
        batch_first (`bool`):
            Indicates whether the `batch_size` or the `seq_length` is the first dimension of `past_values`.
        scale_factor (`float`):
            The scaling factor to adjust the parameter `Delta` of the S5 block and the Functional Basis Decoder.
        prediction_length (`int`, *optional*):
            Number of time steps to forecast for a forecasting task. Also known as the Forecast Horizon.
            If not provided, or < 0, one forecasting patch is returned.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class FlowStateCausalRevIN(nn.Module):
    def __init__(
        self,
        eps=1e-5,
        with_missing=False,
    ):
        """
        Causal RevIN implementation to enable parallel predictions during training of FlowState

        :param eps: a value added for numerical stability
        :param with_missing (bool): whether contiguous patch masking (CPM) is used or not, interpreting nans as missing values
        """
        super(FlowStateCausalRevIN, self).__init__()
        self.eps = eps
        self.missing = with_missing

    def forward(self, x, mode: str):
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        elif mode == "transform":
            x = self._normalize(x)
        else:
            raise NotImplementedError
        return x

    def _get_statistics(self, x):
        if self.missing:
            n = torch.cumsum(1 - x[..., -1], dim=1).unsqueeze(-1)
            n = torch.where(n == 0, 1.0, n)
        else:
            n = torch.arange(1, x.shape[1] + 1, device=x.device).unsqueeze(-1)
        self.mean = (torch.cumsum(x, dim=1) / n).detach()
        mask = 1.0 if not self.missing else 1 - x[:, :, 1:]
        self.stdev = torch.sqrt(torch.clamp(torch.cumsum(((x - self.mean) * mask) ** 2, 1) / n, min=self.eps)).detach()
        if self.missing:
            self.mean = self.mean[..., :-1]
            self.stdev = self.stdev[..., :-1]

    def set_statistics(self, mean, stdev):
        self.mean = mean
        self.stdev = stdev

    def _normalize(self, x):
        if x.ndim == 4:
            self.stdev = self.stdev[:, -x.shape[0] :].transpose(0, 1).unsqueeze(2)
            self.mean = self.mean[:, -x.shape[0] :].transpose(0, 1).unsqueeze(2)
        if x.shape[-1] == self.mean.shape[-1] + 1:  # with missing and not target
            x[..., :-1] = (x[..., :-1] - self.mean) / self.stdev
            # apply mask again after normalization
            x[..., :-1] *= 1 - x[..., -1].unsqueeze(-1)
        else:
            x = (x - self.mean) / self.stdev
        return x

    def _denormalize(self, x):
        self.stdev = self.stdev[:, -x.shape[0] :].transpose(0, 1).unsqueeze(2)
        self.mean = self.mean[:, -x.shape[0] :].transpose(0, 1).unsqueeze(2)
        if x.ndim == 5:  # quantile predictions
            return x * self.stdev.unsqueeze(-2) + self.mean.unsqueeze(-2)
        x = x * self.stdev

        return x


class FlowStateEmbedding(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        embedding_feature_dim: int,
    ):
        """
        Linear input embedding layer

        Args:
            n_inputs (int): The number of input features or channels
            embedding_feature_dim (int): The embedding dimension
        """
        super(FlowStateEmbedding, self).__init__()
        self.embed = nn.Linear(n_inputs, embedding_feature_dim, bias=True)

    def forward(self, x):
        """
        Args:
            x `torch.FloatTensor` of shape `(seq_length, batch_size, num_input_channels)`: The normalized context time series

        Returns:
            `torch.FloatTensor` of shape `(seq_length, batch_size, embedding_feature_dim)`
        """
        return self.embed(x)


# https://github.com/goroda/PyTorchPoly/blob/master/poly.py adapted for handing batches
def FlowStateLegendreBasis(x, degree):
    """
    Legendre basis functions used in the Functional Basis Decoder.

    Args:
        x (`torch.FloatTensor` of shape `(decoder_dim)`): A batch of discrete time vectors
        degree (`int`): The degree of the polynomial to use

    Returns:
        `torch.FloatTensor` of shape `(batch_size, seq_length, num_input_channels)`: The basis function values at x
    """
    retvar = torch.ones(*x.shape, degree + 1).type(x.type()).to(x.device)
    if retvar.ndim == 3:
        retvar = retvar.permute(1, 2, 0)
        x = x.transpose(0, -1)
    # retvar[:, 0] = x * 0 + 1
    if degree > 0:
        retvar[:, 1] = x
        for ii in range(1, degree):
            retvar[:, ii + 1] = ((2 * ii + 1) * x * retvar[:, ii] - ii * retvar[:, ii - 1]) / (ii + 1)
    if retvar.ndim == 3:
        retvar = retvar.permute(2, 0, 1)
    return retvar


def FlowStateFourierBasis(x, degree):
    """
    Fourier basis functions used in the Functional Basis Decoder.

    Args:
        x (`torch.FloatTensor` of shape `(batch_size, seq_length, num_input_channels)`): A batch of discrete time vectors
        degree (`int`): The degree of the polynomial to use

    Returns:
        `torch.FloatTensor` of shape `(batch_size, seq_length, num_input_channels)`: The basis function values at x
    """
    retvar = torch.ones(*x.shape, degree + 1).type(x.type()).to(x.device)
    if retvar.ndim == 3:
        retvar = retvar.permute(1, 2, 0)
        x = x.transpose(0, -1)
    if degree > 0:
        t2 = torch.einsum("t...,n->tn...", x, 2 * torch.pi * torch.arange(1, degree // 2 + 1).to(x.device))
        retvar[:, 1::2] = torch.sin(t2)
        retvar[:, 2::2] = torch.cos(t2)
    if retvar.ndim == 3:
        retvar = retvar.permute(2, 0, 1)
    return retvar


@dataclass
class FlowStateEncoderOutput(ModelOutput):
    """
    Base class for `FlowStateEncoderOutput`, with potential hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(1, batch_size, encoder_state_dim)`):
            Hidden-state at the output of the last layer of the model.
            These are the outputs of the last S5 layer.
        hidden_states (Tuple[`torch.FloatTensor`], the first of shape `(encoder_state_dim, batch_size, embedding_feature_dim)`,
                                                   the second to the second to last of shape `(encoder_state_dim, batch_size, encoder_state_dim)`,
                                                   and the last of shape `(1, batch_size, encoder_state_dim)`):
            Hidden-states of the model at the output of each layer.
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class FlowStateDecoderOutput(ModelOutput):
    """
    Base class for `FlowStateDecoderOutput`.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(num_channels, batch_size, len(quantiles), prediction_length, 1)`):
            Hidden-state at the output of the decoder.
            These are the final outputs of the decoder after sampling considering the scaling factor.
        hidden_states (Tuple[`torch.FloatTensor`] of one element of shape `(encoder_state_dim, batch_size, embedding_feature_dim)`):
            Hidden-state of the decoder before `sampling`.
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class FlowStateModelOutput(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(num_channels, batch_size, len(quantiles), prediction_length, 1)`):
            Final output of the model, after denormalization
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden-states of the encoder and of the decoder
        embedded_input (`torch.FloatTensor` of shape `(seq_len, batch_size, num_channels)`):
            Inputs of the encoder, result of the embedding layer.
        embedded_output (`torch.FloatTensor` of shape `(seq_len, batch_size, embedding_feature_dim)`):
            Outputs of the encoder, result of the embedding layer.
        backbone_hidden_state (`torch.FloatTensor` of shape `(1, batch_size, encoder_state_dim)`):
            Last hidden state at the output of the backbone before passing through the decoder
        decoder_hidden_state (`torch.FloatTensor` of shape `(num_channels, batch_size, len(quantiles), prediction_length, 1)`):
            Last hidden state of the decoder embeddings.
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    embedded_input: torch.FloatTensor = None
    embedded_output: torch.FloatTensor = None
    backbone_hidden_state: torch.FloatTensor = None
    decoder_hidden_state: torch.FloatTensor = None


class FlowStateS5Block(nn.Module):
    def __init__(self, config, last=False):
        super().__init__()
        self.last = last
        self.config = config
        self.init_params()

    def init_params(self):
        with torch.no_grad():
            state_dim = self.config.encoder_state_dim
            H = self.config.embedding_feature_dim
            encoder_num_hippo_blocks = self.config.encoder_num_hippo_blocks
            if state_dim % encoder_num_hippo_blocks != 0:
                raise ValueError("encoder_state_dim has to be divisible by encoder_num_hippo_blocks.")
            block_size = int(state_dim / encoder_num_hippo_blocks)
            # blockwise Hippo-N diagonalized
            n = torch.sqrt(2 * (torch.arange(block_size) + 1.0) + 1)
            A = -torch.outer(n, n) / 2
            A = -0.5 * torch.eye(block_size) + torch.triu(A) - torch.tril(A)
            Lambda, V = torch.linalg.eig(torch.block_diag(*encoder_num_hippo_blocks * [A]))
            self.log_Lambda_real = torch.nn.Parameter(torch.log(-Lambda.real))
            self.Lambda_imag = torch.nn.Parameter(Lambda.imag)
            B = 0.5 / torch.sqrt(torch.tensor(H)) * (torch.randn(state_dim, H) + torch.randn(state_dim, H) * 1.0j)
            C = (
                0.5
                / torch.sqrt(torch.tensor(state_dim))
                * (torch.randn(H, state_dim) + torch.randn(H, state_dim) * 1.0j)
            )
            self.B_tilde_r = nn.Parameter((V.inverse() @ B).real)
            self.B_tilde_i = nn.Parameter((V.inverse() @ B).imag)
            self.C_tilde_r = nn.Parameter((C @ V).real)
            self.C_tilde_i = nn.Parameter((C @ V).imag)
            self.D = nn.Parameter(torch.randn(H))
            log_min, log_max = torch.log(torch.tensor(0.001)), torch.log(torch.tensor(0.1))
            self.log_Delta = nn.Parameter(log_min + (log_max - log_min) * torch.rand(state_dim))

    def get_discretized(self, L=None, scale_factor=1.0):
        """Discretize a diagonalized, continuous-time linear SSM
        Args:
        L (int): length of the sequence
        scale_factor (float32): mult factor for discretization step sizes (scalar, or per el in batch)
        Returns:
        discretized kernel (complex64), B_bar (complex64) (P,), (P,H)"""
        if L is None:
            L = int((self.config.context_length + 1 - self.config.patch) / self.config.stride - 1e-9) + 1
        lambda_ = -torch.exp(self.log_Lambda_real) + 1j * self.Lambda_imag
        device = lambda_.device
        Identity = torch.ones(lambda_.shape[0], device=device)
        B_tilde = self.B_tilde_r + 1.0j * self.B_tilde_i
        if type(scale_factor) is torch.Tensor and scale_factor.ndim > 0:  # different scale factor per sample
            log_Lambda_bar = torch.outer(scale_factor, lambda_ * torch.exp(self.log_Delta))
            kernel = torch.einsum("bd,L->bLd", log_Lambda_bar, torch.arange(L - 1, -1, -1, device=device)).exp()
            B_bar = (1 / lambda_ * (kernel[:, -2] - 1.0))[..., None] * B_tilde
        else:
            log_Lambda_bar = scale_factor * lambda_ * torch.exp(self.log_Delta)
            kernel = torch.multiply(
                log_Lambda_bar.unsqueeze(0), torch.arange(L - 1, -1, -1, device=device).unsqueeze(-1)
            ).exp()
            B_bar = (1 / lambda_ * (kernel[-2] - Identity))[..., None] * B_tilde

        return kernel, B_bar

    def apply_ssm_kern_ff(self, Bu_elements, kernel):
        """Compute the LxBxH output of discretized SSM given an LxBxH input.
        Args:
        Bu_elements (float32): projected input sequence of features (L, B, H)
        Lambda_bar (float32): discretized
        Returns:
        ys (float32): the SSM outputs (S5 layer preactivations) (L, H)"""
        l, b, d = Bu_elements.shape
        if kernel.ndim == 2:
            kernel = kernel.unsqueeze(0)
        kff = torch.fft.fft(kernel.transpose(0, 1).flip(dims=(0,)), n=2 * l, dim=0)
        buff = torch.fft.fft(Bu_elements, n=2 * l, dim=0)
        o = torch.fft.ifft(kff * buff, n=2 * l, dim=0)[:l]

        if self.last and self.config.min_context == Bu_elements.shape[0]:  # only last hidden state will be used
            return o[-1].unsqueeze(0)
        elif self.last:
            return o[min(self.config.min_context, o.shape[0]) - 1 :]
        else:
            return o

    def forward(self, input_sequences, scale_factor=1.0):
        """Computes LxBxH output sequence of an S5 layer given LxBxH input sequence.
        Args:
        params: tuple of the continuous time SSM parameters
        input_sequences: batch of input feature sequences (L, B ,H)
        Returns:
        Batch of S5 layer output sequences (L, B, H)"""
        input_sequences = input_sequences + 0.0j
        kernel, B_bar = self.get_discretized(L=input_sequences.shape[0], scale_factor=scale_factor)
        if kernel.ndim == 3:
            Bu_elements = torch.einsum("bnm,lbm->lbn", B_bar, input_sequences)
        else:
            Bu_elements = torch.einsum("nm,lbm->lbn", B_bar, input_sequences)
        xs = self.apply_ssm_kern_ff(Bu_elements, kernel)
        if self.last:
            input_sequences = input_sequences[min(self.config.min_context, input_sequences.shape[0]) - 1 :]
        # Compute SSM output sequence
        xs = torch.einsum("hn,...bn->...bh", self.C_tilde_r + 1.0j * self.C_tilde_i, xs)
        xs += torch.einsum("h,...bh->...bh", self.D, input_sequences.real)
        return xs.real


class FlowStateS5Layer(nn.Module):
    def __init__(self, config, last=False, ssm=True):
        super().__init__()
        self.config = config
        n = config.embedding_feature_dim
        self.last = last
        self.ssm = FlowStateS5Block(config, last=last)
        self.out = nn.Linear(n, n)
        self.norm = nn.LayerNorm(n)

    def forward(self, x, scale_factor=1.0):
        skip = (
            x if (not self.last or x.ndim == 2) else x[min(self.config.min_context, x.shape[0]) - 1 :]
        )  # last layer doesn't need MLP on all timesteps
        # SSM
        x = self.ssm(x, scale_factor=scale_factor)

        # self gated MLP
        x = F.selu(x)
        x = x * F.sigmoid(self.out(x))

        # pre layernorm
        x = self.norm(x)
        x = skip + x
        return x


class FlowStatePreTrainedModel(PreTrainedModel):
    # Weight initialization
    config_class = FlowStateConfig
    base_model_prefix = "model"
    main_input_name = "past_values"
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        """Initialize weights"""

        # print("Should not reach here, all parameters should have been initialized!")
        # For training, here would be place to initialize the parameters of FlowState


class FlowStateEncoder(FlowStatePreTrainedModel):
    """
    Encoder for FlowState which inputs time-series and outputs embeddings.

    Args:
        config (`FlowStateConfig`, *required*):
            Configuration.
    """

    def __init__(self, config: FlowStateConfig):
        if config.init_processing is False:
            config.check_and_init_preprocessing()

        super().__init__(config)

        self.use_return_dict = config.use_return_dict

        self.layers = nn.ModuleList(
            [
                FlowStateS5Layer(config, last=(i == config.encoder_num_layers - 1), ssm=True)
                for i in range(config.encoder_num_layers)
            ]
        )

    @replace_return_docstrings(output_type=FlowStateEncoderOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        encoder_inputs: torch.Tensor,
        scale_factor: Optional[float] = 1.0,
    ) -> Union[Tuple, FlowStateEncoderOutput]:
        r"""
        Args:
            past_values (`torch.FloatTensor` of shape `(seq_len, batch_size, embedding_feature_dim)`):
                Context values of the time series.
                For univariate time series, `num_input_channels` dimension should be 1. For multivariate time series,
                it is greater than 1.

        Returns:
            `torch.FloatTensor` of shape `(1, batch_size, encoder_state_dim)`
        """
        x = encoder_inputs
        output = x

        all_hidden_states = []

        # Encoder
        for _, lay in enumerate(self.layers):
            output = lay(output, scale_factor=scale_factor)
            all_hidden_states.append(output)

        return FlowStateEncoderOutput(last_hidden_state=output, hidden_states=all_hidden_states)


class FlowStateFunctionalBasisDecoder(FlowStatePreTrainedModel):
    """
    This is the Functional Basis Decoder (FBD) of FlowState.

    Args:
        config (`FlowStateConfig`, *required*):
            Configuration.
    """

    def __init__(self, config):
        if config.init_processing is False:
            config.check_and_init_preprocessing()

        super().__init__(config)

        n_out = config.decoder_dim
        if "legs" in config.decoder_type.lower():
            self.range = [-1.0, 0.95]
            if config.decoder_type.lower() == "hlegs":
                self.range = [0.0, 0.95]
            self.basis_f = lambda t: FlowStateLegendreBasis(t, n_out - 1)
        elif config.decoder_type.lower() == "four":
            if n_out % 2 == 0:
                raise ValueError("Fourier decoder must have odd dimension.")
            self.range = [0.0, 1.0]
            self.basis_f = lambda t: FlowStateFourierBasis(t, n_out - 1)
        else:
            raise ValueError("Unknown decoder" + str(config.decoder_type))
        n_lin = n_out * len(config.quantiles)
        self.pred_dist = config.decoder_patch_len
        n = config.embedding_feature_dim
        self.config = config
        self.lin = nn.Linear(n, n_lin)

    def get_t(self, scale, pred_dim, device):
        dt = scale * (self.range[1] - self.range[0]) / self.pred_dist
        t = self.range[0] + torch.arange(1, pred_dim + 1, dtype=torch.float, device=device) * dt
        return t

    def get_kernel(self, sampling_factor, target_points, device):
        with torch.no_grad():
            if type(sampling_factor) is torch.Tensor and sampling_factor.ndim > 0:
                # individual factor per sample
                t = torch.stack([self.get_t(sf, target_points) for sf in sampling_factor], dim=0).to(device)
            else:
                t = self.get_t(sampling_factor, target_points, device)

            t = torch.clamp(t, min=self.range[0])
            f = self.basis_f(t)

        return f.float() / 4.0

    @replace_return_docstrings(output_type=FlowStateDecoderOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        encoder_output: FlowStateEncoderOutput,
        prediction_length: int,
        scale_factor: Optional[float] = 1.0,
    ) -> FlowStateDecoderOutput:
        """
        The FBD receives the `encoder_output` (final output of the encoder) a prediction_length and the `scale_factor`.
        It first linearly encodes the `encoder_output`, then samples the continuous basis functions based on `prediction_length`
        and `scale_factor` producing a set of configured discrete basis functions and
        finally applies the linearly encoded `encoder_output` to the discrete basis function by matrix multiplication
        to produce the (time-)scaled prediction for the Forecasting Horizon.

        Args:
            encoder_output (FlowStateEncoderOutput): The output of the encoder.
            prediction_length (int): Number of time steps to forecast for a forecasting task. Also known as the Forecast Horizon.
            scale_factor (Optional[float], optional): The scaling factor to adjust the parameter `Delta` of the S5 block and the Functional Basis Decoder, defaults to 1.

        Returns:
            FlowStateDecoderOutput: The final outputs of the decoder.
        """

        if prediction_length is None or scale_factor is None:
            raise ValueError("Provide valid scale factor and Nr. of target points")

        values = self.lin(encoder_output.last_hidden_state)
        hidden_state = values.view((*values.shape[:-1], -1, self.config.decoder_dim))
        W = self.get_kernel(scale_factor, prediction_length, hidden_state.device)
        if W.ndim == 2:
            return FlowStateDecoderOutput(
                last_hidden_state=torch.einsum("...h,ph->...p", hidden_state, W).unsqueeze(-1),
                hidden_states=[hidden_state],
            )
        else:
            return FlowStateDecoderOutput(
                last_hidden_state=torch.einsum("...bqh,bph->...bqp", hidden_state, W).unsqueeze(-1),
                hidden_states=[hidden_state],
            )


@add_start_docstrings(
    "The FlowState Model for time-series forecasting.",
    FLOWSTATE_START_DOCSTRING,
)
class FlowStateModel(FlowStatePreTrainedModel):
    def __init__(self, config: FlowStateConfig):
        if config.init_processing is False:
            config.check_and_init_preprocessing()

        super().__init__(config)

        self.config = config

        n_inputs = 1
        self.norm = FlowStateCausalRevIN(with_missing=config.with_missing)

        n_inputs = 2 if config.with_missing else 1
        self.embed = FlowStateEmbedding(n_inputs, config.embedding_feature_dim)

        self.encoder = FlowStateEncoder(config)

        self.decoder = FlowStateFunctionalBasisDecoder(config)

        trainable_paras = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        decoder_paras = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        logger.info(f"Number of encoder parameters: {trainable_paras * 1e-3}k")
        logger.info(
            f"Number of dencoder parameters: {decoder_paras * 1e-3}k ({100 * decoder_paras / trainable_paras:.2f}%)"
        )

    @add_start_docstrings_to_model_forward(FLOWSTATE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlowStateModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        past_values: torch.Tensor,
        return_dict: Optional[bool] = None,
        scale_factor: Optional[float] = None,
        batch_first: Optional[bool] = None,
        mask_n: Optional[int] = None,
    ) -> FlowStateModelOutput:
        r"""
        past_values (`torch.FloatTensor` of shape `(batch_size, seq_length, num_input_channels)`):
            Context values of the time series. For a forecasting task, this denotes the history/past time series values.
            For univariate time series, `num_input_channels` dimension should be 1. So far only univariate forecasts are supported.
            If `batch_first=False`, the shape of `past_values` is `(seq_length, batch_size, num_input_channels)`
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        scale_factor (`float`):
            The scaling factor to adjust the parameter `Delta` of the S5 block and the Functional Basis Decoder
        batch_first (`bool`):
            Indicates whether the `batch_size` or the `seq_length` is the first dimension of `past_values`.
        mask_n (`int`, *optional*):
            When contiguous patch masking (CPM) is used during prediction, the `mask_n` indicates how many
            elements of `past_values` should be treated as unknown.

        Returns:
            FlowStateModelOutput: The final denormalized prediction of FlowState.

        """
        if batch_first is None:
            batch_first = self.config.batch_first
        if scale_factor is None:
            scale_factor = self.config.scale_factor
        if mask_n is None:
            mask_n = 0

        if past_values.dim() != 3:
            raise ValueError(
                "`past_values` must have 3 dimensions of shape `(sequence_length, batch_size, num_input_channels)`."
            )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if batch_first:
            time_first_past_values = torch.transpose(past_values, 1, 0)
        else:
            time_first_past_values = past_values

        L, batch, n_ch = time_first_past_values.shape
        if n_ch > 1:
            raise RuntimeError("past_values may only contain a single variate / channel.")

        if self.config.with_missing:
            # if input contains nans, fill as missing
            mask = torch.where(time_first_past_values.isnan(), torch.ones_like(time_first_past_values), 0)
            time_first_past_values = torch.nan_to_num(time_first_past_values, 0.0)
            time_first_past_values = torch.cat((time_first_past_values, mask), dim=-1)
            if mask_n > 0:
                apdx = torch.cat((torch.zeros(mask_n, batch, n_ch), torch.ones(mask_n, batch, 1)), dim=-1).to(
                    past_values.device
                )
                time_first_past_values = torch.cat((time_first_past_values, apdx), dim=0)

        # Normalize the inputs
        time_first_past_values = self.norm(time_first_past_values.transpose(0, 1), "norm").transpose(0, 1)

        # Emebd the inputs
        encoder_inputs = self.embed(time_first_past_values)

        # Execute encoder
        encoder_output = self.encoder(encoder_inputs, scale_factor=scale_factor)

        if isinstance(encoder_output, tuple):
            encoder_output = FlowStateEncoderOutput(*encoder_output)

        # Execute decoder
        decoder_output = self.decoder(
            encoder_output=encoder_output,
            prediction_length=int(self.config.decoder_patch_len / scale_factor),
            scale_factor=scale_factor,
        )

        if isinstance(decoder_output, tuple):
            encoder_output = FlowStateDecoderOutput(*decoder_output)

        # denorm during evaluation
        pred = self.norm(decoder_output.last_hidden_state, "denorm")

        if not return_dict:
            return tuple(
                v
                for v in [
                    pred,
                    encoder_output.hidden_states + decoder_output.hidden_states,
                    past_values,
                    encoder_inputs,
                    encoder_output.last_hidden_state,
                    decoder_output.last_hidden_state,
                ]
            )

        return FlowStateModelOutput(
            last_hidden_state=pred,
            hidden_states=encoder_output.hidden_states + decoder_output.hidden_states,
            embedded_input=past_values,
            embedded_output=encoder_inputs,
            backbone_hidden_state=encoder_output.last_hidden_state,
            decoder_hidden_state=decoder_output.last_hidden_state,
        )


@dataclass
class FlowStateForPredictionOutput(ModelOutput):
    """
    Output type of [`FlowStateForPredictionOutput`].

    Args:
        loss (*optional*, returned when `y` is provided, `torch.FloatTensor` of shape `()`):
            Total loss.
        prediction_outputs (`torch.FloatTensor` of shape `(num_channels, batch_size, len(quantiles), prediction_length, 1)`):
            Prediction output from FlowState model.
        backbone_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_input_channels, num_patches, d_model)`):
            Last hidden state at the output of the backbone before passing through the decoder
        decoder_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_input_channels, num_patches, d_model)`):
            Last hidden state of the decoder embeddings.
    """

    loss: Optional[torch.FloatTensor] = None
    prediction_outputs: torch.FloatTensor = None
    backbone_hidden_state: torch.FloatTensor = None
    decoder_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class FlowStateForPrediction(FlowStatePreTrainedModel):
    r"""
    `FlowState` for forecasting application.

    Args:
        config (`FlowStateConfig`, *required*):
            Configuration.
    """

    def __init__(self, config: FlowStateConfig):
        config.check_and_init_preprocessing()
        super().__init__(config)

        self.config = config

        self.model = FlowStateModel(config)

    def _combine_cpm_predictions(self, pred, pred_len):
        n_preds, batch, quants, fl, n_ch = pred.shape
        rest = pred[-1]
        pred = torch.cat([pred[ix] for ix in range(0, n_preds, fl)], dim=2)
        if pred.shape[2] < pred_len:
            pred = torch.cat((pred, rest[:, :, -(pred_len - pred.shape[2]) :]), dim=2)
        pred = pred[:, :, :pred_len]
        return pred

    @add_start_docstrings_to_model_forward(FLOWSTATE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlowStateForPredictionOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        past_values: torch.Tensor,
        future_values: Optional[torch.Tensor] = None,
        past_observed_mask: Optional[torch.Tensor] = None,
        future_observed_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_loss: bool = True,
        return_dict: Optional[bool] = None,
        batch_first: Optional[bool] = None,
        scale_factor: Optional[float] = None,
        prediction_length: Optional[int] = None,
        prediction_type: Optional[bool] = None,
    ) -> FlowStateForPredictionOutput:
        r"""
        past_values (`torch.FloatTensor` of shape `(batch_size, seq_length, num_input_channels)`):
            Context values of the time series. For a forecasting task, this denotes the history/past time series values.
            For univariate time series, `num_input_channels` dimension should be 1. For multivariate time series, it is
            greater than 1.
            If `batch_first=False`, the shape of `past_values` is `(seq_length, batch_size, num_input_channels)`
        future_values: (`torch.FloatTensor`, *optional*):
            currently not used.
        past_observed_mask: (`torch.FloatTensor`, *optional*):
            currently not used.
        future_observed_mask: (`torch.FloatTensor`, *optional*):
            currently not used.
        output_hidden_states: (`bool`, *optional*):
            currently not used.
        return_loss: (`bool`, *optional*):
            currently not used.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        batch_first (`bool`, *optional*):
            Indicates whether the `batch_size` or the `seq_length` is the first dimension of `past_values`.
        scale_factor (`float`, *optional*):
            The scaling factor to adjust the parameter `Delta` of the S5 block and the Functional Basis Decoder
        prediction_length (`int`, *optional*):
            Number of time steps to forecast for a forecasting task. Also known as the Forecast Horizon.
            If not provided, or < 0, one forecasting patch is returned.
        prediction_type (`str`, *optional*):
            Indicates the desired return type of the model. Can be any either:
            quantile: The predictions for all predicted quantiles is returned
            mean: The mean of the predicted quantiles is returned
            median: The median of the predicted quantiles is returned

        Returns:
            FlowStateModelOutput: The final denormalized prediction of FlowState.
        """
        if batch_first is None:
            batch_first = self.config.batch_first
        if scale_factor is None:
            scale_factor = self.config.scale_factor
        if prediction_length is None:
            prediction_length = self.config.prediction_length
        if prediction_type is None:
            prediction_type = self.config.prediction_type

        if past_values.dim() != 3:
            raise ValueError(
                "`past_values` must have 3 dimensions of shape `(sequence_length, batch_size, num_input_channels)`."
            )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        max_context = min(8 * 1024, int(self.config.context_length / scale_factor))
        max_decoder_patch_len = int(self.config.decoder_patch_len / scale_factor + 1e-6)
        if prediction_length == -1:
            prediction_length = max_decoder_patch_len
        # prepare multi patch inferencing
        mask_n = max(0, prediction_length - max_decoder_patch_len)
        max_context -= mask_n
        if batch_first:
            past_values = past_values[:, -max_context:]
        else:
            past_values = past_values[-max_context:]
        if mask_n > 0:
            self.model.config.min_context = (
                past_values.shape[1] if batch_first else past_values.shape[0]
            )  # min context from which to start predicting
        else:
            self.model.config.min_context = 0
        # past_values: tensor [batch_size x seq_length x num_input_channels], or [seq_length x batch_size x num_input_channels]
        model_output = self.model(
            past_values,
            return_dict=return_dict,
            batch_first=batch_first,
            scale_factor=scale_factor,
            mask_n=mask_n,
        )

        if isinstance(model_output, tuple):
            model_output = FlowStateModelOutput(*model_output)
        model_output.last_hidden_state = self._combine_cpm_predictions(
            model_output.last_hidden_state, prediction_length
        )

        if self.config.prediction_type == "quantile":
            pass
        elif self.config.prediction_type == "mean":
            # calculate an approximate mean from quantiles
            quant_prob = 0.5 - (0.5 - torch.tensor(self.config.quantiles)).abs()
            quant_prob /= quant_prob.sum()  # normalize quantile weights
            quant_prob = quant_prob.view(1, -1, 1, 1).to(past_values.device)
            model_output.last_hidden_state = (model_output.last_hidden_state * quant_prob).sum(dim=1)
        elif self.config.prediction_type == "median":
            model_output.last_hidden_state = model_output.last_hidden_state[:, 4, :]
        else:
            raise RuntimeError("Unknown prediction_type detected. Should be one of ['quantile', 'mean', 'median']")

        loss_val = None

        if not return_dict:
            return tuple(
                v
                for v in [
                    loss_val,
                    model_output.last_hidden_state,
                    model_output.backbone_hidden_state,
                    model_output.decoder_hidden_state,
                    model_output.hidden_states,
                ]
            )

        return FlowStateForPredictionOutput(
            loss=loss_val,
            prediction_outputs=model_output.last_hidden_state,  # tensor [batch_size x prediction_length x num_input_channels]
            backbone_hidden_state=model_output.backbone_hidden_state,
            decoder_hidden_state=model_output.decoder_hidden_state,
            hidden_states=model_output.hidden_states,
        )
