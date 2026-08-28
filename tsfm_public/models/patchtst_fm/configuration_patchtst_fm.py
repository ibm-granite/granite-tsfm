# Copyright contributors to the TSFM project
#
"""PatchTST-FM model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)

PATCHTSTFM_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class PatchTSTFMConfig(PretrainedConfig):
    model_type = "patchtst_fm"
    attribute_map = {
        "hidden_size": "d_model",
        "num_hidden_layers": "n_layer",
    }

    def __init__(
        self,
        context_length: int = 8192,
        prediction_length: int = 64,
        d_patch: int = 16,
        patch_stride: int = None,  # None disables striding and windowing
        patch_loss_windowing: str = None,  # "rectangular", "triangular", "hamming"; must not be None is striding is active
        d_model: int = 384,
        n_head: int = 6,
        n_layer: int = 6,
        norm_first: bool = True,
        qkv_bias: bool = True,
        prehead_norm: bool = False,
        pretrain_mask_ratio: float = 0.4,
        pretrain_mask_cont: int = 8,
        randomize_cpm: bool = False,
        num_quantile: int = 99,
        block_type: str = "transformer",  # "transformer", "conformer"
        conv_kernel_size: int | list = 3,
        is_causal: bool = False,
        mlp_type: str = "mlp",  # "mlp" or "swiglu"
        use_pruning: bool = True,
        **kwargs,
    ):
        """Configuration for PatchTST-FM.

        Args:
            context_length (int, optional): Number of timesteps for the context, includes both historical and predicted
                timesteps. Defaults to 8192.
            prediction_length (int, optional): Number of timesteps in the prediction output. Defaults to 64.
            d_patch (int, optional): Size of patches. Defaults to 16. Context length must be divisible by d_patch.
            patch_stride (int, optional): Stride of patches. Defaults to None, which indicates no striding.
            patch_loss_windowing (str, optional): Specifies how patches are weighted during loss and combination, used
                when striding. Must be one of "rectangular", "triangular", "hamming" when striding.
            d_model (int, optional): Size of patches after projection before feeding into the transformer layers. Defaults to
                384.
            n_head (int, optional): Number of attention heads. Defaults to 6.
            n_layer (int, optional): Number of transformer layers. Defaults to 6.
            norm_first (bool, optional): If true, normalization is done first in the transformer layers. Defaults to True.
            qkv_bias (bool): Indicates if bias should be enabled in linear layers associated with attention qkv calculations.
                Defaults to True.
            prehead_norm (bool): Indicates if an additional norm lauer should be addded prior to the quantile head after the
                backbone layers.
            pretrain_mask_ratio (float, optional): _description_. Defaults to 0.4.
            pretrain_mask_cont (int, optional): Number of contiguous patches to mask during pretraining. Defaults to 8.
            randomize_cpm (bool): Pretraining flag that indicates when increased randomization of continuous patch masking is
                enabled. Defaults to False.
            num_quantile (int, optional): Number of quantiles to include in the quantile outputs. Defaults to 99.
            block_type (str, optional): our pipeline supports multiple block types "transformer" and "conformer"
            conv_kernel_size: (int | list, optional): specifies kernel size within the conformer convolution sublayer. This can be either
                a single int (applies to all blocks uniformly) or a list of integeres, in which case each value applies to
                the corresponding block starting from lowest up to n_layer. Note that values<=0 will insert a transformer
                block (instead of conformer). Example: conv_kernel_size=[5, 3, 0] will configure three blocks as follows:
                conformer(size 5)->conformer(size 3)->transformer
            is_causal (bool, optional): if True, the convolution kernel within the convolution sublayer will be applied to the left
                of current slot, as opposed to symmetrically. In this setting, even-sized kernel sizes are allowed.
            mlp_type (str, optional): Can be "mlp" or "swiglu" - type of FFN within conformer block.
            use_pruning (bool, optional): When the input tensor has context less than the configured context_length, it will
                be padded. The leading padding can be pruned to reduce unneccessary computation if value of the earlier
                indices are not needed. Defaults to True.
        """
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.d_patch = d_patch
        self.patch_stride = patch_stride
        self.patch_loss_windowing = patch_loss_windowing
        self.n_patch = (
            (context_length - d_patch) // self.patch_stride + 1
            if self.patch_stride is not None
            else int(context_length // d_patch)
        )
        self.d_model = d_model
        self.n_head = n_head
        self.n_layer = n_layer
        self.norm_first = norm_first
        self.qkv_bias = qkv_bias
        self.prehead_norm = prehead_norm
        self.pretrain_mask_ratio = pretrain_mask_ratio
        self.pretrain_mask_cont = pretrain_mask_cont
        self.randomize_cpm = randomize_cpm
        self.num_quantile = num_quantile
        self.block_type = block_type
        self.conv_kernel_size = conv_kernel_size
        self.is_causal = is_causal
        self.mlp_type = mlp_type
        self.use_pruning = use_pruning

        if num_quantile % 9 == 0:
            quantiles = [i / (self.num_quantile + 1) for i in range(1, self.num_quantile + 1)]
        else:
            quantiles = [i / (self.num_quantile - 1) for i in range(1, self.num_quantile - 1)]
            quantiles = [0.01] + quantiles + [0.99]
        self.quantile_levels = quantiles
        # self.quantiles = quantiles
        super().__init__(**kwargs)
