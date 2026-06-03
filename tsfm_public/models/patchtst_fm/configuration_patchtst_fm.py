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
        d_model: int = 384,
        n_head: int = 6,
        n_layer: int = 6,
        norm_first: bool = True,
        pretrain_mask_ratio: float = 0.4,
        pretrain_mask_cont: int = 8,
        num_quantile: int = 99,
        use_pruning: bool = True,
        **kwargs,
    ):
        """Configuration for PatchTST-FM.

        Args:
            context_length (int, optional): Number of timesteps for the context, includes both historical and predicted
                timesteps. Defaults to 8192.
            prediction_length (int, optional): Number of timesteps in the prediction output. Defaults to 64.
            d_patch (int, optional): Size of patches. Defaults to 16. Context length must be divisible by d_patch.
            d_model (int, optional): Size of patches after projection before feeding into the transformer layers. Defaults to
                384.
            n_head (int, optional): Number of attention heads. Defaults to 6.
            n_layer (int, optional): Number of transformer layers. Defaults to 6.
            norm_first (bool, optional): If true, normalization is done first in the transformer layers. Defaults to True.
            pretrain_mask_ratio (float, optional): _description_. Defaults to 0.4.
            pretrain_mask_cont (int, optional): Number of contiguous patches to mask during pretraining. Defaults to 8.
            num_quantile (int, optional): Number of quantiles to include in the quantile outputs. Defaults to 99.
            use_pruning (bool, optional): When the input tensor has context less than the configured context_length, it will
                be padded. The leading padding can be pruned to reduce unneccessary computation if value of the earlier
                indices are not needed. Defaults to True.
        """
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.d_patch = d_patch
        self.n_patch = int(context_length // d_patch)
        self.d_model = d_model
        self.n_head = n_head
        self.n_layer = n_layer
        self.norm_first = norm_first
        self.pretrain_mask_ratio = pretrain_mask_ratio
        self.pretrain_mask_cont = pretrain_mask_cont
        self.num_quantile = num_quantile

        if num_quantile % 9 == 0:
            quantiles = [i / (self.num_quantile + 1) for i in range(1, self.num_quantile + 1)]
        else:
            quantiles = [i / (self.num_quantile - 1) for i in range(1, self.num_quantile - 1)]
            quantiles = [0.01] + quantiles + [0.99]
        self.quantile_levels = quantiles
        # self.quantiles = quantiles
        super().__init__(**kwargs)
