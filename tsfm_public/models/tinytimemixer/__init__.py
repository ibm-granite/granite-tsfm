# Copyright contributors to the TSFM project
#

from typing import TYPE_CHECKING

# rely on isort to merge the imports
from transformers.utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
)


_import_structure = {
    "configuration_tinytimemixer": [
        "TINYTIMEMIXER_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "TinyTimeMixerConfig",
    ],
}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tinytimemixer"] = [
        "TINYTIMEMIXER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TinyTimeMixerModel",
        "TinyTimeMixerForPrediction",
    ]

    _import_structure["utils_tinytimemixer"] = [
        "get_freq_mapping",
        "get_freq_token",
        "get_freq_vocab_size",
    ]


if TYPE_CHECKING:
    from .configuration_tinytimemixer import (
        TINYTIMEMIXER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        TinyTimeMixerConfig,
    )

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_tinytimemixer import (
            TINYTIMEMIXER_PRETRAINED_MODEL_ARCHIVE_LIST,
            TinyTimeMixerForPrediction,
            TinyTimeMixerModel,
        )
        from .utils_tinytimemixer import (
            get_freq_mapping,
            get_freq_token,
            get_freq_vocab_size,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
