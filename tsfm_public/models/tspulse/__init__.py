from typing import TYPE_CHECKING

# rely on isort to merge the imports
from transformers.utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
)


_import_structure = {
    "configuration_tspulse": [
        "TSPULSE_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "TSPulseConfig",
    ],
}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tspulse"] = [
        "TSPULSE_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TSPulseModel",
        "TSPulseForReconstruction",
        "TSPulseForClassification",
        "TSPulsePreTrainedModel",
    ]


if TYPE_CHECKING:
    from .configuration_tspulse import (
        TSPULSE_PRETRAINED_CONFIG_ARCHIVE_MAP,
        TSPulseConfig,
    )

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_tspulse import (
            TSPULSE_PRETRAINED_MODEL_ARCHIVE_LIST,
            TSPulseForClassification,
            TSPulseForReconstruction,
            TSPulseModel,
            TSPulsePreTrainedModel,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
