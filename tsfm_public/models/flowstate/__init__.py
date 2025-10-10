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
    "configuration_flowstate": [
        "FLOWSTATE_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "FlowStateConfig",
    ],
}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_flowstate"] = [
        "FLOWSTATE_PRETRAINED_MODEL_ARCHIVE_LIST",
        "FlowStateModel",
        "FlowStateForPrediction",
    ]

    _import_structure["utils_utils"] = ["get_fixed_factor"]

if TYPE_CHECKING:
    from .configuration_flowstate import (
        FLOWSTATE_PRETRAINED_CONFIG_ARCHIVE_MAP,
        FlowStateConfig,
    )

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_flowstate import (
            FLOWSTATE_PRETRAINED_MODEL_ARCHIVE_LIST,
            FlowStateForPrediction,
            FlowStateModel,
        )
        from .utils.utils import get_fixed_factor

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
