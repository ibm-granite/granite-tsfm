# Copyright contributors to the TSFM project
#

try:
    # Local
    from ._version import __version__, __version_tuple__  # noqa: F401 # unused import
except ImportError:
    __version__ = "unknown"
    __version_tuple__ = (0, 0, __version__)
