"""Utilities for directory operations."""

import os
from pathlib import Path


def resolve_model_path(search_path: str, model_id: str) -> Path:
    """Find the first path under search_path for model_id. All entries in
    search_path must be:
    * an existing directory
    * must be readable by the current process

    Args:
        search_path (str): A unix-like ":" separated list of directories such a "dir1:dir2"
        model_id (str): a model_id (which is really just a subdirectory under dir1 or dir2)

    Returns:
        Path: the first matching path, None if no path is fount.
    """

    _amodeldir_found = next(
        (
            adir
            for adir in (Path(p) for p in search_path.split(":"))
            if adir.exists()
            and adir.is_dir()
            and os.access(adir, os.R_OK)
            and (adir / model_id).exists()
            and os.access(adir / model_id, os.R_OK)
        ),
        None,
    )
    if not _amodeldir_found:
        return None
    return _amodeldir_found / model_id
