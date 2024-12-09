import os
import tempfile
from pathlib import Path

from tsfminference.dirutil import resolve_model_path


def test_resolve_model_path():
    with tempfile.TemporaryDirectory() as dir1:
        with tempfile.TemporaryDirectory() as dir2:
            dirpath = f"{dir1}:{dir2}"
            os.mkdir(Path(dir1) / "amodel")
            assert resolve_model_path(dirpath, "amodel") == Path(dir1) / "amodel"
            assert resolve_model_path(dirpath, "foobar") is None
            assert resolve_model_path("fzbatt:zap", "amodel") is None
            os.mkdir(Path(dir2) / "anewmodel")
            assert resolve_model_path(dirpath, "anewmodel") == Path(dir2) / "anewmodel"
