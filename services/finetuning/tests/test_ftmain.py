import pathlib
import subprocess
import sys
import tempfile
from subprocess import CompletedProcess


def test_ftmain_good_input():
    with tempfile.TemporaryDirectory() as target_dir:
        model_name = "afinetuned_model"
        args = [
            "python",
            "tsfmfinetuning/ftmain.py",
            "--payload",
            "./data/ftpayload.json",
            "--target_dir",
            target_dir,
            "--model_name",
            model_name,
            "--config_file",
            "tsfmfinetuning/default_config.yml",
            "--model_arch",
            "ttm",
            "--task_type",
            "forecasting",
        ]

        other_popen_kwargs = {}
        cp: CompletedProcess = subprocess.run(
            args,
            stdin=None,
            input=None,
            capture_output=True,
            text=True,
            shell=False,
            cwd=None,
            timeout=None,
            check=False,
            encoding=None,
            errors=None,
            env=None,
            universal_newlines=None,
            **other_popen_kwargs,
        )

        print(cp.stdout)
        print(cp.stderr)
        sys.stdout.flush()
        assert cp.returncode == 0
        tdir = pathlib.Path(target_dir) / model_name
        assert (tdir / "config.json").exists() and (tdir / "config.json").stat().st_size > 0

        # make sure our tracking log exists
        logfile = pathlib.Path(target_dir) / "output" / "training_logs.jsonl"
        assert logfile.exists() and logfile.stat().st_size > 0
