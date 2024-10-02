# Standard

import tempfile

import pytest
import yaml
from tsfmfinetuning import TSFM_CONFIG_FILE
from tsfmfinetuning.finetuning import FinetuningRuntime
from tsfmfinetuning.ftpayloads import TinyTimeMixerForecastingTuneInput


PAYLOADS = [
    {
        "data": "file://./path/to/pandas.csv/or/pandas.feather/or/pandas.csv.gz",
        "model_id": "ibm/test-ttm-v1",
        "schema": {
            "timestamp_column": "date",
            "target_columns": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"],
        },
        "parameters": {
            "tune_prefix": "fine_tuned/",
            "trainer_args": {"num_train_epochs": 1, "per_device_train_batch_size": 256},
            "fewshot_fraction": 0.05,
        },
    }
]

file_data_uris = [
    "file://./tests/data/ETTh2.feather",
]


@pytest.mark.parametrize("uri", file_data_uris)
@pytest.mark.parametrize("payload", PAYLOADS)
def test_fine_tune_forecasting_with_local_io(uri, payload):
    payload["data"] = uri
    input: TinyTimeMixerForecastingTuneInput = TinyTimeMixerForecastingTuneInput(**payload)
    if TSFM_CONFIG_FILE:
        with open(TSFM_CONFIG_FILE, "r") as file:
            config = yaml.safe_load(file)
    else:
        config = {}

    ftr: FinetuningRuntime = FinetuningRuntime(config=config)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tuned_model_name = "pytest_tuned_model"
        response = ftr.finetuning(input=input, tuned_model_name=tuned_model_name, output_dir=tmp_dir)
        assert response.exists()
        assert (response / "config.json").exists() and (response / "config.json").stat().st_size > 0
