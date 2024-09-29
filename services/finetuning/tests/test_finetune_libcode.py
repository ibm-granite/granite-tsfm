# Standard
import tempfile

import pytest
from tsfmfinetuning.ftcommon import forecasting_tuning_to_local
from tsfmfinetuning.ftpayloads import TinyTimeMixerForecastingTuneInput, TuneOutput


PAYLOADS = [
    {
        "model_id": "ibm/tinytimemixer-monash-fl_96",
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
    "file://./tests/data/ETTh2.csv",
    "file://./tests/data/ETTh2.csv.gz",
]


@pytest.mark.parametrize("uri", file_data_uris)
@pytest.mark.parametrize("payload", PAYLOADS)
def test_fine_tune_forecasting_with_local_io(uri, payload):
    input = TinyTimeMixerForecastingTuneInput(**payload)
    response = forecasting_tuning_to_local(input=input, target_dir=tempfile.gettempdir(), model_name="finetuned")
    assert isinstance(response, TuneOutput)
