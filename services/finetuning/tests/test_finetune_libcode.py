# Standard

import tempfile

import pytest
from tsfmfinetuning.finetuning import FinetuningRuntime
from tsfmfinetuning.ftpayloads import TinyTimeMixerForecastingTuneInput


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
    payload["data"] = uri
    input: TinyTimeMixerForecastingTuneInput = TinyTimeMixerForecastingTuneInput(**payload)
    ftr: FinetuningRuntime = FinetuningRuntime()
    response = ftr.finetuning(input=input, tuned_model_name="my_tuned_ttm_model", output_dir=tempfile.gettempdir())
    print(response)
