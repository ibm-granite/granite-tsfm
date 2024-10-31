# Copyright contributors to the TSFM project
#

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
import yaml
from tsfminference import TSFM_CONFIG_FILE
from tsfminference.inference import InferenceRuntime
from tsfminference.inference_payloads import (
    ForecastingInferenceInput,
    ForecastingMetadataInput,
    ForecastingParameters,
    PredictOutput,
)


@pytest.fixture(scope="module")
def ts_data_base() -> pd.DataFrame:
    # Generate a date range
    length = 512
    date_range = pd.date_range(start="2023-10-01", periods=length, freq="H")

    # Create a DataFrame
    df = pd.DataFrame(
        {
            "date": date_range,
            "ID": "1",
            "VAL": np.random.rand(length),
        }
    )

    return df


if TSFM_CONFIG_FILE:
    with open(TSFM_CONFIG_FILE, "r") as file:
        config = yaml.safe_load(file)
else:
    config = {}

MODEL_ID = "mytest-tsfm/ttm-r1"


def test_forecast_with_good_data(ts_data_base: pd.DataFrame):
    df: pd.DataFrame = ts_data_base
    runtime: InferenceRuntime = InferenceRuntime(config=config)
    schema: ForecastingMetadataInput = ForecastingMetadataInput(
        timestamp_column="date", id_columns=["ID"], target_columns=["VAL"]
    )
    parameters: ForecastingParameters = ForecastingParameters()
    input: ForecastingInferenceInput = ForecastingInferenceInput(
        model_id=MODEL_ID, schema=schema, parameters=parameters, data=df.to_dict(orient="list")
    )
    po: PredictOutput = runtime.forecast(input=input)

    results = pd.DataFrame.from_dict(po.results[0])

    # expected length
    assert len(results) == 96
    # expected start time
    assert results["date"].iloc[0] - df["date"].iloc[-1] == timedelta(hours=1)
    # expected end time
    assert results["date"].iloc[-1] - df["date"].iloc[-1] == timedelta(hours=96)
