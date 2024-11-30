# Copyright contributors to the TSFM project
#

import copy
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
import yaml
from fastapi import HTTPException
from tsfminference import TSFM_CONFIG_FILE
from tsfminference.inference import InferenceRuntime
from tsfminference.inference_payloads import (
    ForecastingInferenceInput,
    ForecastingMetadataInput,
    ForecastingParameters,
    PredictOutput,
)


SERIES_LENGTH = 512
FORECAST_LENGTH = 96

MODEL_IDS = ["mytest-tsfm/ttm-r1", "mytest-tsfm/chronos-t5-tiny"]


@pytest.fixture(scope="module")
def ts_data_base() -> pd.DataFrame:
    # Generate a date range
    length = SERIES_LENGTH
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


@pytest.fixture(scope="module")
def forecasting_input_base() -> ForecastingInferenceInput:
    # df: pd.DataFrame = ts_data_base
    schema: ForecastingMetadataInput = ForecastingMetadataInput(
        timestamp_column="date", id_columns=["ID"], target_columns=["VAL"]
    )
    parameters: ForecastingParameters = ForecastingParameters()
    input: ForecastingInferenceInput = ForecastingInferenceInput(
        model_id=MODEL_ID,
        schema=schema,
        parameters=parameters,
        data={
            "date": [
                "2024-10-18T01:00:21+00:00",
            ],
            "ID1": [
                "I1",
            ],
            "VAL": [
                10.0,
            ],
        },  # this should get replaced in each test case anyway,
    )
    return input


def _basic_result_checks(results: PredictOutput, df: pd.DataFrame):
    # expected length
    assert len(results) == FORECAST_LENGTH
    # expected start time
    assert results["date"].iloc[0] - df["date"].iloc[-1] == timedelta(hours=1)
    # expected end time
    assert results["date"].iloc[-1] - df["date"].iloc[-1] == timedelta(hours=FORECAST_LENGTH)


@pytest.mark.parametrize("forecasting_input_base", MODEL_IDS, indirect=True)
def test_forecast_with_good_data(ts_data_base: pd.DataFrame, forecasting_input_base: ForecastingInferenceInput):
    input = forecasting_input_base
    df = copy.deepcopy(ts_data_base)
    input.data = df.to_dict(orient="list")
    runtime: InferenceRuntime = InferenceRuntime(config=config)
    po: PredictOutput = runtime.forecast(input=input)
    results = pd.DataFrame.from_dict(po.results[0])
    _basic_result_checks(results, df)


@pytest.mark.parametrize("forecasting_input_base", MODEL_IDS, indirect=True)
def test_forecast_with_schema_missing_target_columns(
    ts_data_base: pd.DataFrame, forecasting_input_base: ForecastingInferenceInput
):
    input = forecasting_input_base
    input.schema.target_columns = []
    df = copy.deepcopy(ts_data_base)
    input.data = df.to_dict(orient="list")
    runtime: InferenceRuntime = InferenceRuntime(config=config)
    po: PredictOutput = runtime.forecast(input=input)
    results = pd.DataFrame.from_dict(po.results[0])
    _basic_result_checks(results, df)


@pytest.mark.parametrize("forecasting_input_base", MODEL_IDS, indirect=True)
def test_forecast_with_integer_timestamps(
    ts_data_base: pd.DataFrame, forecasting_input_base: ForecastingInferenceInput
):
    input: ForecastingInferenceInput = copy.deepcopy(forecasting_input_base)
    df = copy.deepcopy(ts_data_base)

    timestamp_column = input.schema.timestamp_column
    df[timestamp_column] = df[timestamp_column].astype(int)
    df[timestamp_column] = range(1, SERIES_LENGTH + 1)
    input.data = df.to_dict(orient="list")
    runtime: InferenceRuntime = InferenceRuntime(config=config)
    po: PredictOutput = runtime.forecast(input=input)
    results = pd.DataFrame.from_dict(po.results[0])
    assert results[timestamp_column].iloc[0] == SERIES_LENGTH + 1
    assert results[timestamp_column].iloc[-1] - df[timestamp_column].iloc[-1] == FORECAST_LENGTH
    assert results.dtypes[timestamp_column] == df.dtypes[timestamp_column]


@pytest.mark.parametrize("forecasting_input_base", MODEL_IDS, indirect=True)
def test_forecast_with_bogus_timestamps(ts_data_base: pd.DataFrame, forecasting_input_base: ForecastingInferenceInput):
    input: ForecastingInferenceInput = copy.deepcopy(forecasting_input_base)
    df = copy.deepcopy(ts_data_base)

    timestamp_column = input.schema.timestamp_column
    df[timestamp_column] = df[timestamp_column].astype(str)
    df[timestamp_column] = [str(x) for x in range(1, SERIES_LENGTH + 1)]
    input.data = df.to_dict(orient="list")
    runtime: InferenceRuntime = InferenceRuntime(config=config)
    with pytest.raises(ValueError) as _:
        runtime.forecast(input=input)


@pytest.mark.parametrize("forecasting_input_base", MODEL_IDS, indirect=True)
def test_forecast_with_bogus_values(ts_data_base: pd.DataFrame, forecasting_input_base: ForecastingInferenceInput):
    input: ForecastingInferenceInput = copy.deepcopy(forecasting_input_base)
    df = copy.deepcopy(ts_data_base)
    df["VAL"] = df["VAL"].astype(str)
    df["VAL"] = [str(x) for x in range(1, SERIES_LENGTH + 1)]
    input.data = df.to_dict(orient="list")
    runtime: InferenceRuntime = InferenceRuntime(config=config)
    with pytest.raises(HTTPException) as _:
        runtime.forecast(input=input)


@pytest.mark.parametrize("forecasting_input_base", MODEL_IDS, indirect=True)
def test_forecast_with_bogus_model_id(ts_data_base: pd.DataFrame, forecasting_input_base: ForecastingInferenceInput):
    input: ForecastingInferenceInput = copy.deepcopy(forecasting_input_base)
    df = copy.deepcopy(ts_data_base)
    input.data = df.to_dict(orient="list")
    input.model_id = "hoo-hah"

    runtime: InferenceRuntime = InferenceRuntime(config=config)
    with pytest.raises(HTTPException) as _:
        runtime.forecast(input=input)


@pytest.mark.parametrize("forecasting_input_base", MODEL_IDS, indirect=True)
def test_forecast_with_insufficient_context_length(
    ts_data_base: pd.DataFrame, forecasting_input_base: ForecastingInferenceInput
):
    input: ForecastingInferenceInput = copy.deepcopy(forecasting_input_base)
    df = copy.deepcopy(ts_data_base)
    df = df.iloc[0:-100]

    input.data = df.to_dict(orient="list")

    runtime: InferenceRuntime = InferenceRuntime(config=config)
    with pytest.raises(HTTPException) as _:
        runtime.forecast(input=input)


@pytest.mark.skip
@pytest.mark.parametrize("forecasting_input_base", MODEL_IDS, indirect=True)
def test_forecast_with_nan_data(ts_data_base: pd.DataFrame, forecasting_input_base: ForecastingInferenceInput):
    input: ForecastingInferenceInput = copy.deepcopy(forecasting_input_base)
    df = copy.deepcopy(ts_data_base)
    df.iloc[0, df.columns.get_loc("VAL")] = np.nan

    input.data = df.to_dict(orient="list")

    runtime: InferenceRuntime = InferenceRuntime(config=config)
    # with pytest.raises(HTTPException) as _:
    runtime.forecast(input=input)


# @pytest.mark.skip
@pytest.mark.parametrize("forecasting_input_base", MODEL_IDS, indirect=True)
def test_forecast_with_missing_row(ts_data_base: pd.DataFrame, forecasting_input_base: ForecastingInferenceInput):
    input: ForecastingInferenceInput = copy.deepcopy(forecasting_input_base)
    df = copy.deepcopy(ts_data_base)
    df = df.drop(index=10)

    # append a row to give it the correct length
    # don't forget to update the timestamp accordingly in the
    # appended row

    input.data = df.to_dict(orient="list")

    runtime: InferenceRuntime = InferenceRuntime(config=config)
    with pytest.raises(HTTPException) as _:
        runtime.forecast(input=input)
