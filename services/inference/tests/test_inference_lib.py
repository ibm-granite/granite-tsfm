# Copyright contributors to the TSFM project
#

import copy
import json
import os
import tempfile
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from fastapi import HTTPException
from pytest import FixtureRequest
from tsfminference import TSFM_CONFIG_FILE, TSFM_MODEL_DIR
from tsfminference.inference import InferenceRuntime
from tsfminference.inference_payloads import (
    ForecastingInferenceInput,
    ForecastingMetadataInput,
    ForecastingParameters,
    PredictOutput,
)
from tsfminference.service_handler import ForecastingServiceHandler


# SERIES_LENGTH = 512
FORECAST_LENGTH = 96


MODEL_IDS = [
    os.path.basename(dirpath)
    for dirpath, _, _ in os.walk("./mytest-tsfm")
    if ".git" not in dirpath and "./mytest-tsfm" != dirpath and "finetuned" not in dirpath and "figures" not in dirpath
]


def min_context_length(model_id):
    model_path: Path = TSFM_MODEL_DIR / model_id
    assert model_path.exists(), f"{model_path} does not exist!"
    handler, e = ForecastingServiceHandler.load(model_id=model_id, model_path=model_path)
    return handler.handler_config.minimum_context_length


@pytest.fixture(scope="module")
def ts_data_base(request: type[FixtureRequest]) -> pd.DataFrame:
    # Generate a date range
    length = min_context_length(request.param)
    date_range = pd.date_range(start="2023-10-01", periods=length, freq="H")

    # Create a DataFrame
    df = pd.DataFrame(
        {
            "date": date_range,
            "ID": "1",
            "VAL": np.random.rand(length),
        }
    )

    return df, request.param


if TSFM_CONFIG_FILE:
    with open(TSFM_CONFIG_FILE, "r") as file:
        config = yaml.safe_load(file)
else:
    config = {}


@pytest.fixture(scope="module")
def forecasting_input_base(request: type[FixtureRequest]) -> ForecastingInferenceInput:
    # df: pd.DataFrame = ts_data_base
    schema: ForecastingMetadataInput = ForecastingMetadataInput(
        timestamp_column="date", id_columns=["ID"], target_columns=["VAL"]
    )
    parameters: ForecastingParameters = ForecastingParameters(prediction_length=FORECAST_LENGTH)
    input: ForecastingInferenceInput = ForecastingInferenceInput(
        model_id=request.param,
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
@pytest.mark.parametrize("ts_data_base", MODEL_IDS, indirect=True)
def test_forecast_with_good_data(ts_data_base: pd.DataFrame, forecasting_input_base: ForecastingInferenceInput):
    input = forecasting_input_base
    data, model_id = ts_data_base
    # since we're sometimes generating non-sensible combinations
    # skip those
    if input.model_id != model_id:
        return
    df = copy.deepcopy(data)
    input.data = df.to_dict(orient="list")

    # useful for generating sample payload files
    if int(os.environ.get("TSFM_TESTS_DO_VERBOSE_DUMPS", "0")) == 1:
        with open(f"{tempfile.gettempdir()}/{model_id}.payload.json", "w") as out:
            foo = copy.deepcopy(df)
            foo["date"] = foo["date"].apply(lambda x: x.isoformat())
            json.dump(foo.to_dict(orient="list"), out)

    runtime: InferenceRuntime = InferenceRuntime(config=config)
    po: PredictOutput = runtime.forecast(input=input)
    results = pd.DataFrame.from_dict(po.results[0])
    _basic_result_checks(results, df)


@pytest.mark.parametrize("forecasting_input_base", MODEL_IDS, indirect=True)
@pytest.mark.parametrize("ts_data_base", MODEL_IDS, indirect=True)
def test_forecast_with_schema_missing_target_columns(
    ts_data_base: pd.DataFrame, forecasting_input_base: ForecastingInferenceInput
):
    input = forecasting_input_base
    input.schema.target_columns = []
    data, model_id = ts_data_base
    # since we're sometimes generating non-sensible combinations
    # skip those
    if input.model_id != model_id:
        return

    df = copy.deepcopy(data)
    input.data = df.to_dict(orient="list")
    runtime: InferenceRuntime = InferenceRuntime(config=config)
    po: PredictOutput = runtime.forecast(input=input)
    results = pd.DataFrame.from_dict(po.results[0])
    _basic_result_checks(results, df)


@pytest.mark.parametrize("forecasting_input_base", MODEL_IDS, indirect=True)
@pytest.mark.parametrize("ts_data_base", MODEL_IDS, indirect=True)
def test_forecast_with_integer_timestamps(
    ts_data_base: pd.DataFrame, forecasting_input_base: ForecastingInferenceInput
):
    input: ForecastingInferenceInput = copy.deepcopy(forecasting_input_base)
    data, model_id = ts_data_base
    # since we're sometimes generating non-sensible combinations
    # skip those
    if input.model_id != model_id:
        return
    df = copy.deepcopy(data)
    series_length = len(df)

    timestamp_column = input.schema.timestamp_column
    df[timestamp_column] = df[timestamp_column].astype(int)
    df[timestamp_column] = range(1, series_length + 1)
    input.data = df.to_dict(orient="list")
    runtime: InferenceRuntime = InferenceRuntime(config=config)
    po: PredictOutput = runtime.forecast(input=input)
    results = pd.DataFrame.from_dict(po.results[0])
    assert results[timestamp_column].iloc[0] == series_length + 1
    assert results[timestamp_column].iloc[-1] - df[timestamp_column].iloc[-1] == FORECAST_LENGTH
    assert results.dtypes[timestamp_column] == df.dtypes[timestamp_column]


@pytest.mark.parametrize("forecasting_input_base", MODEL_IDS, indirect=True)
@pytest.mark.parametrize("ts_data_base", MODEL_IDS, indirect=True)
def test_forecast_with_bogus_timestamps(ts_data_base: pd.DataFrame, forecasting_input_base: ForecastingInferenceInput):
    input: ForecastingInferenceInput = copy.deepcopy(forecasting_input_base)
    data, model_id = ts_data_base
    # since we're sometimes generating non-sensible combinations
    # skip those
    if input.model_id != model_id:
        return
    df = copy.deepcopy(data)
    series_length = len(df)
    timestamp_column = input.schema.timestamp_column
    df[timestamp_column] = df[timestamp_column].astype(str)
    df[timestamp_column] = [str(x) for x in range(1, series_length + 1)]
    input.data = df.to_dict(orient="list")
    runtime: InferenceRuntime = InferenceRuntime(config=config)
    with pytest.raises(ValueError) as _:
        runtime.forecast(input=input)


@pytest.mark.parametrize("forecasting_input_base", MODEL_IDS, indirect=True)
@pytest.mark.parametrize("ts_data_base", MODEL_IDS, indirect=True)
def test_forecast_with_bogus_values(ts_data_base: pd.DataFrame, forecasting_input_base: ForecastingInferenceInput):
    input: ForecastingInferenceInput = copy.deepcopy(forecasting_input_base)
    data, model_id = ts_data_base
    # since we're sometimes generating non-sensible combinations
    # skip those
    if input.model_id != model_id:
        return
    df = copy.deepcopy(data)
    series_length = len(df)
    df["VAL"] = df["VAL"].astype(str)
    df["VAL"] = [str(x) for x in range(1, series_length + 1)]
    input.data = df.to_dict(orient="list")
    runtime: InferenceRuntime = InferenceRuntime(config=config)
    with pytest.raises(HTTPException) as _:
        runtime.forecast(input=input)


@pytest.mark.parametrize("forecasting_input_base", MODEL_IDS, indirect=True)
@pytest.mark.parametrize("ts_data_base", MODEL_IDS, indirect=True)
def test_forecast_with_bogus_model_id(ts_data_base: pd.DataFrame, forecasting_input_base: ForecastingInferenceInput):
    input: ForecastingInferenceInput = copy.deepcopy(forecasting_input_base)
    data, model_id = ts_data_base
    # since we're sometimes generating non-sensible combinations
    # skip those
    if input.model_id != model_id:
        return
    df = copy.deepcopy(data)
    input.data = df.to_dict(orient="list")
    input.model_id = "hoo-hah"

    runtime: InferenceRuntime = InferenceRuntime(config=config)
    with pytest.raises(HTTPException) as _:
        runtime.forecast(input=input)


@pytest.mark.parametrize("forecasting_input_base", MODEL_IDS, indirect=True)
@pytest.mark.parametrize("ts_data_base", MODEL_IDS, indirect=True)
def test_forecast_with_insufficient_context_length(
    ts_data_base: pd.DataFrame, forecasting_input_base: ForecastingInferenceInput
):
    input: ForecastingInferenceInput = copy.deepcopy(forecasting_input_base)
    data, model_id = ts_data_base
    # since we're sometimes generating non-sensible combinations
    # skip those
    if input.model_id != model_id:
        return
    df = copy.deepcopy(data)
    df = df.iloc[0:10]  # should be well below the min context lengths for our models

    input.data = df.to_dict(orient="list")

    runtime: InferenceRuntime = InferenceRuntime(config=config)
    with pytest.raises(HTTPException) as _:
        runtime.forecast(input=input)


@pytest.mark.skip
@pytest.mark.parametrize("forecasting_input_base", MODEL_IDS, indirect=True)
@pytest.mark.parametrize("ts_data_base", MODEL_IDS, indirect=True)
def test_forecast_with_nan_data(ts_data_base: pd.DataFrame, forecasting_input_base: ForecastingInferenceInput):
    input: ForecastingInferenceInput = copy.deepcopy(forecasting_input_base)
    data, model_id = ts_data_base
    # since we're sometimes generating non-sensible combinations
    # skip those
    if input.model_id != model_id:
        return
    df = copy.deepcopy(data)
    df.iloc[0, df.columns.get_loc("VAL")] = np.nan

    input.data = df.to_dict(orient="list")

    runtime: InferenceRuntime = InferenceRuntime(config=config)
    # with pytest.raises(HTTPException) as _:
    runtime.forecast(input=input)


# @pytest.mark.skip
@pytest.mark.parametrize("forecasting_input_base", MODEL_IDS, indirect=True)
@pytest.mark.parametrize("ts_data_base", MODEL_IDS, indirect=True)
def test_forecast_with_missing_row(ts_data_base: pd.DataFrame, forecasting_input_base: ForecastingInferenceInput):
    input: ForecastingInferenceInput = copy.deepcopy(forecasting_input_base)
    data, model_id = ts_data_base
    # since we're sometimes generating non-sensible combinations
    # skip those
    if input.model_id != model_id:
        return
    df = copy.deepcopy(data)
    df.drop(index=10, inplace=True)
    input.data = df.to_dict(orient="list")

    runtime: InferenceRuntime = InferenceRuntime(config=config)
    with pytest.raises(HTTPException) as _:
        runtime.forecast(input=input)
