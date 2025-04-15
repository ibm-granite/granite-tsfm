# Standard

import json
import os

import numpy as np
import pandas as pd

# Third Party
from locust import FastHttpUser, task
from tsfminference.inference_payloads import (
    ForecastingInferenceInput,
    ForecastingMetadataInput,
    ForecastingParameters,
)


def ts_data_base(series_length: int, num_timeseries: int, num_targets: int) -> pd.DataFrame:
    # Generate a date range
    length = series_length
    date_range = pd.date_range(start="2023-10-01", periods=length, freq="h")

    timeseries = []
    for idx in range(num_timeseries):
        data: dict = {"date": date_range, "ID": str(idx)}
        for idx in range(num_targets):
            data[f"VAL{idx}"] = np.random.rand(series_length)
        timeseries.append(pd.DataFrame(data))

    answer = pd.concat(timeseries, ignore_index=True)
    answer["date"] = answer["date"].astype(int)
    return answer


def forecasting_input_base(model_id: str, series_length: int, num_timeseries: int, num_targets: int) -> dict:
    df: pd.DataFrame = ts_data_base(series_length, num_timeseries, num_targets)
    schema: ForecastingMetadataInput = ForecastingMetadataInput(
        timestamp_column="date", id_columns=["ID"], target_columns=[f"VAL{idx}" for idx in range(num_targets)]
    )
    parameters: ForecastingParameters = ForecastingParameters()
    input: ForecastingInferenceInput = ForecastingInferenceInput(
        model_id=model_id,
        schema=schema,
        parameters=parameters,
        data=df.to_dict(orient="list"),  # this should get replaced in each test case anyway,
    )
    return input.model_dump()


class MyUser(FastHttpUser):
    max_retries = 3  # custom retry count

    @task
    def forecast_synchronous(self):
        forecasting_url = self.host + "/inference/forecasting"
        for attempt in range(self.max_retries):
            try:
                if attempt > 0:
                    print(f"retrying on attempt {attempt+1}")
                response = self.client.post(forecasting_url, json=self.payload, timeout=1200)
                if response.status_code == 200:
                    break
                else:
                    print(f"Attempt {attempt+1} failed with status: {response.status_code}")
            except Exception as e:
                print(f"Attempt {attempt+1} raised exception: {e}")

    def on_start(self):
        model_param_map = {
            "ttm-r1": {"context_length": 512, "prediction_length": 96},
            "ttm-1024-96-r1": {"context_length": 1024, "prediction_length": 96},
            "ttm-r2": {"context_length": 512, "prediction_length": 96},
            "ttm-r2-etth-finetuned": {"context_length": 512, "prediction_length": 96},
            "ttm-r2-etth-finetuned-control": {"context_length": 512, "prediction_length": 96},
            "ttm-1024-96-r2": {"context_length": 1024, "prediction_length": 96},
            "ttm-1536-96-r2": {"context_length": 1536, "prediction_length": 96},
            "ibm/test-patchtst": {"context_length": 512, "prediction_length": 96},
            "ibm/test-patchtsmixer": {"context_length": 512, "prediction_length": 96},
        }

        model_id = os.environ.get("MODEL_ID", "ttm-r1")
        self.payload = forecasting_input_base(
            model_id=model_id,
            num_timeseries=int(os.environ.get("NUM_TIMESERIES", "10")),
            series_length=model_param_map[model_id]["context_length"],
            num_targets=int(os.environ.get("NUM_TARGETS", "1")),
        )

    def on_stop(self):
        metrics_url = self.host.replace("/v1", "") + "/metrics"
        print(self.client.post(metrics_url, json=self.payload, timeout=None, retries=10).text)
        print(f"payload length was {len(json.dumps(self.payload))/1E6}MB")
