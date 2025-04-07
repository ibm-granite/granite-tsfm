# Standard

import json

import numpy as np
import pandas as pd

# Third Party
from locust import FastHttpUser, task
from tsfminference.inference_payloads import (
    ForecastingInferenceInput,
    ForecastingMetadataInput,
    ForecastingParameters,
)


# SERIES_LENGTH = int(os.getenv("TSFM_PROFILE_SERIES_LENGTH", 512))
# FORECAST_LENGTH = 96
# MODEL_ID = "ttm-r1"
# NUM_TIMESERIES = int(os.getenv("TSFM_PROFILE_NUM_TIMESERIES", 2))


def ts_data_base(series_length: int, num_timeseries: int) -> pd.DataFrame:
    # Generate a date range
    length = series_length
    date_range = pd.date_range(start="2023-10-01", periods=length, freq="h")

    timeseries = []
    for idx in range(num_timeseries):
        timeseries.append(
            pd.DataFrame(
                {
                    "date": date_range,
                    "ID": str(idx),
                    "VAL": np.random.rand(length),
                }
            )
        )

    answer = pd.concat(timeseries, ignore_index=True)
    answer["date"] = answer["date"].astype(int)
    return answer


def forecasting_input_base(model_id: str, series_length: int, num_timeseries: int) -> dict:
    df: pd.DataFrame = ts_data_base(series_length, num_timeseries)
    schema: ForecastingMetadataInput = ForecastingMetadataInput(
        timestamp_column="date", id_columns=["ID"], target_columns=["VAL"]
    )
    parameters: ForecastingParameters = ForecastingParameters()
    input: ForecastingInferenceInput = ForecastingInferenceInput(
        model_id=model_id,
        schema=schema,
        parameters=parameters,
        data=df.to_dict(orient="list"),  # this should get replaced in each test case anyway,
    )
    return input.model_dump()


class QuickstartUser(FastHttpUser):
    _printed_error = {}

    @task
    def forecast_synchronous(self):
        forecasting_url = self.host + "/inference/forecasting"

        response = self.client.post(forecasting_url, json=self.payload, timeout=None, retries=10)
        if not response.status_code == 200 and response.text not in QuickstartUser._printed_error:
            print("#" * 25)
            print(response.text)
            print("#" * 25)
            QuickstartUser._printed_error[response.text] = True

    def on_start(self):
        self.payload = forecasting_input_base(model_id="ttm-r1", num_timeseries=500, series_length=512)

    def on_stop(self):
        metrics_url = self.host.replace("/v1", "") + "/metrics"
        print(self.client.post(metrics_url, json=self.payload, timeout=None, retries=10).text)
        print(f"payload length was {len(json.dumps(self.payload))/1E6}MB")
