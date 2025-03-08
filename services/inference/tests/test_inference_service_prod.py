# Copyright contributors to the TSFM project
#

import json
import os
from typing import Any, Dict

import pandas as pd
import requests


def get_tuple(resp):
    resp = resp.json()
    df = [pd.DataFrame.from_dict(r) for r in resp["results"]]
    return df, {k: v for k, v in resp.items() if "data_point" in k}


def get_inference_response(
    msg: Dict[str, Any],
    url: str = "http://127.0.0.1:8000/v1/inference/forecasting",
) -> requests.Response:
    headers = {}
    return requests.post(url, json=msg, headers=headers)


def numts(msg):
    return 1 if "id_cols" not in msg["parameters"] else len(set(msg["parameters"]["id_columns"]))


"""
Scenario Outline: should get a 200 response code and 10 rows from request for model "granite-ttm-512-96-r2" with 512 time points input # features/time_series_forecast.feature:123
    And the response result length should be 10 # features/time_series_forecast.feature:137
      Error: Actual HUFL length is not equal to 10.
"""


def test_scenario_01():
    """Comments:
    The payload as given to me from product asks for only two predictions,
    not 10, as shown in the
    """
    msg = json.load(open(os.path.join(os.path.dirname(__file__), "prodpayload.json")))
    # change prediction length
    msg["parameters"]["prediction_length"] = 10
    resp = get_inference_response(msg)

    assert resp.ok
    df, _ = get_tuple(resp)
    numtseries = numts(msg)
    for idx in range(numtseries):
        assert df[idx].shape[0] == msg["parameters"]["prediction_length"]
        assert df[idx]["HUFL"].shape[0] == msg["parameters"]["prediction_length"]


"""
  Scenario Outline: should get a 400 response code from request for model "granite-ttm-512-96-r2" with 512 time points input and 1024 prediction_length # features/time_series_forecast.feature:145
    Then the response code should be 400 # features/time_series_forecast.feature:158
      Error: expected response code to be: 400, but actual is: 200.
"""


def test_scenario_02():
    msg = json.load(open(os.path.join(os.path.dirname(__file__), "prodpayload.json")))
    # change prediction length
    msg["parameters"]["prediction_length"] = 1024
    resp: requests.Response = get_inference_response(msg)
    assert resp.status_code == 400
