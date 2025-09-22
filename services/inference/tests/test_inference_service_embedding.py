# Copyright contributors to the TSFM project
#
import json
import os
from typing import Any, Dict, Optional, Union

import pandas as pd
import pytest
import requests

from tsfm_public.toolkit.util import convert_tsfile_to_dataframe, encode_data, select_by_index


DUMPPAYLOADS = int(os.getenv("TSFM_TESTS_DUMP_PAYLOADS", "0")) == 1

model_param_map = {
    "tspulse-r1": {"context_length": 512},
}


@pytest.fixture(scope="module")
def ts_data_base():
    dataset_path = "/Users/shoffman/Projects/External/IBM/granite-tsfm/notebooks/hfdemo/Univariate_ts/ShapesAll/ShapesAll_train.ts"

    # forecast_length = 96
    # context_length = 512
    # timestamp_column = "date"

    data = convert_tsfile_to_dataframe(
        dataset_path,
        return_separate_X_and_y=False,
    )

    return data


@pytest.fixture(scope="module")
def ts_data(ts_data_base, request):
    # forecast_length = 96
    # context_length = 512
    model_id = request.param
    # prediction_length = model_param_map[model_id]["prediction_length"]
    context_length = model_param_map[model_id]["context_length"]
    # timestamp_column = "date"

    test_data = select_by_index(
        ts_data_base,
        start_index=0,
        end_index=2,
    ).reset_index(drop=True)
    test_data = pd.concat(
        [pd.DataFrame(test_data["dim_0"].tolist()), test_data["class_vals"]],
        axis=1
    ).reset_index(drop=False).melt(id_vars=["index", "class_vals"])
    test_data["class_vals"] = test_data["class_vals"].astype(int)
    test_data["variable"] = test_data["variable"].astype(int)

    # test_data["id"] = np.array(["a", "b", "c", "d", "e"]).repeat(context_length)

    return test_data, {
        "timestamp_column": "variable",
        "id_columns": ["index"],
        "label_column": "class_vals",
        "input_columns": ["value"],
        "context_length": context_length,
        "model_id": model_id,
    }


def get_inference_response(
    msg: Dict[str, Any], dumpfile: Optional[Union[str, None]] = None
) -> pd.DataFrame:
    URL = (
        "http://127.0.0.1:8000/v1/inference/embeddings"
        if os.environ.get("TSFM_EMBEDDING_ENDPOINT", None) is None
        else os.environ.get("TSFM_EMBEDDING_ENDPOINT")
    )
    headers = {}
    if dumpfile:
        json.dump(msg, fp=open(dumpfile, "w"), indent=4)
    req = requests.post(URL, json=msg, headers=headers)

    #
    if req.ok:
        resp = req.json()

        df = [pd.DataFrame.from_dict(r) for r in resp["results"]]
        return df, {k: v for k, v in resp.items() if "data_point" in k}
    else:
        print(req.text)
        return req


@pytest.mark.parametrize(
    "ts_data",
    ["tspulse-r1"],
    indirect=True,
)
def test_zero_shot_embedding_inference(ts_data):
    test_data, params = ts_data

    # prediction_length = params["prediction_length"]
    context_length = params["context_length"]
    model_id = params["model_id"]
    model_id_path: str = model_id

    id_columns = params["id_columns"]

    num_ids = test_data[id_columns[0]].nunique()

    # test single
    test_data_ = test_data.copy()

    msg = {
        "model_id": model_id_path,
        "parameters": {
            # "prediction_length": params["prediction_length"],
        },
        "schema": {
            "timestamp_column": params["timestamp_column"],
            "id_columns": params["id_columns"],
            "label_column": params["label_column"],
            "input_columns": params["input_columns"],
        },
        "data": encode_data(test_data_, params["timestamp_column"]),
    }

    df_out, counts = get_inference_response(
        msg, dumpfile="/tmp/test_zero_shot_embedding_inference.json"
    )
    assert len(df_out) == 1
    assert df_out[0].shape[0] == num_ids
    assert counts["input_data_points"] == context_length * len(params["input_columns"]) * num_ids
    # assert counts["output_data_points"] == prediction_length * len(params["input_columns"]) * num_ids
