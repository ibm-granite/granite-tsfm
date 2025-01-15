import os


# this needs to come before any tsfm imports
os.environ["TSFM_MODEL_DIR"] = "granite-tsfm/services/inference/mytest-tsfm"

import json  # noqa: E402
import shutil  # noqa: E402
import tarfile  # noqa: E402
from datetime import datetime  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from inference import input_fn, model_fn, output_fn, predict_fn  # noqa: E402
from tsfminference.inference_payloads import PredictOutput  # noqa: E402


def payload(json_file_name: str = None):
    if json_file_name:
        return open(json_file_name).read()

    # Generate 10 sequential dates starting from today
    tslength = 1024
    start_date = datetime(2020, 1, 1)
    date_range = [d.isoformat() for d in pd.date_range(start=start_date, periods=tslength, freq="h")]
    ids = ["A" for _ in range(tslength)]
    values = np.random.rand(tslength)

    # Create the DataFrame
    df = pd.DataFrame({"date": date_range, "ID": ids, "value": values})

    return json.dumps(
        {
            "inference_type": "forecasting",  # we currently support only 'forecasting'
            "model_id": "ttm-1024-96-r1",
            "parameters": {},
            "schema": {
                "timestamp_column": "date",
                "id_columns": ["ID"],  # multiple columns are supported
                "target_columns": ["value"],  # what we're generating a forecast for
            },
            "data": df.to_dict(orient="list"),
            "future_data": {},  # used for things like exogenous data
        }
    )


def fetch_model(model_data):
    """Untar the model.tar.gz object either from local file system
    or a S3 location

    Args:
        model_data (str): either a path to local file system starts with
        file:/// that points to the `model.tar.gz` file or an S3 link
        starts with s3:// that points to the `model.tar.gz` file

    Returns:
        model_dir (str): the directory that contains the uncompress model
        checkpoint files
    """

    # no user serviceable parts here!
    # this just emulates what sagemaker would do
    # on the cloud

    model_dir = "/tmp/model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # _check_model(model_data)
    shutil.copy2(
        model_data,
        model_dir,
    )

    # untar the model
    tar = tarfile.open(os.path.join(model_dir, "model.tar.gz"))
    tar.extractall(model_dir)
    tar.close()

    return model_dir


def test(model_data, json_file_name: str = None):
    # decompress the model.tar.gz file
    model_dir = fetch_model(model_data)

    # load the model
    net = model_fn(model_dir)
    jstr = payload(json_file_name=json_file_name)
    # encode numpy array to binary stream
    content_type = "application/json"
    input_object = input_fn(jstr, content_type)
    predictions = predict_fn(input_object, net)

    if not isinstance(predictions, PredictOutput):
        print(f"non-normal return {predictions}")
        sys.exit(1)

    res = output_fn(predictions, content_type)
    res = json.loads(res)
    assert res["output_data_points"] == 96


if __name__ == "__main__":
    model_data = "../model.tar.gz"

    import sys

    json_file_name = None
    if len(sys.argv) > 1:
        json_file_name = sys.argv[1]

    test(model_data, json_file_name)
