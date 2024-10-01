import base64
import datetime
import io

import numpy as np
import pandas as pd
import pytest
from tsfmfinetuning.ioutils import to_pandas


def test_to_pandas_with_feather_content_from_filesystem():
    df = pd.read_feather("tests/data/ETTh2.feather")
    df2 = to_pandas(uri="file://./tests/data/ETTh2.feather")
    assert isinstance(df2.dtypes["date"], np.dtypes.DateTime64DType)
    assert df.equals(df2)


def test_to_pandas_with_unsupported_uri():
    with pytest.raises(NotImplementedError) as _:
        to_pandas("https://foobaz/fizbot.csv")
    with pytest.raises(NotImplementedError) as _:
        to_pandas("s3a://tsfm-services/ETTh2.feather")
    with pytest.raises(NotImplementedError) as _:
        to_pandas("http://tsfm-services/ETTh2.feather")


def test_to_pandas_with_bad_binary_content():
    with pytest.raises(Exception):
        to_pandas(uri="makesnosense")


def test_to_pandas_with_base64_feathered_content():
    df = pd.DataFrame(columns=["date", "value"], data=[[datetime.datetime.now(), 1.0]])
    buf = io.BytesIO()
    df.to_feather(buf)
    buf.seek(0)
    output = base64.b64encode(buf.getbuffer())
    df2 = to_pandas(uri=output.decode("utf-8"), timestamp_column="date")
    assert df.equals(df2)


def test_to_pandas_with_compressed_csv_from_filesystem():
    df2 = to_pandas(uri="file://./tests/data/ETTh2.csv.gz", timestamp_column="date")
    assert isinstance(df2.dtypes["date"], np.dtypes.DateTime64DType)


def test_to_pandas_with_base64_csv_content():
    df = pd.DataFrame(columns=["date", "value"], data=[[datetime.datetime.now(), 1.0]])
    buf = io.BytesIO()
    df.to_csv(buf, header=True, index=False)
    buf.seek(0)
    output = base64.b64encode(buf.getbuffer())
    df2 = to_pandas(uri=output.decode("utf-8"), timestamp_column="date")
    assert df.equals(df2)


def test_to_pandas_with_unsupported_binary_content():
    df = pd.DataFrame(columns=["date", "value"], data=[[datetime.datetime.now(), 1.0]])
    buf = io.BytesIO()
    df.to_pickle(buf)
    buf.seek(0)
    output = base64.b64encode(buf.getbuffer())

    with pytest.raises(Exception):
        to_pandas(uri=base64.b64encode(output.decode("utf-8")))
