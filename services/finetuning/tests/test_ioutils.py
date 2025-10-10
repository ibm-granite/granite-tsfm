import base64
import datetime
import io
import tempfile

import numpy as np
import pandas as pd
import pytest
from tsfmfinetuning.ioutils import _split_text_file, to_pandas


def test_to_pandas_with_feather_content_from_filesystem():
    df = pd.read_feather("data/ETTh1.feather")
    df2 = to_pandas(uri="file://./data/ETTh1.feather")
    assert isinstance(df2.dtypes["date"], np.dtypes.DateTime64DType)
    assert df.equals(df2)


def test_to_pandas_with_csv_content_from_filesystem():
    df = pd.read_csv(filepath_or_buffer="data/ETTh1.csv", parse_dates=["date"])
    df2 = to_pandas(uri="file://./data/ETTh1.csv", timestamp_column="date")
    assert isinstance(df2.dtypes["date"], np.dtypes.DateTime64DType)
    assert df.equals(df2)


def test_to_pandas_with_multipart_csv_content_from_filesystem():
    df = pd.read_csv(filepath_or_buffer="data/ETTh1.csv", parse_dates=["date"])
    df2 = to_pandas(uri="file://./data/multipart")
    assert isinstance(df2.dtypes["date"], np.dtypes.DateTime64DType)
    # we can't check equality as pyarrow is parsing dates like '2016-07-01 00:00:00' to second
    # resolution whereas pd.read_csv resolves it to ns resolution. I think pyparrow is "more" correct.
    assert len(df) == len(df2)
    assert list(df.columns) == list(df2.columns)
    np.array_equal(df.to_numpy(), df2.to_numpy())


def test___split_text_file():
    original = open("./data/ETTh1.csv").readlines()
    with tempfile.TemporaryDirectory() as td:
        parts = 10
        filenames = _split_text_file(
            source="./data/ETTh1.csv", target_dir=td, has_header=True, shuffle=False, parts=parts
        )
        sum = 0
        for fn in filenames:
            sum += len(open(fn).readlines())
        sum -= parts - 1  # account for added header in each part
        assert len(original) == sum, "Line lengths do not match"
        assert original[-1] == open(filenames[-1]).readlines()[-1]
        assert original[1] == open(filenames[0]).readlines()[1]


def test_to_pandas_with_unsupported_uri():
    with pytest.raises(NotImplementedError) as _:
        to_pandas("https://foobaz/fizbot.csv")
    with pytest.raises(NotImplementedError) as _:
        to_pandas("s3a://tsfm-services/ETTh1.feather")
    with pytest.raises(NotImplementedError) as _:
        to_pandas("http://tsfm-services/ETTh1.feather")
    with pytest.raises(NotImplementedError) as _:
        to_pandas("./data/ETTh1.csv")  # we require a proper uri
    with pytest.raises(ValueError) as _:
        to_pandas("./data/foo.csv")
    with pytest.raises(ValueError) as _:
        to_pandas("./data/foo")  # passing multipart that's not there


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
    df2 = to_pandas(uri="file://./data/ETTh1.csv.gz", timestamp_column="date")
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
