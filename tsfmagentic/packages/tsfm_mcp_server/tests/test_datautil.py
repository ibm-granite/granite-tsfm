from pydantic import ValidationError


def test_load_timeseries_from_data_uri():
    import pandas as pd
    from tsfm_mcp_server.datautil import load_timeseries
    from tsfm_mcp_server.payloads import DataInput

    # Patch to_pandas
    called = {}

    def fake_to_pandas(uri, timestamp_column=None):
        called["uri"] = uri
        called["timestamp_column"] = timestamp_column
        return pd.DataFrame({"ts": ["2021-01-01"], "value": [123]})

    import tsfm_mcp_server.datautil as datautil_mod

    orig = datautil_mod.to_pandas
    datautil_mod.to_pandas = fake_to_pandas
    try:
        input_obj = DataInput(data_uri="file:///tmp/fake.csv", timestamp_column="ts", target_columns=["value"])
        df = load_timeseries(input_obj)
        assert called["uri"] == "file:///tmp/fake.csv"
        assert called["timestamp_column"] == "ts"
        assert list(df.columns) == ["ts", "value"]
    finally:
        datautil_mod.to_pandas = orig


def test_bad_payloads_throw_validation_error():
    import pytest
    from tsfm_mcp_server.payloads import DataInput

    with pytest.raises(ValidationError):
        _ = DataInput(data=None, data_uri=None, timestamp_column="time")
    with pytest.raises(ValidationError):
        _ = DataInput(data="file://./foo.csv")
