def test_load_timeseries_from_inline_data():
    import pandas as pd
    from tsfm_mcp_server.datautil import load_timeseries
    from tsfm_mcp_server.payloads import DataInput

    # Patch to_pandas
    called = {}

    def fake_to_pandas(uri, timestamp_column=None):
        called["uri"] = uri
        called["timestamp_column"] = timestamp_column
        return pd.DataFrame({"timestamp": ["2021-01-01", "2021-01-02"], "value": [100, 200]})

    import tsfm_mcp_server.datautil as datautil_mod

    orig = datautil_mod.to_pandas
    datautil_mod.to_pandas = fake_to_pandas
    try:
        fake_csv = b"timestamp,value\n2021-01-01,100\n2021-01-02,200\n"
        input_obj = DataInput(data=fake_csv, data_uri=None, timestamp_column="timestamp")
        df = load_timeseries(input_obj)
        assert called["timestamp_column"] == "timestamp"
        assert called["uri"].startswith("file://")
        assert list(df.columns) == ["timestamp", "value"]
        assert df.shape == (2, 2)
    finally:
        datautil_mod.to_pandas = orig


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
        input_obj = DataInput(data=None, data_uri="file:///tmp/fake.csv", timestamp_column="ts")
        df = load_timeseries(input_obj)
        assert called["uri"] == "file:///tmp/fake.csv"
        assert called["timestamp_column"] == "ts"
        assert list(df.columns) == ["ts", "value"]
    finally:
        datautil_mod.to_pandas = orig


def test_load_timeseries_no_data():
    import pytest
    from tsfm_mcp_server.datautil import load_timeseries
    from tsfm_mcp_server.payloads import DataInput

    input_obj = DataInput(data=None, data_uri=None, timestamp_column="time")
    with pytest.raises(ValueError, match="No data provided"):
        load_timeseries(input_obj)
