import base64
from unittest.mock import patch

import pandas as pd



@pytest.mark.asyncio
async def test_load_timeseries_from_inline_data():
    """It should load from inline binary data and call to_pandas with a base64 string."""
    fake_csv = b"timestamp,value\n2021-01-01,100\n2021-01-02,200\n"

    input_obj = DataInput(data=fake_csv, data_uri=None, timestamp_column="timestamp")

    mock_df = pd.DataFrame({"timestamp": ["2021-01-01", "2021-01-02"], "value": [100, 200]})

    with patch("your_module.to_pandas", return_value=mock_df) as mock_to_pandas:
        result = await load_timeseries(input_obj)

    # Validate: the mock was called correctly
    args, kwargs = mock_to_pandas.call_args
    uri_arg = kwargs["uri"]

    # Check that uri_arg is valid base64 of the CSV
    decoded = base64.b64decode(uri_arg.encode("utf-8"))
    assert decoded == fake_csv
    assert result.equals(mock_df)
    mock_to_pandas.assert_called_once_with(uri=uri_arg, timestamp_column="timestamp")


@pytest.mark.asyncio
async def test_load_timeseries_from_data_uri():
    """It should call to_pandas with the given data_uri directly."""
    input_obj = DataInput(data=None, data_uri="s3://my-bucket/data.csv", timestamp_column="ts")

    mock_df = pd.DataFrame({"ts": ["2021-01-01"], "value": [123]})

    with patch("your_module.to_pandas", return_value=mock_df) as mock_to_pandas:
        result = await load_timeseries(input_obj)

    mock_to_pandas.assert_called_once_with(uri="s3://my-bucket/data.csv", timestamp_column="ts")
    assert result.equals(mock_df)


@pytest.mark.asyncio
async def test_load_timeseries_no_data_raises():
    """It should raise ValueError if neither data nor data_uri is provided."""
    input_obj = DataInput(data=None, data_uri=None, timestamp_column="time")

    with pytest.raises(ValueError, match="No data provided"):
        await load_timeseries(input_obj)
