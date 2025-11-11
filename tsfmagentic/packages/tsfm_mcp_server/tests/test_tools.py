import sys


def test_forecast_tool():
    import pandas as pd
    from tsfm_mcp_server.payloads import DataInput, ForecastResult
    from tsfm_mcp_server.tools import forecast_tool

    # Create a simple time series DataFrame and write to a temp CSV file
    df = pd.DataFrame(
        {
            "date": pd.date_range(start="2023-01-01", periods=1024, freq="h"),
            "value": range(1024),
        }
    )
    import tempfile

    with tempfile.NamedTemporaryFile(prefix="test_data", suffix=".csv", delete=False) as tmp_file:
        df.to_csv(tmp_file.name, index=False)
        data_input = DataInput(
            data_uri=f"file:///{tmp_file.name}" if sys.platform == "win32" else f"file://{tmp_file.name}",
            timestamp_column="date",
            target_columns=["value"],
            horizon=96,
        )
        result: ForecastResult = forecast_tool(data_input)
        assert result.forecast_uri.startswith("file://")
        assert "forecast_result" in result.forecast_uri

        # Load the forecasted results and check contents
        forecast_path = (
            result.forecast_uri.replace("file:///", "")
            if sys.platform == "win32"
            else result.forecast_uri.replace("file://", "")
        )
        forecast_df = pd.read_csv(forecast_path)
        assert len(forecast_df) == 96
        assert "date" in forecast_df.columns
        assert "value" in forecast_df.columns
