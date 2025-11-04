from .datautil import load_timeseries
from .payloads import DataInput, ForecastResult


# -----------------------------------------------------------------------------
# 4. FastMCP tool definitions
# -----------------------------------------------------------------------------


async def forecast_tool(data: DataInput, forecast_as_data: bool = True) -> ForecastResult:
    """
    A forecasting tool definition that uses the hybrid DataInput data model.

    Args:
        data: A DataInput object containing either inline numeric arrays or a URI to tabular data.
        forecast_as_data: Boolean flag indicating whether to return the forecast as inline data or a URI pointing to the results.
        Currently only local file system CSV output is supported.

    Returns:
        ForecastResult object with forecasted values.
    """
    ts = load_timeseries(data)

    # Very simple dummy forecast: repeat last value
    last_value = ts.iloc[-1]
    forecast = [float(last_value)] * 10

    return ForecastResult(forecast_data=[forecast, forecast])
