from __future__ import annotations

import asyncio

from fastmcp import tool

from .datautil import load_timeseries
from .payloads import DataInput, ForecastResult


# -----------------------------------------------------------------------------
# 4. FastMCP tool definition
# -----------------------------------------------------------------------------


@tool
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
    ts = await load_timeseries(data)

    # Very simple dummy forecast: repeat last value
    last_value = ts.iloc[-1]
    forecast = [float(last_value)] * horizon

    return ForecastResult(forecast=forecast, horizon=horizon, model="NaiveRepeat")


# -----------------------------------------------------------------------------
# 5. Optional: demo/test runner
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    async def main():
        # Example 1: Inline data
        req_inline = DataInput(values=[1, 2, 3, 4, 5])
        result_inline = await forecast_tool(req_inline, horizon=3)
        print("Inline forecast:", result_inline)

        # Example 2: Remote data URI
        # req_uri = DataInput(data_uri="https://example.com/timeseries.csv")
        # result_uri = await forecast_tool(req_uri, horizon=5)
        # print("URI forecast:", result_uri)

    asyncio.run(main())
