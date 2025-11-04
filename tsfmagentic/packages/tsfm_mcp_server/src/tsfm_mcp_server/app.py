from fastmcp import FastMCP

from tsfm_mcp_server.payloads import DataInput, ForecastResult
from tsfm_mcp_server.tools import forecast_tool as iforecast_tool


# Create the FastMCP server instance
mcp = FastMCP(
    name="tsfm_mcp_server",
    version="1.0.0",
    instructions="""This server provides tools for single and multivariate time series analytics. Use it to provide the following functionalities:
    - Time series forecasting
    """,
)


@mcp.tool()
async def forecast_tool(data: DataInput, forecast_as_data: bool = True) -> ForecastResult:
    return await iforecast_tool(data, forecast_as_data)


if __name__ == "__main__":
    # Start the MCP server
    mcp.run()
