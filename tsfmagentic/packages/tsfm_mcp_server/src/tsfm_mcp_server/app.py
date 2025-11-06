import logging

from fastmcp import FastMCP

from tsfm_mcp_server.payloads import DataInput, ForecastResult
from tsfm_mcp_server.tools import forecast_tool as iforecast_tool


# --------------------------------------------------------------------
# Redirect all logging output (including warnings, etc.) to a file
# so nothing contaminates stdout, which is reserved for MCP messages.
# --------------------------------------------------------------------
LOG_FILE = "tsfm_mcp_server.log"

# Remove any default handlers that may write to stdout
root_logger = logging.getLogger()
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

# Configure logging to go only to a file
logging.basicConfig(
    level=logging.DEBUG,  # or DEBUG, depending on your needs
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8")],
)

# Also silence noisy libraries that log to stdout by default
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("uvicorn").setLevel(logging.ERROR)
logging.getLogger("fastapi").setLevel(logging.ERROR)

LOGGER = logging.getLogger(__name__)

# Create the FastMCP server instance
mcp = FastMCP(
    name="tsfm_mcp_server",
    version="1.0.0",
    instructions="""This server provides tools for single and multivariate time series analytics. Use it to provide the following functionalities:
    - Time series forecasting
    """,
)


@mcp.tool(
    name="forecast_timeseries",
    description=(
        """Forecasts a time series from a URI that points to CSV data.
         At present, the only acceptable form of uri is a file:// URI.
         You must include both `timestamp_column` and `target_columns`
         in your input payload. You should confirm with the user that your
         choice of these columns is correct before proceeding with the forecast.
         The forecast results will be written to a temporary file in CSV format
         in the system's temporary directory, with a unique filename prefixed by
         "forecast_result". The output file will be in the same format as
         the input file and will contain predicted values for each target column,
         beginning immediately after the last timestamp in the input data.

         You will receive a `ForecastResult` object containing a `forecast_uri`
         that points to the generated results file, along with optional metadata
         """
    ),
)
async def forecast_tool(input_pydantic_model: DataInput) -> ForecastResult:
    """
    Generate time series forecasts from tabular data referenced by a URI.

    This tool accepts a `DataInput` object containing a `data_uri` that points to
    a time series dataset (e.g., a CSV file). The tool loads the dataset, performs
    forecasting for each target column, and returns a `ForecastResult` object that
    includes a `forecast_uri` pointing to the generated results file.

    The forecast results are always written to the system's temporary directory.
    Each output file is assigned a unique filename with the prefix `"forecast_result"`,
    and saved in CSV format. The file contains predicted values for each target
    column, beginning immediately after the last timestamp in the input data.

    Args:
        input (DataInput): Input specification containing a URI reference to the source data,
            column mappings, and optional forecasting parameters (e.g., `horizon`).

    Returns:
        ForecastResult: A model containing the URI to the forecasted results and
        optional metadata describing the forecast context.

    Example:
        >>> forecast_tool(
        ...     DataInput(
        ...         data_uri="file://./data.csv",
        ...         timestamp_column="timestamp",
        ...         target_columns=["value"],
        ...         identifier_column="identifier",
        ...         horizon=4
        ...     )
        ... )
        ForecastResult(
        ...     forecast_uri="file:///tmp/forecast_result_abcd1234.csv",
        ...     context="Forecast generated using default temporal model with horizon=4."
        ... )
    """
    LOGGER.info("Received forecast_tool request with input_pydantic_model: %s", input_pydantic_model)
    return await iforecast_tool(input_pydantic_model)


if __name__ == "__main__":
    # Start the MCP server
    LOGGER.info("Starting TSFM MCP server...")
    mcp.run()
