import logging

from .datautil import load_timeseries
from .payloads import DataInput, ForecastResult


LOGGER = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 4. FastMCP tool definitions
# -----------------------------------------------------------------------------


def forecast_tool(data: DataInput) -> ForecastResult:
    """See docstring in app.py for details."""
    ts = load_timeseries(data)

    # Very simple dummy forecast: repeat last value
    last_value = ts.iloc[-1]
    forecast = [float(last_value)] * 10

    return ForecastResult(forecast_data=[forecast, forecast])
