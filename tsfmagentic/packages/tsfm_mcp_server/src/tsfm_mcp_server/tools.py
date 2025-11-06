import logging
import sys
import tempfile

import pandas as pd
from tsfminference.ioutils import path_to_uri

from .datautil import load_timeseries
from .payloads import DataInput, ForecastResult


LOGGER = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 4. FastMCP tool definitions
# -----------------------------------------------------------------------------


def forecast_tool(data: DataInput) -> ForecastResult:
    """See docstring in app.py for details."""

    # we should be pydantic validated at this point so no need to re-validate
    data.data_uri = data.data_uri.strip().upper()

    # some llms consistently send an incorrect uri for win32 platforms so fix it here
    if sys.platform == "win32":
        data.data_uri = (
            data.data_uri.replace("FILE://", "FILE:///") if data.data_uri.find("FILE:///") < 0 else data.data_uri
        )

    ts = load_timeseries(data)

    LOGGER.info(f"Loaded time series data with {len(ts)} rows and columns: {ts.columns.tolist()}")

    # Perform dummy forecast (to be replaced with real model inference

    # take the last 96 rows of ts
    dummy_ts = ts.tail(96)
    timestamp_col = data.timestamp_column
    # get the last timestamp in ts
    last_timestamp = ts[timestamp_col].iloc[-1]
    # replace the timestamp column in dummy_ts with last_timestamp + 1 hour increments
    new_timestamps = [last_timestamp + pd.Timedelta(hours=i + 1) for i in range(96)]
    dummy_ts[timestamp_col] = new_timestamps
    # write dummy_ts to a csv file in our system directory with the prefix "forecast_result" using mkstemp

    with tempfile.NamedTemporaryFile(prefix="forecast_result", suffix=".csv", delete=False) as tmp_file:
        dummy_ts.to_csv(tmp_file.name, index=False)
        LOGGER.info(f"Written dummy forecast results to {tmp_file.name}")
        return ForecastResult(
            forecast_uri=path_to_uri(tmp_file.name),
            context=f"CSV file with dummy forecast results written to temporary {path_to_uri(tmp_file.name)}.",
        )
