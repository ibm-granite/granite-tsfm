import pandas as pd
from tsfminference.ioutils import to_pandas

from .payloads import DataInput


def load_timeseries(input: DataInput) -> pd.DataFrame:
    """Load time series data either from inline arrays or a URI."""
    return to_pandas(uri=input.data_uri, timestamp_column=input.timestamp_column)
