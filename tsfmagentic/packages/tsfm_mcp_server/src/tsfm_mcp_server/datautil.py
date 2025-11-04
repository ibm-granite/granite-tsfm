import os
import tempfile

import pandas as pd
from tsfmfinetuning.ioutils import to_pandas

from .payloads import DataInput


def load_timeseries(input: DataInput) -> pd.DataFrame:
    """Load time series data either from inline arrays or a URI."""
    tmp_file = None
    try:
        if input.data is not None:
            if isinstance(input.data, bytes):
                mode = "wb"
            else:
                mode = "w"
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode=mode) as tmp_file_handle:
                tmp_file_handle.write(input.data)
                tmp_file = tmp_file_handle  # Save reference for cleanup
            uri = f"file://{tmp_file.name}"
        elif input.data_uri:
            uri = input.data_uri
        else:
            raise ValueError("No data provided.")

        return to_pandas(uri=uri, timestamp_column=input.timestamp_column)
    finally:
        if tmp_file:
            try:
                os.unlink(tmp_file.name)
            except Exception as e:
                import logging

                logging.exception(f"Failed to delete temporary file {tmp_file.name}: {e}")
