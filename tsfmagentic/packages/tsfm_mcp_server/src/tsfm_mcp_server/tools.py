import logging
import sys
import tempfile

import pandas as pd
from tsfminference.inference import InferenceRuntime
from tsfminference.inference_payloads import (
    ForecastingInferenceInput,
    ForecastingMetadataInput,
    ForecastingParameters,
    PredictOutput,
)
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

    df = load_timeseries(data)
    schema: ForecastingMetadataInput = ForecastingMetadataInput(
        timestamp_column=data.timestamp_column,
        id_columns=[data.identifier_column] if data.identifier_column else [],
        target_columns=data.target_columns,
    )
    params: ForecastingParameters = ForecastingParameters(
        prediction_length=data.forecast_length if data.forecast_length else None
    )
    input: ForecastingInferenceInput = ForecastingInferenceInput(
        model_id="ttm-1024-96-r2", data=df.to_dict(orient="list"), schema=schema, parameters=params
    )

    runtime: InferenceRuntime = InferenceRuntime()
    po: PredictOutput = runtime.forecast(input=input)
    results = pd.DataFrame.from_dict(po.results[0])

    with tempfile.NamedTemporaryFile(prefix="forecast_result", suffix=".csv", delete=False) as tmp_file:
        results.to_csv(tmp_file.name, index=False)
        LOGGER.info(f"Written forecast results to {tmp_file.name}")
        return ForecastResult(
            forecast_uri=path_to_uri(tmp_file.name),
            context=f"CSV file with forecasted values written to {path_to_uri(tmp_file.name)}.",
        )
