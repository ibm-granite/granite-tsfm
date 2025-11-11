from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


# -----------------------------------------------------------------------------
# Input model
# -----------------------------------------------------------------------------


class DataInput(BaseModel):
    """
    Input payload for tools that operate on time series data loaded from an external URI.

    The dataset referenced by ``data_uri`` should be a tabular file (e.g., CSV)
    containing at least the following elements:

    - **timestamp_column** (required): The name of the column containing timestamps.
      Timestamps may be ISO-8601 strings or numeric epoch values.
    - **target_columns** (required): One or more numeric columns representing the
      target variable(s) to forecast or analyze.
    - **identifier_column** (optional): A column distinguishing multiple time series
      within the same dataset.

    Example Input:
        >>> DataInput(
        ...     data_uri="file://./data.csv",
        ...     timestamp_column="timestamp",
        ...     target_columns=["value1", "value2"],
        ...     identifier_column="identifier"
        ... )


    Example CSV (comma-separated, with header) format for multivariate forecast:
        timestamp,identifier,value1,value2
        2024-11-03T10:15:00Z,id1,14.8,28.4
        2024-11-03T10:20:00Z,id1,15.2,29.1
        2024-11-03T10:25:00Z,id1,15.6,29.8
        2024-11-03T10:30:00Z,id1,15.9,30.2

    **Column headers are required and you must use a comma delimiter.**
    """

    model_config = ConfigDict(json_schema_extra={"required": ["data_uri", "timestamp_column", "target_columns"]})

    data_uri: str = Field(
        description=(
            "URI pointing to the external dataset. Supported schemes include `file://`, "
            "with others (e.g., `s3://`, `gcs://`) planned in the future. "
            "The referenced file should be in CSV format with tabular structure."
        ),
        json_schema_extra={"example": "file://./data.csv"},
    )

    timestamp_column: str = Field(
        description="**Required.** The name of the column containing timestamps.",
        json_schema_extra={"example": "timestamp"},
    )

    identifier_column: Optional[str] = Field(
        default=None,
        description=("Optional. The name of the column distinguishing multiple series within the dataset."),
        json_schema_extra={"example": "identifier"},
    )

    target_columns: List[str] = Field(
        description="**Required.** Columns containing target values for forecasting or analysis.",
        json_schema_extra={"example": ["value"]},
    )

    forecast_length: Optional[int] = Field(
        default=96,
        description="Forecasting forecast_length (number of future time steps). Defaults to 96.",
        json_schema_extra={"example": 96},
    )


# -----------------------------------------------------------------------------
# Output model
# -----------------------------------------------------------------------------


class ForecastResult(BaseModel):
    """
    Output payload for forecast results produced by time series tools.

    The forecast results are provided as a **URI** reference that points to a
    tabular file (e.g., CSV) containing the predicted values for each target column.
    This file will typically match the format of the input dataset, beginning
    immediately after the last timestamp observed in the input.

    The model may also include optional context or metadata describing the forecast or
    any error conditions that may have occurred during processing.

    Example CSV (comma-separated, with header) format for multivariate forecast:
        timestamp,identifier,value1,value2
        2024-11-03T10:35:00Z,id1,14.8,28.4
        2024-11-03T10:40:00Z,id1,15.2,29.1
        2024-11-03T10:45:00Z,id1,15.6,29.8
        2024-11-03T10:50:00Z,id1,15.9,30.2

    Example instance:
        >>> ForecastResult(
        ...     forecast_uri="file://./forecast.csv",
        ...     context="Forecast generated using default temporal model with forecast_length=96."
        ... )
    """

    model_config = ConfigDict(json_schema_extra={"required": ["forecast_uri"]})

    forecast_uri: str = Field(
        description=(
            "URI pointing to the forecasted results. The file should contain the "
            "predicted future values starting one step after the last timestamp "
            "in the input data. The format typically matches the input dataset "
            "(e.g., CSV with the same column names)."
        ),
        json_schema_extra={"example": "file://./forecast.csv"},
    )

    context: Optional[str] = Field(
        default=None,
        description="Optional context, notes, or metadata about the forecast.",
        json_schema_extra={
            "example": (
                "Forecast generated using default temporal model (forecast_length=96). "
                "WARNING: results may be less accurate due to short context length."
            )
        },
    )
