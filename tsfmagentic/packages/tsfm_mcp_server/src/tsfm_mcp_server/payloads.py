from typing import List, Optional

from pydantic import BaseModel, Field


# -----------------------------------------------------------------------------
# Input model
# -----------------------------------------------------------------------------


class DataInput(BaseModel):
    """
    Input payload for tools that accepts either a string
    of tab separated CSV data or an external URI reference.

    The model is intentionally flexible: you may provide the raw CSV data as a string, or point to a remote source via
    ``data_uri``.  **Exactly one should be supplied** – if
    both are present, the CSV content takes
    precedence and the URI is ignored. In either case, the data should be tabular and contain one or more of the following elements:

    * A timestamp column which should contain either a string that's parsable as a pandas datetime type,
    or a numeric value representing epoch time or a time offset (e.g, 0, 1, 2, ...).
    * One or more numeric measurement columns representing the target values for tasks such as forecasting.
    * A identifier column is optional, but may be included to distinguish multiple time series within the same dataset.

    See the detailed payload documentation for more information regarding which columns are required for specific tasks.

    Your data needs to contain sufficient historical context for the task at hand (e.g., forecasting). Typically this is at least 96 historical points but can be
    as much as 1024 historical points for improved accuracy. This tool will internally select the appropriate choice of model based on the historical context length.

    Example (inline data)
        >>> DataInput(
                data="timestamp\tidentifier\tvalue\n2024-11-03T10:00:00Z\tid1\t12.3\n2024-11-03T10:01:00Z\tid1\t7.8\n2024-11-03T10:02:00Z\tid1\t9.1",
                timestamp_column="timestamp",
                target_columns=["value"],
                identifier_column="identifier"
            )

    Example (URI)
        # echo "timestamp,identifier,value\n2024-11-03T10:00:00Z,id1,12.3\n2024-11-03T10:01:00Z,id1,7.8\n2024-11-03T10:02:00Z,id1,9.1" > data.csv
        >>> DataInput(data_uri="file://./data.csv",
                timestamp_column="timestamp",
                target_columns=["value"],
                identifier_column="identifier"
            )
    """

    data: Optional[str] = Field(
        default=None,
        description=(
            "Data as a string (currently only tab-separated CSV is supported with json support coming soon). If provided, this takes precedence over ``data_uri``."
        ),
        json_schema_extra={
            "example": (
                "timestamp\tidentifier\tvalue\n"
                "2024-11-03T10:00:00Z\tid1\t12.3\n"
                "2024-11-03T10:01:00Z\tid1\t7.8\n"
                "2024-11-03T10:02:00Z\tid1\t9.1"
            )
        },
    )

    data_uri: Optional[str] = Field(
        default=None,
        description=(
            "A URI that points to an external dataset. Currently supported schemes"
            " are `file://` with additional schemes such as `https://`, `http://`, `s3://`,"
            " and `gcs://` to come later. Supported file formats are CSV with arrow"
            " table support to follow."
        ),
        json_schema_extra={"example": "file://./data.csv"},
    )

    timestamp_column: Optional[str] = Field(
        default=None,
        description=("The name of the column containing timestamps in the dataset."),
        json_schema_extra={"example": "timestamp"},
    )

    identifier_column: Optional[str] = Field(
        default=None,
        description=(
            "The name of the column containing identifiers in the dataset. This column distinguishes multiple time series within the same dataset."
        ),
        json_schema_extra={"example": "identifier"},
    )

    timestamp_column: Optional[str] = Field(
        default=None,
        description=("The name of the column containing timestamps in the dataset."),
        json_schema_extra={"example": "timestamp"},
    )

    target_columns: List[str] = Field(
        default=None,
        description=(
            "The names of the columns containing target values in the dataset. These are the values to be forecasted or analyzed."
        ),
        json_schema_extra={"example": ["value"]},
    )

    horizon: Optional[int] = Field(
        default=10,
        description="The forecasting horizon (when performing forecasting).",
        json_schema_extra={"example": 10},
    )


# -----------------------------------------------------------------------------
# Output model
# -----------------------------------------------------------------------------


class ForecastResult(BaseModel):
    """Example forecast output. This can contain either inline forecast data or a URI to the results.
    The behavior is dependent on the boolean parameter `forecast_as_data` in the definition of forecast_tool."""

    forecast_data: Optional[List[List[float]]] = Field(
        default=None,
        description=(
            "The forecasted values for each target column for the given horizon. The values represent future values starting after the last timestamp in the input data."
        ),
        json_schema_extra={"example": [[12.5, 13.0, 13.5], [7.8, 8.1, 8.4]]},  # two target columns, three steps each
    )

    forecast_uri: Optional[str] = Field(
        default=None,
        description=(
            "A URI that points to the forecasted values. The saved results will be in the same format as the input data (e.g., CSV). With just the forecasted values starting one step after the last input timestamp."
        ),
        json_schema_extra={"example": "file://./forecast.csv"},
    )

    context: Optional[str] = Field(
        description="Optional context or metadata about the forecast.",
        default=None,
        json_schema_extra={"example": "WARNING: this forecast has the potential to be inaccurate."},
    )
