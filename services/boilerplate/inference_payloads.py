# Copyright contributors to the TSFM project
#
"""Payload definitions for tsfminference"""

from typing import Annotated, Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


# WARNING: DO NOT IMPORT util here or else you'll get a circular dependency

EverythingPatternedString = Annotated[str, Field(min_length=0, max_length=100, pattern=".*")]


class BaseMetadataInput(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    timestamp_column: str = Field(
        description="A valid column in the data that should be treated as the timestamp.",
        pattern=".*",
        min_length=1,
        max_length=100,
        example="date",
    )
    id_columns: List[EverythingPatternedString] = Field(
        description="Columns that define a unique key for time series.",
        default_factory=list,
        max_length=10,
        example=["ID1", "ID2"],
        min_length=0,
    )
    freq: Optional[str] = Field(
        description="""A freqency indicator for the given timestamp_column. See https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#period-aliases for a description of the allowed values. If not provided, we will attempt to infer it from the data.""",
        default=None,
        pattern=r"\d+[B|D|W|M|Q|Y|h|min|s|ms|us|ns]|^\s*$",
        min_length=0,
        max_length=100,
        example="1h",
    )


class ForecastingMetadataInput(BaseMetadataInput):
    target_columns: List[EverythingPatternedString] = Field(
        default_factory=list,
        max_length=500,
        min_length=0,
        example=["TARGET1", "TARGET1"],
        description="An array of column headings which constitute the target variables.",
    )
    observable_columns: List[EverythingPatternedString] = Field(
        default_factory=list,
        max_length=500,
        min_length=0,
        example=["OBS1", "OBS2"],
        description="An optional array of column headings which constitute the observable variables.",
    )
    control_columns: List[EverythingPatternedString] = Field(
        default_factory=list,
        max_length=500,
        min_length=0,
        example=["CNTRL1", "CNTRL2"],
        description="An optional array of column headings which constitute the control variables.",
    )
    conditional_columns: List[EverythingPatternedString] = Field(
        default_factory=list,
        max_length=500,
        min_length=0,
        example=["CONDL1", "CONDL2"],
        description="An optional array of column headings which constitute the conditional variables.",
    )
    static_categorical_columns: List[EverythingPatternedString] = Field(
        default_factory=list,
        max_length=500,
        min_length=0,
        example=["SCV1", "SCV2"],
        description="An optional array of column headings which constitute the static categorical variables.",
    )


class ForecastingParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    prediction_length: Optional[int] = Field(
        description="The prediction length for the forecast.",
        default=None,
    )


class BaseInferenceInput(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_id: str = Field(
        description="A model identifier.",
        pattern=r"^\S+$",
        min_length=1,
        max_length=100,
        example="ibm/tinytimemixer-monash-fl_96",
    )


class ForecastingInferenceInput(BaseInferenceInput):
    schema: ForecastingMetadataInput

    parameters: ForecastingParameters

    data: Dict[str, List[Any]] = Field(
        description="Data",
    )

    future_data: Optional[Dict[str, List[Any]]] = Field(description="Future data", default=None)

    @model_validator(mode="before")
    @classmethod
    def grpcswaparoos(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "future_data" in data and data["future_data"] is None:
                data["future_data"] = ""
        return data


class PredictOutput(BaseModel):
    model_id: str = Field(
        description="Model ID for the model that produced the prediction.",
        default=None,
    )
    created_at: str = Field(
        description="Timestamp indicating when the prediction was created. ISO 8601 format.",
        default=None,
    )
    results: List[Dict[str, List[Any]]] = Field(
        description="List of prediction results.",
        default=None,
    )
