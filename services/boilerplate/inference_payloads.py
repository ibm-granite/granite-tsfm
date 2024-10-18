# Copyright contributors to the TSFM project
#
"""Payload definitions for tsfminference"""

from typing import Annotated, Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# WARNING: DO NOT IMPORT util here or else you'll get a circular dependency

EverythingPatternedString = Annotated[str, Field(min_length=0, max_length=100, pattern=".*")]


class BaseMetadataInput(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    timestamp_column: str = Field(
        description="A valid column in the data that should be treated as the timestamp."
        " Although not absolutely necessary, if using calendar dates,"
        " users should consider using a format that"
        " includes a UTC offset(e.g., '2024-10-18T01:09:21.454746+00:00'). This will avoid"
        " potential issues such as duplicate dates appearing due to daylight savings"
        " change overs.",
        pattern=".*",
        min_length=1,
        max_length=100,
        example="date",
    )
    id_columns: List[EverythingPatternedString] = Field(
        description="Columns that define a unique key for time series."
        " This is similar to a compound primary key in a database table.",
        default_factory=list,
        max_length=10,
        example=["ID1", "ID2"],
        min_length=0,
    )
    freq: Optional[str] = Field(
        description="A freqency indicator for the given timestamp_column."
        " See https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#period-aliases"
        " for a description of the allowed values. If not provided, we will attempt to infer it from the data.",
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
        description="An array of column headings which constitute the target variables in the data."
        " These are the data that will be forecasted.",
    )
    observable_columns: List[EverythingPatternedString] = Field(
        default_factory=list,
        max_length=500,
        min_length=0,
        example=["OBS1", "OBS2"],
        description="An optional array of column headings which identify"
        " the observables in the data. Observables are features (commonly called channels in timeseries forecasting problems)"
        " which we have knowledge about in the past and future. For example, weather"
        " conditions such as temperature or precipitation may be known or estimated in the future"
        " but cannot be changed. This field supports specialized uses of timeseries forecasting"
        " that the average user is unlikely to encounter.",
    )
    control_columns: List[EverythingPatternedString] = Field(
        default_factory=list,
        max_length=500,
        min_length=0,
        example=["CNTRL1", "CNTRL2"],
        description="An optional array of column headings which identify the control channels in the input."
        " Control channels are similar to observable channels, except that future values may be controlled."
        " For example, the discount percentage of a particular product is known and controllable in the future."
        " Similar to observable_columns, control_columns is intended for advanced use cases not typical"
        " in most timeseries forecasting problems.",
    )
    conditional_columns: List[EverythingPatternedString] = Field(
        default_factory=list,
        max_length=500,
        min_length=0,
        example=["CONDL1", "CONDL2"],
        description="An optional array of column headings which constitute the conditional variables."
        " The conditional_columns in the data are those known in the past, but not known in the future.",
    )
    static_categorical_columns: List[EverythingPatternedString] = Field(
        default_factory=list,
        max_length=500,
        min_length=0,
        example=["SCV1", "SCV2"],
        description="An optional array of column headings which identify"
        " categorical-valued channels in the input which are fixed over time.",
    )


class ForecastingParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    prediction_length: Optional[int] = Field(description="The prediction length for the forecast.", default=None)

    @field_validator("prediction_length")
    @classmethod
    def check_prediction_length(cls, v: int) -> float:
        if v is not None and v < 1:
            raise ValueError(
                "If specified, `prediction_length` must be an integer >=1"
                " and no more than the model default prediction length."
                " When omitted the model default prediction_length will be used."
            )
        return v


class BaseInferenceInput(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_id: str = Field(
        description="A model identifier.",
        pattern=r"^\S+$",
        min_length=1,
        max_length=256,
        example="ibm/tinytimemixer-monash-fl_96",
    )


class ForecastingInferenceInput(BaseInferenceInput):
    schema: ForecastingMetadataInput = Field(
        description="An object of ForecastingMetadataInput that contains the schema" " metadata of the data input.",
    )

    parameters: ForecastingParameters

    data: Dict[str, List[Any]] = Field(description="Data", min_length=1)

    future_data: Optional[Dict[str, List[Any]]] = Field(
        description="Exogenous or supporting features that extend into"
        " the forecasting horizon (e.g., a weather forecast or calendar"
        " of special promotions) which are known in advance.",
        default=None,
    )


class PredictOutput(BaseModel):
    model_id: str = Field(
        description="Model identifier for the model that produced the prediction.",
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
