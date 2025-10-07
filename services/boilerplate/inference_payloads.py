# Copyright contributors to the TSFM project
#
"""Payload definitions for tsfminference"""

from typing import Annotated, Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# WARNING: DO NOT IMPORT util here or else you'll get a circular dependency

EverythingPatternedString = Annotated[str, Field(min_length=0, max_length=100, pattern=r"^\S.*\S$|^\S$")]


class BaseMetadataInput(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    timestamp_column: str = Field(
        description="A valid column in the data that should be treated as the timestamp."
        " Although not absolutely necessary, if using calendar dates "
        " (simple integer time offsets are also allowed),"
        " users should consider using a format such as ISO 8601 that"
        " includes a UTC offset (e.g., '2024-10-18T01:09:21.454746+00:00')."
        " This will avoid potential issues such as duplicate dates appearing"
        " due to daylight savings change overs. There are many date formats"
        " in existence and inferring the correct one can be a challenge"
        " so please do consider adhering to ISO 8601.",
        pattern=r"^\S.*\S$|^\S$",
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
    scaling_id_columns: Optional[List[EverythingPatternedString]] = Field(
        description="Columns that define unique subsets of the time series data"
        " over which separate scaling factors should be applied. These columns"
        " must be a subset of the id_columns.",
        default=None,
        max_length=10,
        example=["ID2"],
        min_length=0,
    )
    freq: Optional[str] = Field(
        description="A frequency indicator for the given timestamp_column."
        " See https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#period-aliases"
        " for a description of the allowed values. If not provided, we will attempt to infer it from the data.",
        default=None,
        pattern=r"^\d*\.?\d*(B|D|W|M|Q|Y|h|min|s|ms|us|ns)?$",
        min_length=0,
        max_length=100,
        example="1h",
    )


class ForecastingMetadataInput(BaseMetadataInput):
    target_columns: List[EverythingPatternedString] = Field(
        default_factory=list,
        max_length=500,
        min_length=0,
        example=["TARGET1", "TARGET2"],
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

    categorical_columns: List[EverythingPatternedString] = Field(
        default_factory=list,
        max_length=500,
        min_length=0,
        example=["CV1", "CV2"],
        description="An optional array of column headings which identify"
        " categorical-valued channels in the input which can vary over time.",
    )


class BaseParameters(BaseModel):
    model_config = ConfigDict(extra="allow", protected_namespaces=())

    inference_batch_size: Optional[int] = Field(
        description="The batch size used during inference."
        " When multiple time series are present, the inference will be"
        " conducted in batches. If not specified, the model default batch"
        " size will be used.",
        default=None,
    )

    @field_validator("inference_batch_size")
    @classmethod
    def check_inference_batch_size(cls, v: Optional[int]) -> float:
        if v is not None and v < 1:
            raise ValueError(
                "If specified, `inference_batch_size` must be an integer >=1."
                " When omitted the model default inference_batch_size will be used."
            )
        return v


class ForecastingParameters(BaseParameters):
    prediction_length: Optional[int] = Field(
        description="The prediction length for the forecast."
        " The service will return this many periods beyond the last"
        " timestamp in the inference data payload."
        " If specified, `prediction_length` must be an integer >=1"
        " and no more than the model default prediction length."
        " When omitted the model default prediction_length will be used.",
        default=None,
    )

    prediction_quantiles: Optional[List[float]] = Field(
        description="A list of prediction quantiles."
        " If given, the service will return probabilistic predictions for each target."
        " For example, passing [0.1, 0.9] will produce two additional prediction"
        " vectors for each target representing the 10th and 90th percentile values"
        " for each point forecast over the prediction period."
        " When omitted only point forecasts will be returned."
        " All values given must be >0.0 and <1.0."
        " If you use the parameter, you must also provide calibration data via"
        " the `quantile_calibration_data` field.",
        default=None,
    )

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

    @field_validator("prediction_quantiles")
    @classmethod
    def check_prediction_quantiles(cls, v: List[float]) -> List[float]:
        if v is not None and not all(0.0 < x < 1.0 for x in v):
            raise ValueError("`prediction_quantiles` must must all be > 0.0 and < 1.0.")
        return v


class BaseInferenceInput(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_id: str = Field(
        description="A model identifier.",
        pattern=r"^\S.*\S$",
        min_length=1,
        max_length=256,
        example="ibm/tinytimemixer-monash-fl_96",
    )


class ForecastingInferenceInput(BaseInferenceInput):
    schema: ForecastingMetadataInput = Field(
        description="An object of ForecastingMetadataInput that contains the schema metadata of the data input.",
    )

    parameters: ForecastingParameters = Field(
        description="additional parameters affecting behavior of the forecast.", default_factory=dict
    )

    data: Dict[str, List[Any]] = Field(
        description="A payload of data matching the schema provided."
        " Let's suppose you have columnar data that looks"
        "  like this (this happens to be csv but it could also be pandas data, for example):\n"
        """
         date,ID1,ID2,TARGET1,VAL2
         2024-10-18T01:00:21+00:00,I1,J1,1.05,10.0
         2024-10-18T01:00:22+00:00,I1,J1,1.75,10.1
         2024-10-18T01:00:21+00:00,I1,J2,2.01,12.8
         2024-10-18T01:00:22+00:00,I1,J2,2.13,13.6\n"""
        " If these data are for two timeseries (each beginning at"
        " 2024-10-18T01:00:21 and ending at 2024-10-18T01:00:22)"
        " given by the compound primary key comprised of ID1 and ID2"
        " and you wish to create predictions only for 'TARGET1',"
        " then your data and schema payload would like like this:\n"
        """
        {
            "schema": {
                "timestamp_column": "date",
                "id_columns": [
                    "ID1",
                    "ID2"
                ],
                "target_columns": [
                    "TARGET1"
                ]
            },
            "data": {
                "date": [
                    "2024-10-18T01:00:21+00:00",
                    "2024-10-18T01:00:22+00:00",
                    "2024-10-18T01:00:21+00:00",
                    "2024-10-18T01:00:22+00:00"
                ],
                "ID1": [
                    "I1",
                    "J1",
                    "I1",
                    "J1"
                ],
                "ID2": [
                    "I1",
                    "J2",
                    "I1",
                    "J2"
                ],
                "TARGET1": [
                    1.05,
                    1.75,
                    2.01,
                    2.13
                ],
                "VAL2": [
                    10.0,
                    10.1,
                    12.8,
                    13.6
                ]
            }
        }\n"""
        "Note that we make no mention of `VAL2` in the schema which means that it will"
        " effectively be ignored by the model when making forecasting predictions."
        " If no `target_columns` are specified, then all columns except `timestamp_column`"
        " will be considered to be targets for prediction. Pandas users can generate the"
        " `data` portion of this content by calling DataFrame.to_dict(orient='list')."
        " The service makes a few assumptions about your data:"
        " * All time series are of equal length and are uniform in nature (the time difference between"
        " two successive rows is constant);"
        " * The above implies that there are no missing rows of data;"
        " * You can not have any missing cells of data within in a row (no null NaN values either);"
        " * The above constraints mean that you are responsible for performing your own imputation on your"
        " data before passing it to the service.",
        min_length=1,
    )

    future_data: Optional[Dict[str, List[Any]]] = Field(
        description="Exogenous or supporting features that extend into"
        " the forecasting horizon (e.g., a weather forecast or calendar"
        " of special promotions) which are known in advance."
        " `future_data` would be in the same format as `data` except"
        "  that all timestamps would be in the forecast horizon and"
        " it would not include previously specified target columns."
        " Here's an example payload for such data:\n"
        """
        {
            "future_data": {
                "date": [
                    "2024-10-18T01:00:23+00:00",
                    "2024-10-18T01:00:24+00:00",
                    "2024-10-18T01:00:23+00:00",
                    "2024-10-18T01:00:24+00:00"
                ],
                "ID1": [
                    "I1",
                    "J1",
                    "I1",
                    "J1"
                ],
                "ID2": [
                    "I1",
                    "J2",
                    "I1",
                    "J2"
                ],
                "VAL2": [
                    11.0,
                    11.1,
                    13.8,
                    14.6
                ]
            }
        }\n"""
        "Note that we make no mention of `TARGET1` (from the `data` field example) and"
        " that all timestamps are in the _future_ relative to the `data` you provided."
        " Given these `future_data` the model (when supported) will factor in `VAL2` when"
        " making predictions for `TARGET1`.",
        default=None,
    )

    quantile_calibration_data: Optional[Dict[str, List[Any]]] = Field(
        description="These data are used to support the creation of"
        " probabilistic forecasts (generated when you use the `prediction_quantiles` parameter)."
        " These data should be provided if and only if `prediction_quantiles` are provided."
        " You should provide at least M + N values where M is the minimum context length for"
        " the model you are using and N is at least (@TODO). This will ensure that there are"
        " enough ground truth values to use to estimate the uncertainty of model predictions"
        " and thus provide reliable quantile values for your predicted future values."
        " Internally the service is generating a forecast for time periods beyond M and comparing"
        " this forecast to the ground truth in N to generate its estimates for quantile values."
        " The data you use should be taken from your original dataset and it should not align with what you used for"
        " `data`. M and N should be in time ordered sequence where the first element of N occurs just"
        " after the last element in M. The data are in the same format as `data`."
        " Below we show an example which is identical to that for `data` excep taken from"
        " one day earlier\n:"
        """
{
            "schema": {
                "timestamp_column": "date",
                "id_columns": [
                    "ID1",
                    "ID2"
                ],
                "target_columns": [
                    "TARGET1"
                ]
            },
            "data": {
                "date": [
                    "2024-10-17T01:00:21+00:00",
                    "2024-10-17T01:00:22+00:00",
                    "2024-10-17T01:00:21+00:00",
                    "2024-10-17T01:00:22+00:00"
                ],
                "ID1": [
                    "I1",
                    "J1",
                    "I1",
                    "J1"
                ],
                "ID2": [
                    "I1",
                    "J2",
                    "I1",
                    "J2"
                ],
                "TARGET1": [
                    1.75,
                    1.86,
                    2.31,
                    2.73
                ],
                "VAL2": [
                    10.45,
                    9.15,
                    13.86,
                    12.65
                ]
            }        }\n""",
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

    input_data_points: int = Field(description="Count of input data points.", default=None)
    output_data_points: int = Field(description="Count of output data points.", default=None)
