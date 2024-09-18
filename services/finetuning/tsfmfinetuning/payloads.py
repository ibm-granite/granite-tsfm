# Copyright contributors to the TSFM project
#
"""Payload definitions for tsfmfinetuning"""

# WARNING: DO NOT IMPORT util here or else you'll get a circular dependency

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .model_parameters import (
    TinyTimeMixerParameters,
)


class TuneTypeEnum(str, Enum):
    full = "full"
    linear_probe = "linear_probe"


class AsyncCallReturn(BaseModel):
    job_id: str = Field(
        description="""A unique job identifier that can later be used in
        calls to jobstatus to obtain information about the asynchronous job."""
    )


class TrainerArguments(BaseModel):
    """Class representing HF trainer arguments"""

    learning_rate: float = 0.0
    num_train_epochs: int = 100
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = per_device_train_batch_size
    # dataloader_num_workers: int = 8
    metric_for_best_model: str = "eval_loss"
    early_stopping: bool = True
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.001


class BaseTuneInput(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    tune_type: TuneTypeEnum = TuneTypeEnum.linear_probe
    trainer_args: TrainerArguments = Field(default=TrainerArguments())

    tune_prefix: str = Field(
        pattern=".*",
        min_length=1,
        max_length=100,
        description="A prefix used when saving a tuned model.",
        example="<a_prefix>",
    )
    fewshot_fraction: float = Field(
        default=1.0,
        description="Fraction of data to use for fine tuning.",
    )
    random_seed: Optional[int] = Field(default=None, description="Random seed set prior to fine tuning.")

    @field_validator("fewshot_fraction")
    @classmethod
    def check_valid_fraction(cls, v: float) -> float:
        if (v > 1) or (v <= 0):
            raise ValueError("`fewshot_fraction` should be a valid fraction between 0 and 1")
        return v


class BaseDataInput(BaseModel):
    data: str = Field(
        description="A supported URI pointing to readable finetuning data.",
        min_length=1,
        max_length=5_00,
        pattern=".*",
        example="file:///persistent_volume/claim/path/data.csv",
    )
    timestamp_column: str = Field(
        description="A valid column in the data that should be treated as the timestamp.",
        pattern=".*",
        min_length=1,
        max_length=100,
        example="date",
    )
    id_columns: List[str] = Field(
        description="Columns that define a unique key for time series.",
        default_factory=list,
        max_length=10,
        example=["ID1", "ID2"],
        min_length=0,
    )
    freq: Optional[str] = Field(
        description="""A freqency indicator for the given timestamp_column.
        See https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#period-aliases for a description of the allowed values.
        If not provided, we will attempt to infer it from the data.""",
        default=None,
        pattern=r"\d+[B|D|W|M|Q|Y|h|min|s|ms|us|ns]|^\s*$",
        min_length=0,
        max_length=100,
        example="1h",
    )


class ForecastingDataInput(BaseDataInput):
    target_columns: List[str] = Field(
        default_factory=list,
        max_length=500,
        min_length=0,
        example=["HUFL", "HULL"],
        description="An array of column headings which constitute the target variables.",
    )
    observable_columns: List[str] = Field(
        default_factory=list,
        max_length=500,
        min_length=0,
        example=["OBS1", "OBS2"],
        description="An optional array of column headings which constitute the observable variables.",
    )
    control_columns: List[str] = Field(
        default_factory=list,
        max_length=500,
        min_length=0,
        example=["CNTRL1", "CNTRL2"],
        description="An optional array of column headings which constitute the control variables.",
    )
    conditional_columns: List[str] = Field(
        default_factory=list,
        max_length=500,
        min_length=0,
        example=["CONDL1", "CONDL2"],
        description="An optional array of column headings which constitute the conditional variables.",
    )
    static_categorical_columns: List[str] = Field(
        default_factory=list,
        max_length=500,
        min_length=0,
        example=["SCV1", "SCV2"],
        description="An optional array of column headings which constitute the static categorical variables.",
    )

    prediction_length: Optional[int] = Field(
        description="The prediction length for the forecast.",
        default=None,
    )
    context_length: Optional[int] = Field(
        description="Context length of the forecast.",
        default=None,
    )


class ForecastingTuneInput(BaseTuneInput, ForecastingDataInput):
    model_config = ConfigDict(extra="forbid")
    validation_data: str = Field(
        description="A URI pointing to readable data or a base64 encoded string of data for validation data.",
        max_length=5_000_000,
        min_length=0,
        pattern=".*",
        default="",
    )


class TinyTimeMixerForecastingTuneInput(ForecastingTuneInput):
    model_config = ConfigDict(protected_namespaces=())
    model_parameters: TinyTimeMixerParameters = Field(default=TinyTimeMixerParameters())
