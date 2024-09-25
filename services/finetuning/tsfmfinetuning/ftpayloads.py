# Copyright contributors to the TSFM project
#
"""Payload definitions for tsfmfinetuning"""

# WARNING: DO NOT IMPORT util here or else you'll get a circular dependency

from enum import Enum
from typing import Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .inference_payloads import ForecastingMetadataInput, ForecastingParameters
from .model_parameters import (
    TinyTimeMixerParameters,
)


class S3aCredentials(BaseModel):
    access_key_id: str = Field(description="s3 acess key id.", pattern=".*", min_length=1, max_length=512)
    secret_access_key: str = Field(description="s3 acess key.", pattern=".*", min_length=1, max_length=512)
    endpoint: str = Field(
        default="s3.us-east.cloud-object-storage.appdomain.cloud",
        pattern=".*",
        min_length=1,
        max_length=128,
    )
    default_region: str = Field(
        default="us-east",
        pattern=".*",
        min_length=1,
        max_length=128,
    )

    @classmethod
    def from_ibmcos_creds(
        cls,
        coscreds: Dict[str, str],
        endpoint: str = "s3.us-east.cloud-object-storage.appdomain.cloud",
    ):
        return S3aCredentials(
            access_key=coscreds["cos_hmac_keys"]["access_key_id"],
            secret_key=coscreds["cos_hmac_keys"]["secret_access_key"],
            endpoint=endpoint,
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


class ForecastingTuneInput(BaseTuneInput):
    schema: ForecastingMetadataInput
    parameters: ForecastingParameters
    data: str = Field(
        description="""A supported URI pointing to data convertible to a pandas dataframe such as
        as csv file with column headings or a pyarrow feather table.""",
        max_length=500,
        min_length=0,
        pattern="file://.*",
        default="",
    )


class TinyTimeMixerForecastingTuneInput(ForecastingTuneInput):
    model_config = ConfigDict(protected_namespaces=())
    model_parameters: TinyTimeMixerParameters = Field(default=TinyTimeMixerParameters())
