# Copyright contributors to the TSFM project
#
"""Preprocessor for time series data preparation"""

import copy
import datetime
import enum
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from warnings import warn

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.preprocessing import OrdinalEncoder as OrdinalEncoder_
from sklearn.preprocessing import StandardScaler as StandardScaler_
from transformers.feature_extraction_utils import FeatureExtractionMixin

from tsfm_public.toolkit.util import select_by_index, select_by_timestamp

INTERNAL_ID_COLUMN = "__id"
INTERNAL_ID_VALUE = "0"


DEFAULT_FREQUENCY_MAPPING = {"oov": 0, "half_hourly": 1, "hourly": 2, "10_minutes": 3, "15_minutes": 4}


class SKLearnFeatureExtractionBase:
    """Simple wrapper class to adapt Sklearn functions to work with the HF
    serialization approach.
    """

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary of parameters from which we can reconstruct"""
        return self.__getstate__()
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(
        cls, feature_extractor_dict: Dict[str, Any], **kwargs
    ) -> "SKLearnFeatureExtractionBase":
        """ """

        t = cls()
        t.__setstate__(feature_extractor_dict)

        return t


class StandardScaler(StandardScaler_, SKLearnFeatureExtractionBase):
    """Simple wrapper class to adapt standard scaler to work with the HF
    serialization approach.
    """


class OrdinalEncoder(OrdinalEncoder_, SKLearnFeatureExtractionBase):
    """Simple wrapper class to adapt OrdinalEncoder to work with the HF
    serialization approach.
    """


class TimeSeriesTask(enum.Enum):
    """`Enum` for the different kinds of time series datasets we need to create."""

    CLASSIFICATION = "classification"
    MASKED_PRETRAINING = "mask_pretraining"
    FORECASTING = "forecasting"
    REGRESSION = "regression"


class TimeSeriesPreprocessor(FeatureExtractionMixin):
    """A preprocessor for supporting time series modeling tasks"""

    def __init__(
        self,
        timestamp_column: Optional[str] = None,
        input_columns: List[str] = [],
        output_columns: List[str] = [],
        id_columns: List[str] = [],
        context_length: int = 64,
        prediction_length: Optional[int] = None,
        scaling: bool = False,
        scale_outputs: bool = False,
        encode_categorical: bool = True,
        categorical_columns: List[str] = [],
        time_series_task: str = TimeSeriesTask.FORECASTING.value,
        frequency_mapping: Dict[str,int] = DEFAULT_FREQUENCY_MAPPING,
        **kwargs,
    ):
        # note base class __init__ methods sets all arguments as attributes

        if not isinstance(id_columns, list):
            raise ValueError(
                f"Invalid argument provided for `id_columns`: {id_columns}"
            )

        self.timestamp_column = timestamp_column
        self.input_columns = list(input_columns)
        self.output_columns = list(output_columns)
        self.id_columns = list(id_columns)
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.scaling = scaling
        self.encode_categorical = encode_categorical
        self.categorical_columns = list(categorical_columns)
        self.time_series_task = time_series_task
        self.scale_outputs = scale_outputs

        self.scaler_dict = dict()
        self.categorical_encoder = None
        self.frequency_mapping = frequency_mapping

        kwargs["processor_class"] = self.__class__.__name__

        super().__init__(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this feature extractor instance.
        """
        output = super().to_dict()

        for k, v in output["scaler_dict"].items():
            output["scaler_dict"][k] = v.to_dict()


        if self.categorical_encoder:
            output["categorical_encoder"] = output["categorical_encoder"].to_dict()

        return output
    
    def to_json_string(self) -> str:
        """
        Serializes this instance to a JSON string.

        Returns:
            `str`: String containing all the attributes that make up this feature_extractor instance in JSON format.
        """
        dictionary = self.to_dict()

        def recursive_check_ndarray(dictionary):
            for key, value in dictionary.items():
                if isinstance(value, np.ndarray):
                    dictionary[key] = value.tolist()
                elif isinstance(value, np.int64):
                    dictionary[key] = int(value)
                elif isinstance(value, dict):
                    dictionary[key] = recursive_check_ndarray(value)
            return dictionary
                    
        dictionary = recursive_check_ndarray(dictionary)

        # make sure private name "_processor_class" is correctly
        # saved as "processor_class"
        _processor_class = dictionary.pop("_processor_class", None)
        if _processor_class is not None:
            dictionary["processor_class"] = _processor_class

        return json.dumps(dictionary, indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_dict(
        cls, feature_extractor_dict: Dict[str, Any], **kwargs
    ) -> "PreTrainedFeatureExtractor":
        """
        Instantiates a type of [`~feature_extraction_utils.FeatureExtractionMixin`] from a Python dictionary of
        parameters.

        Args:
            feature_extractor_dict (`Dict[str, Any]`):
                Dictionary that will be used to instantiate the feature extractor object. Such a dictionary can be
                retrieved from a pretrained checkpoint by leveraging the
                [`~feature_extraction_utils.FeatureExtractionMixin.to_dict`] method.
            kwargs (`Dict[str, Any]`):
                Additional parameters from which to initialize the feature extractor object.

        Returns:
            [`~feature_extraction_utils.FeatureExtractionMixin`]: The feature extractor object instantiated from those
            parameters.
        """

        scaler_params = feature_extractor_dict.get("scaler_dict", None)

        if scaler_params is not None:
            for k, v in scaler_params.items():
                scaler_params[k] = StandardScaler.from_dict(v)

        return super().from_dict(feature_extractor_dict, **kwargs)

    # def _prepare_single_time_series(self, name, d):
    #     """
    #     Segment and prepare the time series based on the configuration arguments.

    #     name: name for the time series, for example as a result of a grouping operation
    #     d: the data for a single time series
    #     """
    #     for s_begin in range(d.shape[0] - self.context_length + 1):
    #         s_end = s_begin + self.context_length
    #         seq_x = d[self.input_columns].iloc[s_begin:s_end].values

    #         if self.time_series_task == TimeSeriesTask.FORECASTING:
    #             seq_y = (
    #                 d[self.output_columns]
    #                 .iloc[s_end : s_end + self.prediction_length]
    #                 .values
    #             )
    #         else:
    #             seq_y = None
    #         # to do: add handling of other types

    #         if self.timestamp_column:
    #             ts = d[self.timestamp_column].iloc[s_end - 1]
    #         else:
    #             ts = None

    #         if self.id_columns:
    #             ids = d[self.id_columns].iloc[s_end - 1].values
    #         else:
    #             ids = None

    #         yield {
    #             "timestamp_column": ts,
    #             "id_columns": ids,
    #             "past_values": seq_x,
    #             "future_values": seq_y,
    #         }

    def _standardize_dataframe(
        self,
        dataset: Union[Dataset, pd.DataFrame],
    ):
        if isinstance(dataset, Dataset):
            df = dataset.to_pandas()
        else:
            df = dataset

        if not self.id_columns:
            df[INTERNAL_ID_COLUMN] = INTERNAL_ID_VALUE

        return df

    def _get_groups(
        self,
        dataset: pd.DataFrame,
    ):
        if self.id_columns:
            group_by_columns = (
                self.id_columns if len(self.id_columns) > 1 else self.id_columns[0]
            )
        else:
            group_by_columns = INTERNAL_ID_COLUMN

        grps = dataset.groupby(by=group_by_columns)
        for name, g in grps:
            g = g.sort_values(by=self.timestamp_column)
            yield name, g

    def _get_columns_to_scale(
        self,
    ) -> List[str]:
        """Returns the columns to perform scaling on, based on the options specified during
        preprocessor init.

        Returns:
            List[str]: List of column names
        """
        cols_to_scale = copy.copy(self.input_columns)
        if self.scale_outputs:
            cols_to_scale.extend(
                [
                    c
                    for c in self.output_columns
                    if c not in self.input_columns and c not in self.categorical_columns
                ]
            )
        return cols_to_scale

    def _get_columns_to_encode(
        self,
    ) -> List[str]:
        """Returns the columns to perform encoding on, based on the options specified during
        preprocessor init.

        Returns:
            List[str]: List of column names
        """
        cols_to_encode = self.categorical_columns
        return cols_to_encode

    def _train_scaler(self, df: pd.DataFrame):
        cols_to_scale = self._get_columns_to_scale()

        for name, g in self._get_groups(df):
            if self.scaling:
                # train and transform
                self.scaler_dict[name] = StandardScaler()
                self.scaler_dict[name].fit(g[cols_to_scale])

    def _train_categorical_encoder(self, df: pd.DataFrame):
        cols_to_encode = self._get_columns_to_encode()

        self.categorical_encoder = OrdinalEncoder()
        self.categorical_encoder.fit(df[cols_to_encode])

    def get_frequency_token(self, token_name: str):

        token = self.frequency_mapping.get(token_name, None)

        if token is None:
            warn(f"Frequency token {token_name} was not found in the frequncy token mapping.")
            token = self.frequency_mapping["oov"]

        return token


    def train(
        self,
        dataset: Union[Dataset, pd.DataFrame],
    ):
        """Train data transformation operations

        Currently iterates over groups defined by id_columns to train the scaler, if enabled.
        This could be generalized to arbitrary sequence of operations to apply to each group.

        The fitted scalers and their parameters are saved in scaler_dict

        Returns: self

        """

        df = self._standardize_dataframe(dataset)

        if self.scaling:
            self._train_scaler(df)

        if self.encode_categorical:
            self._train_categorical_encoder(df)

        return self

    def preprocess(
        self,
        dataset: Union[Dataset, pd.DataFrame],
    ) -> Dataset:
        """Main function used to return preprocessed data"""
        # for now we assume data is already fully loaded
        # eventually we need a strategy for dealing with:
        # 1) lists of references to datasets
        # 2) incremental / batch based processing of datasets to minimize memory impact

        df = self._standardize_dataframe(dataset)

        if self.scaling:
            cols_to_scale = self._get_columns_to_scale()

            if self.scaling and len(self.scaler_dict) == 0:
                # trying to get output, but we never trained the scaler
                raise RuntimeError(
                    "Attempt to get scaled output, but scaler has not yet been trained. Please run the `train` method first."
                )

            # note, we might want an option to return a copy of the data rather than modifying in place

            def scale_func(grp, id_columns):
                if isinstance(id_columns, list):
                    name = tuple(grp.iloc[0][id_columns].tolist())
                else:
                    name = grp.iloc[0][id_columns]
                grp[cols_to_scale] = self.scaler_dict[name].transform(
                    grp[cols_to_scale]
                )
                return grp

            if self.id_columns:
                id_columns = (
                    self.id_columns if len(self.id_columns) > 1 else self.id_columns[0]
                )
            else:
                id_columns = INTERNAL_ID_COLUMN

            df_out = df.groupby(id_columns, group_keys=False).apply(
                scale_func,
                id_columns=id_columns,
            )
            df = df_out

        if self.encode_categorical:
            if not self.categorical_encoder:
                raise RuntimeError(
                    "Attempt to encode categorical columns, but the encoder has not been trained yet."
                )

            cols_to_encode = self._get_columns_to_encode()

            df[cols_to_encode] = self.categorical_encoder.transform(df[cols_to_encode])

        return df
