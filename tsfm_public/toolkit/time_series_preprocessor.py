# Copyright contributors to the TSFM project
#
"""Preprocessor for time series data preparation"""

import datetime
import enum
import json
from collections import defaultdict
from datetime import timedelta
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
from warnings import warn

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.preprocessing import MinMaxScaler as MinMaxScaler_
from sklearn.preprocessing import OrdinalEncoder as OrdinalEncoder_
from sklearn.preprocessing import StandardScaler as StandardScaler_
from transformers.feature_extraction_utils import (
    FeatureExtractionMixin,
    PreTrainedFeatureExtractor,
)

from .dataset import ForecastDFDataset
from .util import (
    get_split_params,
    join_list_without_repeat,
    select_by_relative_fraction,
)


INTERNAL_ID_COLUMN = "__id"
INTERNAL_ID_VALUE = "0"


DEFAULT_FREQUENCY_MAPPING = {
    "oov": 0,
    "half_hourly": 1,
    "hourly": 2,
    "10_minutes": 3,
    "15_minutes": 4,
}


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
    def from_dict(cls, feature_extractor_dict: Dict[str, Any], **kwargs) -> "SKLearnFeatureExtractionBase":
        """ """

        t = cls()
        t.__setstate__(feature_extractor_dict)

        return t


class StandardScaler(StandardScaler_, SKLearnFeatureExtractionBase):
    """Simple wrapper class to adapt standard scaler to work with the HF
    serialization approach.
    """


class MinMaxScaler(MinMaxScaler_, SKLearnFeatureExtractionBase):
    """Simple wrapper class to adapt min/max scaler to work with the HF
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


class ScalerType(enum.Enum):
    """`Enum` for the different kinds of scalers."""

    MINMAX = "minmax"
    STANDARD = "standard"


class TimeSeriesPreprocessor(FeatureExtractionMixin):
    """A preprocessor for supporting time series modeling tasks"""

    def __init__(
        self,
        id_columns: List[str] = [],
        timestamp_column: Optional[str] = None,
        target_columns: List[str] = [],
        observable_columns: List[str] = [],
        control_columns: List[str] = [],
        conditional_columns: List[str] = [],
        static_categorical_columns: List[str] = [],
        context_length: int = 64,
        prediction_length: Optional[int] = None,
        scaling: bool = False,
        # scale_outputs: bool = False,
        scaler_type: ScalerType = ScalerType.STANDARD.value,
        encode_categorical: bool = True,
        time_series_task: str = TimeSeriesTask.FORECASTING.value,
        frequency_mapping: Dict[str, int] = DEFAULT_FREQUENCY_MAPPING,
        freq: Optional[Union[int, float, timedelta, pd.Timedelta, str]] = None,
        **kwargs,
    ):
        # note base class __init__ methods sets all arguments as attributes

        if not isinstance(id_columns, list):
            raise ValueError(f"Invalid argument provided for `id_columns`: {id_columns}")

        self.id_columns = id_columns
        self.timestamp_column = timestamp_column
        self.target_columns = list(target_columns)
        self.observable_columns = list(observable_columns)
        self.control_columns = list(control_columns)
        self.conditional_columns = list(conditional_columns)
        self.static_categorical_columns = list(static_categorical_columns)

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.scaling = scaling
        self.encode_categorical = encode_categorical
        self.time_series_task = time_series_task
        # self.scale_outputs = scale_outputs
        self.scaler_type = scaler_type

        # we maintain two scalers per time series to facilitate inverse scaling of the targets
        self.scaler_dict = {}
        self.target_scaler_dict = {}
        self.categorical_encoder = None
        self.frequency_mapping = frequency_mapping
        self.freq = freq

        kwargs["processor_class"] = self.__class__.__name__

        self._validate_columns()

        super().__init__(**kwargs)

    def _validate_columns(self):
        """Check column specification parameters

        Raises:
            ValueError: Raised when a given column appears in multiple column specifiers.
        """

        counter = defaultdict(int)

        for c in (
            self.target_columns
            + self.observable_columns
            + self.control_columns
            + self.conditional_columns
            + self.static_categorical_columns
        ):
            counter[c] += 1

        if max(counter.values()) > 1:
            raise ValueError(
                "A column name should appear only once in `target_columns`, `observable_colums`, `control_columnts`, `conditional_columns`, `categorical_columns`, and `static_columns`."
            )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this feature extractor instance.
        """
        output = super().to_dict()

        for k, v in output["scaler_dict"].items():
            output["scaler_dict"][k] = v.to_dict()

        for k, v in output["target_scaler_dict"].items():
            output["target_scaler_dict"][k] = v.to_dict()

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
                if key == "dtype":
                    # to do: ensure deserializable
                    dictionary[key] = value.__name__
                elif isinstance(value, np.ndarray):
                    dictionary[key] = value.tolist()
                elif isinstance(value, np.int64):
                    dictionary[key] = int(value)
                elif isinstance(value, list):
                    dictionary[key] = [vv.tolist() if isinstance(vv, np.ndarray) else vv for vv in value]
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
    def from_dict(cls, feature_extractor_dict: Dict[str, Any], **kwargs) -> "PreTrainedFeatureExtractor":
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

        scaler_type = feature_extractor_dict.get("scaler_type", None)

        scaler_class = cls._get_scaler_class(scaler_type)

        scaler_params = feature_extractor_dict.get("scaler_dict", None)
        if scaler_params is not None:
            for k, v in scaler_params.items():
                scaler_params[k] = scaler_class.from_dict(v)

        target_scaler_params = feature_extractor_dict.get("target_scaler_dict", None)
        if target_scaler_params is not None:
            for k, v in target_scaler_params.items():
                target_scaler_params[k] = scaler_class.from_dict(v)

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

    @classmethod
    def _get_scaler_class(cls, scaler_type):
        if scaler_type == ScalerType.MINMAX.value:
            return MinMaxScaler

        if scaler_type == ScalerType.STANDARD.value:
            return StandardScaler

        raise ValueError(f"Unknown scaler type {scaler_type} specified.")

    def _standardize_dataframe(
        self,
        dataset: Union[Dataset, pd.DataFrame],
    ) -> pd.DataFrame:
        """For given supported inputs, appropriately converts to a pandas dataframe. Adds an ID column
        if needed.

        Args:
            dataset (Union[Dataset, pd.DataFrame]): Input dataset

        Returns:
            pd.DataFrame: Converted dataframe with ID column.
        """
        if isinstance(dataset, Dataset):
            df = dataset.to_pandas()
        else:
            df = dataset.copy()

        if not self.id_columns:
            df[INTERNAL_ID_COLUMN] = INTERNAL_ID_VALUE

        return df

    def _get_groups(
        self,
        dataset: pd.DataFrame,
    ) -> Generator[Tuple[Any, pd.DataFrame], None, None]:
        """Get groups of the time series dataset (multi-time series) based on the ID columns.

        Args:
            dataset (pd.DataFrame): Input dataset

        Yields:
            Generator[Any, pd.DataFrame]: Group name and resulting pandas dataframe for the group.
        """
        if self.id_columns:
            group_by_columns = self.id_columns if len(self.id_columns) > 1 else self.id_columns[0]
        else:
            group_by_columns = INTERNAL_ID_COLUMN

        grps = dataset.groupby(by=group_by_columns)
        for name, g in grps:
            # g = g.sort_values(by=self.timestamp_column)
            yield name, g

    def _get_other_columns_to_scale(
        self,
    ) -> List[str]:
        """Returns the columns to perform scaling on, based on the options specified during
        preprocessor init.

        Returns:
            List[str]: List of column names
        """

        cols_to_scale = join_list_without_repeat(
            self.observable_columns,
            self.control_columns,
            self.conditional_columns,
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
        cols_to_encode = self.static_categorical_columns
        return cols_to_encode

    def _train_scaler(self, df: pd.DataFrame):
        cols_to_scale = self._get_other_columns_to_scale()
        scaler_class = self._get_scaler_class(self.scaler_type)

        for name, g in self._get_groups(df):
            if self.scaling:
                # train and transform
                if cols_to_scale:
                    self.scaler_dict[name] = scaler_class()
                    self.scaler_dict[name].fit(g[cols_to_scale])

                self.target_scaler_dict[name] = scaler_class()
                self.target_scaler_dict[name].fit(g[self.target_columns])

    def _train_categorical_encoder(self, df: pd.DataFrame):
        cols_to_encode = self._get_columns_to_encode()

        if cols_to_encode:
            self.categorical_encoder = OrdinalEncoder()
            self.categorical_encoder.fit(df[cols_to_encode])

    def get_frequency_token(self, token_name: str):
        token = self.frequency_mapping.get(token_name, None)

        if token is None:
            warn(f"Frequency token {token_name} was not found in the frequncy token mapping.")
            token = self.frequency_mapping["oov"]

        return token

    def _get_real_valued_dynamic_channels(
        self,
    ) -> List[str]:
        """Helper function to return list of the real-valued dynamic channels (columns)"""
        real_valued_dynamic_columns = join_list_without_repeat(
            self.target_columns,
            self.observable_columns,
            self.control_columns,
            self.conditional_columns,
        )
        return real_valued_dynamic_columns

    @property
    def num_input_channels(
        self,
    ) -> int:
        return len(self._get_real_valued_dynamic_channels())

    @property
    def exogenous_channel_indices(self) -> List[int]:
        return [
            i
            for i, c in enumerate(self._get_real_valued_dynamic_channels())
            if c in self.control_columns + self.observable_columns
        ]

    @property
    def prediction_channel_indices(self) -> List[int]:
        return [i for i, c in enumerate(self._get_real_valued_dynamic_channels()) if c in self.target_columns]

    def _check_dataset(self, dataset: Union[Dataset, pd.DataFrame]):
        """Basic checks for input dataset.

        Args:
            dataset (Union[Dataset, pd.DataFrame]): Input time series data.

        Raises:
            ValueError: Raised if the dataset is empty.
        """
        if dataset is None or len(dataset) == 0:
            raise ValueError("Input dataset must not be null or zero length.")

    def _estimate_frequency(self, df: pd.DataFrame):
        if self.timestamp_column:
            if self.id_columns:
                # to do: be more efficient
                grps = df.groupby(self.id_columns)
                _, df_subset = list(grps)[0]
            else:
                df_subset = df

            # to do: make more robust
            self.freq = df_subset[self.timestamp_column].iloc[-1] - df_subset[self.timestamp_column].iloc[-2]
        else:
            # no timestamp, assume sequential count?
            self.freq = 1

    def train(
        self,
        dataset: Union[Dataset, pd.DataFrame],
    ) -> "TimeSeriesPreprocessor":
        """Train data transformation operations

        Currently iterates over groups defined by id_columns to train the scaler, if enabled.
        This could be generalized to arbitrary sequence of operations to apply to each group.

        The fitted scalers and their parameters are saved in scaler_dict

        Returns: self

        """

        self._check_dataset(dataset)

        df = self._standardize_dataframe(dataset)

        if self.freq is None:
            self._estimate_frequency(df)

        if self.scaling:
            self._train_scaler(df)

        if self.encode_categorical:
            self._train_categorical_encoder(df)

        return self

    def inverse_scale_targets(self, dataset: Union[Dataset, pd.DataFrame]) -> Dataset:
        df = self._standardize_dataframe(dataset)

        if not self.scaling or len(self.target_scaler_dict) == 0:
            # trying to inverse scale but this preprocessor is not set up for scaling
            raise RuntimeError(
                "Attempt to perform inverse scaling, but time series preprocess is not configured for scaling or scaler has not yet been trained. Please run the `train` method first."
            )

        cols_to_scale = self.target_columns

        def inverse_scale_func(grp, id_columns):
            if isinstance(id_columns, list):
                name = tuple(grp.iloc[0][id_columns].tolist())
            else:
                name = grp.iloc[0][id_columns]
            grp[cols_to_scale] = self.target_scaler_dict[name].inverse_transform(grp[cols_to_scale])
            return grp

        if self.id_columns:
            id_columns = self.id_columns if len(self.id_columns) > 1 else self.id_columns[0]
        else:
            id_columns = INTERNAL_ID_COLUMN

        return df.groupby(id_columns, group_keys=False).apply(
            inverse_scale_func,
            id_columns=id_columns,
        )

    def preprocess(
        self,
        dataset: Union[Dataset, pd.DataFrame],
    ) -> Dataset:
        """Main function used to return preprocessed data"""
        # for now we assume data is already fully loaded
        # eventually we need a strategy for dealing with:
        # 1) lists of references to datasets
        # 2) incremental / batch based processing of datasets to minimize memory impact

        self._check_dataset(dataset)

        df = self._standardize_dataframe(dataset)

        if self.scaling:
            other_cols_to_scale = self._get_other_columns_to_scale()

            if self.scaling and len(self.target_scaler_dict) == 0:
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
                grp[self.target_columns] = self.target_scaler_dict[name].transform(grp[self.target_columns])
                if other_cols_to_scale:
                    grp[other_cols_to_scale] = self.scaler_dict[name].transform(grp[other_cols_to_scale])

                return grp

            if self.id_columns:
                id_columns = self.id_columns if len(self.id_columns) > 1 else self.id_columns[0]
            else:
                id_columns = INTERNAL_ID_COLUMN

            df_out = df.groupby(id_columns, group_keys=False).apply(
                scale_func,
                id_columns=id_columns,
            )
            df = df_out

        cols_to_encode = self._get_columns_to_encode()
        if self.encode_categorical and cols_to_encode:
            if not self.categorical_encoder:
                raise RuntimeError("Attempt to encode categorical columns, but the encoder has not been trained yet.")
            df[cols_to_encode] = self.categorical_encoder.transform(df[cols_to_encode])

        return df

    def get_datasets(
        self,
        dataset: Union[Dataset, pd.DataFrame],
        split_config: Dict[str, Any],
        fewshot_fraction: Optional[float] = None,
    ) -> Tuple[Any]:
        """Creates the preprocessed pytorch datasets needed for training and evaluation
        using the HuggingFace trainer

        Args:
            dataset (Union[Dataset, pd.DataFrame]): Loaded pandas dataframe
                split_config (Dict[str, Any]): Dictionary of dictionaries containing
                split parameters. For example:
                    {
                        train: [0, 50],
                        valid: [50, 70],
                        test:  [70, 100]
                    }
                end value is not inclusive
            fewshot_fraction (float, optional): When non-null, return this percent of the original training
                dataset. This is done to support fewshot fine-tuning. The fraction of data chosen is at the
                end of the training dataset.

        Returns:
            Tuple of pytorch datasets, including: train, validation, test.
        """

        data = self._standardize_dataframe(dataset)

        # get split_params
        # split_params = get_split_params(config, self.context_length, len(data))

        split_params, split_function = get_split_params(split_config)

        # specify columns
        column_specifiers = {
            "id_columns": self.id_columns,
            "timestamp_column": self.timestamp_column,
            "target_columns": self.target_columns,
            "observable_columns": self.observable_columns,
            "control_columns": self.control_columns,
            "conditional_columns": self.conditional_columns,
            "static_categorical_columns": self.static_categorical_columns,
        }

        # split data
        train_data = split_function["train"](data, id_columns=self.id_columns, **split_params["train"])
        valid_data = split_function["valid"](data, id_columns=self.id_columns, **split_params["valid"])
        test_data = split_function["test"](data, id_columns=self.id_columns, **split_params["test"])

        # data preprocessing
        self.train(train_data)

        # handle fewshot operation
        if fewshot_fraction is not None:
            if not ((fewshot_fraction <= 1) and (fewshot_fraction > 0)):
                raise ValueError(f"Fewshot fraction should be between 0 and 1, received {fewshot_fraction}")
            train_data = select_by_relative_fraction(train_data, start_fraction=1 - fewshot_fraction, end_fraction=1)

        params = column_specifiers
        params["context_length"] = self.context_length
        params["prediction_length"] = self.prediction_length

        # get torch datasets
        test_dataset = ForecastDFDataset(
            self.preprocess(test_data),
            **params,
        )
        train_dataset = ForecastDFDataset(self.preprocess(train_data), **params)
        valid_dataset = ForecastDFDataset(
            self.preprocess(valid_data),
            **params,
        )
        return train_dataset, valid_dataset, test_dataset


def create_timestamps(
    last_timestamp: Union[datetime.datetime, pd.Timestamp],
    freq: Optional[Union[int, float, datetime.timedelta, pd.Timedelta, str]] = None,
    time_sequence: Optional[Union[List[int], List[float], List[datetime.datetime], List[pd.Timestamp]]] = None,
    periods: int = 1,
):
    """Simple utility to create a list of timestamps based on start, delta and number of periods"""

    if freq is None and time_sequence is None:
        raise ValueError("Neither `freq` nor `time_sequence` provided, cannot determine frequency.")

    if freq is None:
        # to do: make more robust
        freq = time_sequence[-1] - time_sequence[-2]

    # more complex logic is required to support all edge cases
    if isinstance(freq, (pd.Timedelta, datetime.timedelta, str)):
        return pd.date_range(
            last_timestamp,
            freq=freq,
            periods=periods + 1,
        ).tolist()[1:]
    else:
        # numerical timestamp column
        return [last_timestamp + i * freq for i in range(1, periods + 1)]


def extend_time_series(
    time_series: pd.DataFrame,
    # last_known_timestamp,
    timestamp_column: str,
    grouping_columns: List[str],
    freq: Optional[Union[int, float, datetime.timedelta, pd.Timedelta]] = None,
    periods: int = 1,
    # delta: datetime.timedelta = datetime.timedelta(days=1),
):
    """Extends the provided time series with empty data for the number of periods specified. For each time series, based
    on groups defined by grouping columns, adds emptry records following the last timestamp. The empty records contain
    only timestamps and grouping indicators, remaining fields will be null.

    Args:
        time_series (pd.DataFrame): _description_
        start_timestamp (_type_): _description_
        column_name (str): _description_
        grouping_columns (List[str]): _description_
        periods (int, optional): _description_. Defaults to 1.
        delta (datetime.timedelta, optional): _description_. Defaults to datetime.timedelta(days=1).
    """

    def augment_one_series(group: Union[pd.Series, pd.DataFrame]):
        last_timestamp = group[timestamp_column].iloc[-1]

        new_data = pd.DataFrame(
            {
                timestamp_column: create_timestamps(
                    last_timestamp,
                    freq=freq,
                    time_sequence=group[timestamp_column].values,
                    periods=periods,
                )
            }
        )

        df = pd.concat(
            (group, new_data),
            axis=0,
        )
        return df.reset_index(drop=True)

    if grouping_columns == []:
        new_time_series = augment_one_series(time_series)
    else:
        new_time_series = time_series.groupby(grouping_columns).apply(augment_one_series, include_groups=False)
        idx_names = list(new_time_series.index.names)
        idx_names[-1] = "__delete"
        new_time_series = new_time_series.reset_index(names=idx_names)
        new_time_series.drop(columns=["__delete"], inplace=True)

    return new_time_series
