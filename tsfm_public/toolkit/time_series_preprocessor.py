# Copyright contributors to the TSFM project
#
"""Preprocessor for time series data preparation"""

import copy
import datetime
import enum
import json
from collections import defaultdict
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
from warnings import warn

import numpy as np
import pandas as pd
from datasets import Dataset
from deprecated import deprecated
from pandas.tseries.frequencies import to_offset
from sklearn.preprocessing import MinMaxScaler as MinMaxScaler_
from sklearn.preprocessing import OrdinalEncoder as OrdinalEncoder_
from sklearn.preprocessing import StandardScaler as StandardScaler_
from transformers.feature_extraction_utils import (
    FeatureExtractionMixin,
    PreTrainedFeatureExtractor,
)

from .dataset import ForecastDFDataset
from .util import (
    FractionLocation,
    convert_to_univariate,
    get_split_params,
    join_list_without_repeat,
    select_by_fixed_fraction,
)


INTERNAL_ID_COLUMN = "__id"
INTERNAL_ID_VALUE = "0"

DEFAULT_FREQUENCY_MAPPING = {
    "oov": 0,
    "min": 1,  # minutely
    "2min": 2,
    "5min": 3,
    "10min": 4,
    "15min": 5,
    "30min": 6,
    "h": 7,  # hourly
    "d": 8,  # daily
    "W": 9,  # weekly
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
        scaling_id_columns: Optional[List[str]] = None,
        encode_categorical: bool = True,
        time_series_task: str = TimeSeriesTask.FORECASTING.value,
        frequency_mapping: Dict[str, int] = DEFAULT_FREQUENCY_MAPPING,
        freq: Optional[Union[int, str]] = None,
        **kwargs,
    ):
        """Multi-time series aware data preprocessor. Provides functions for scaling data and facitilitates downstream
        operations on time series data, including model training and inference.

        Args:
            id_columns (List[str]): List of column names which identify different time series in a multi-time series input. Defaults to [].
            timestamp_column (Optional[str], optional): The name of the column containing the timestamp of the time series. Defaults to None.
            target_columns (List[str], optional): List of column names which identify the target channels in the input, these are the
                columns that will be forecasted. Defaults to [].
            observable_columns (List[str], optional): List of column names which identify the observable channels in the input.
                Observable channels are channels which we have knowledge about in the past and future. For example, weather
                conditions such as temperature or precipitation may be known or estimated in the future, but cannot be
                changed. Defaults to [].
            control_columns (List[str], optional): List of column names which identify the control channels in the input. Control
                channels are similar to observable channels, except that future values may be controlled. For example, discount
                percentage of a particular product is known and controllable in the future. Defaults to [].
            conditional_columns (List[str], optional): List of column names which identify the conditional channels in the input.
                Conditional channels are channels which we know in the past, but do not know in the future. Defaults to [].
            static_categorical_columns (List[str], optional): List of column names which identify categorical-valued channels in the input
                which are fixed over time. Defaults to [].
            context_length (int, optional): The length of the input context window. Defaults to 64.
            prediction_length (Optional[int], optional): The length of the prediction window. Defaults to None.
            scaling (bool, optional): If True, data is scaled. Defaults to False.
            scaler_type (ScalerType, optional): The type of scaling to perform. See ScalerType for available scalers. Defaults to ScalerType.STANDARD.value.
            scaling_id_columns (Optional[List[str]], optional): In some cases we need to separate data by a different set of id_columns
                when determining scaling factors. For the purposes of determining scaling, data will be grouped by the provided columns.
                If None, the `id_columns` will be used. Defaults to None.
            encode_categorical (bool, optional): If True any categorical columns will be encoded using ordinal encoding. Defaults to True.
            time_series_task (str, optional): Reserved for future use. Defaults to TimeSeriesTask.FORECASTING.value.
            frequency_mapping (Dict[str, int], optional): _description_. Defaults to DEFAULT_FREQUENCY_MAPPING.
            freq (Optional[Union[int, str]], optional): A frequency indicator for the given `timestamp_column`. See
                https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#period-aliases for a description of the
                allowed values. If not provided, we will attempt to infer it from the data. If not provided, frequency will be
                inferred from `timestamp_column`. Defaults to None.

        Raises:
            ValueError: Raised if `id_columns` is not a list.
            ValueError: Raised if `timestamp_column` is not a scalar.
        """
        # note base class __init__ method sets all arguments as attributes

        if not isinstance(id_columns, list):
            raise ValueError(f"Invalid argument provided for `id_columns`: {id_columns}")

        if isinstance(timestamp_column, list):
            raise ValueError(f"`timestamp_column` should not be a list, received: {timestamp_column}")

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
        self.scaling_id_columns = scaling_id_columns if scaling_id_columns is not None else copy.copy(id_columns)

        # we maintain two scalers per time series to facilitate inverse scaling of the targets
        self.scaler_dict = {}
        self.target_scaler_dict = {}
        self.categorical_encoder = None
        self.frequency_mapping = frequency_mapping
        self.freq = freq

        kwargs["processor_class"] = self.__class__.__name__

        # self._validate_columns()

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

    def _clean_up_dataframe(self, df: pd.DataFrame) -> None:
        """Removes columns added during internal processing of the provided dataframe.

        Currently, the following checks are done:
         - Remove INTERNAL_ID_COLUMN if present

        Args:
            df (pd.DataFrame): Input pandas dataframe

        Returns:
            pd.DataFrame: Cleaned up dataframe
        """

        if not self.id_columns:
            if INTERNAL_ID_COLUMN in df.columns:
                df.drop(columns=INTERNAL_ID_COLUMN, inplace=True)

    def _get_groups(
        self,
        dataset: pd.DataFrame,
    ) -> Generator[Tuple[Any, pd.DataFrame], None, None]:
        """Get groups of the time series dataset (multi-time series) based on the ID columns for scaling.
        Note that this is used for scaling purposes only.

        Args:
            dataset (pd.DataFrame): Input dataset

        Yields:
            Generator[Any, pd.DataFrame]: Group name and resulting pandas dataframe for the group.
        """
        if self.scaling_id_columns:
            group_by_columns = (
                self.scaling_id_columns if len(self.scaling_id_columns) > 1 else self.scaling_id_columns[0]
            )
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
        if token is not None:
            return token

        # try to map as a frequency string
        try:
            token_name_offs = to_offset(token_name).freqstr
            token = self.frequency_mapping.get(token_name_offs, None)
            if token is not None:
                return token
        except ValueError:
            # lastly try to map the timedelta to a frequency string
            token_name_td = pd._libs.tslibs.timedeltas.Timedelta(token_name)
            token_name_offs = to_offset(token_name_td).freqstr
            token = self.frequency_mapping.get(token_name_offs, None)
            if token is not None:
                return token

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

    def _set_targets(self, dataset: pd.DataFrame) -> None:
        if self.target_columns == []:
            skip_columns = copy.copy(self.id_columns) + [INTERNAL_ID_COLUMN]
            if self.timestamp_column:
                skip_columns.append(self.timestamp_column)

            skip_columns.extend(self.observable_columns)
            skip_columns.extend(self.control_columns)
            skip_columns.extend(self.conditional_columns)
            skip_columns.extend(self.static_categorical_columns)

            self.target_columns = [c for c in dataset.columns.to_list() if c not in skip_columns]

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

            if not isinstance(self.freq, (str, int)):
                self.freq = str(self.freq)

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
        self._set_targets(df)
        self._validate_columns()

        if self.freq is None:
            self._estimate_frequency(df)

        if self.scaling:
            self._train_scaler(df)

        if self.encode_categorical:
            self._train_categorical_encoder(df)

        self._clean_up_dataframe(df)
        return self

    def inverse_scale_targets(
        self, dataset: Union[Dataset, pd.DataFrame], suffix: Optional[str] = None
    ) -> Union[Dataset, pd.DataFrame]:
        self._check_dataset(dataset)
        df = self._standardize_dataframe(dataset)

        if not self.scaling:
            return dataset

        if len(self.target_scaler_dict) == 0:
            # trying to inverse scale but this preprocessor is not set up for scaling
            raise RuntimeError(
                "Attempt to perform inverse scaling, but time series preprocessor has not yet been trained. Please run the `train` method first."
            )

        cols_to_scale = self.target_columns
        if suffix is not None:
            cols_to_scale = [f"{c}{suffix}" for c in cols_to_scale]

        col_has_list = [df[c].dtype == np.dtype("O") for c in cols_to_scale]

        def explode_row(df_row, name, columns):
            df = pd.DataFrame(df_row[columns].to_dict())
            inv_scale = self.target_scaler_dict[name].inverse_transform(df)
            df_out = df_row.copy()
            for idx, c in enumerate(columns):
                df_out[c] = inv_scale[:, idx]
            return df_out

        def inverse_scale_func(grp, id_columns):
            if isinstance(id_columns, list):
                name = tuple(grp.iloc[0][id_columns].tolist())
            else:
                name = grp.iloc[0][id_columns]

            if not np.any(col_has_list):
                grp[cols_to_scale] = self.target_scaler_dict[name].inverse_transform(grp[cols_to_scale])
            else:
                grp[cols_to_scale] = grp[cols_to_scale].apply(
                    lambda x: explode_row(x, name, cols_to_scale), axis="columns"
                )
            return grp

        if self.scaling_id_columns:
            id_columns = self.scaling_id_columns if len(self.scaling_id_columns) > 1 else self.scaling_id_columns[0]
        else:
            id_columns = INTERNAL_ID_COLUMN

        df_inv = df.groupby(id_columns, group_keys=False).apply(
            inverse_scale_func,
            id_columns=id_columns,
        )
        self._clean_up_dataframe(df_inv)
        return df_inv

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

            if self.scaling_id_columns:
                id_columns = (
                    self.scaling_id_columns if len(self.scaling_id_columns) > 1 else self.scaling_id_columns[0]
                )
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

        self._clean_up_dataframe(df)
        return df

    @deprecated(version="0.1.1", reason="Please use the standalone function `get_datasets()`.")
    def get_datasets(
        self,
        dataset: Union[Dataset, pd.DataFrame],
        split_config: Dict[str, Union[List[Union[int, float]], float]],
        fewshot_fraction: Optional[float] = None,
        fewshot_location: str = FractionLocation.LAST.value,
        use_frequency_token: bool = False,
    ) -> Tuple[Any]:
        """Creates the preprocessed pytorch datasets needed for training and evaluation
        using the HuggingFace trainer

        Args:
            dataset (Union[Dataset, pd.DataFrame]): Loaded pandas dataframe
                split_config (Dict[str, Union[List[Union[int, float]], float]]): Dictionary of dictionaries containing
                split parameters. Two configurations are possible:
                1. Specify train/valid/test indices or relative fractions
                    {
                        train: [0, 50],
                        valid: [50, 70],
                        test:  [70, 100]
                    }
                end value is not inclusive
                2. Specify train/test fractions:
                    {
                        train: 0.7
                        test: 0.2
                    }
                    A valid split should not be specified directly; the above implies valid = 0.1

            fewshot_fraction (float, optional): When non-null, return this percent of the original training
                dataset. This is done to support fewshot fine-tuning.
            fewshot_location (str): Determines where the fewshot data is chosen. Valid options are "first" and "last"
                as described in the enum FewshotLocation. Default is to choose the fewshot data at the end
                of the training dataset (i.e., "last").

        Returns:
            Tuple of pytorch datasets, including: train, validation, test.
        """

        return get_datasets(
            self,
            dataset,
            split_config=split_config,
            fewshot_fraction=fewshot_fraction,
            fewshot_location=fewshot_location,
            use_frequency_token=use_frequency_token,
        )


def prepare_data_splits(
    data: pd.DataFrame,
    id_columns: List[str] = [],
    context_length: int = 64,
    split_config: Dict[str, Union[List[Union[int, float]], float]] = {"train": 0.7, "test": 0.2},
) -> Tuple[pd.DataFrame]:
    """Splits the input dataframe according to the split_config.

    Args:
        data (pd.DataFrame): Input dataframe.
        id_columns (List[str]): List of column names which identify different time series in a multi-time series input. Defaults to [].
        context_length (int, optional): Specifies the length of the context windows extracted from the historical data for feeding into
                the model. Defaults to 64.
        split_config (Dict[str, Union[List[Union[int, float]], float]]): Dictionary of dictionaries containing
            split parameters.  Defaults to {"train": 0.7, "test": 0.2}. Two configurations are possible:
            1. Specify train/valid/test indices or relative fractions
                {
                    train: [0, 50],
                    valid: [50, 70],
                    test:  [70, 100]
                }
            end value is not inclusive
            2. Specify train/test fractions:
                {
                    train: 0.7
                    test: 0.2
                }
                A valid split should not be specified directly; the above implies valid = 0.1
    Returns:
        Tuple of pandas dataframes, including: train, validation, test.
    """
    # get split_params
    split_params, split_function = get_split_params(split_config, context_length=context_length)

    # split data
    if isinstance(split_function, dict):
        train_data = split_function["train"](data, id_columns=id_columns, **split_params["train"])
        valid_data = split_function["valid"](data, id_columns=id_columns, **split_params["valid"])
        test_data = split_function["test"](data, id_columns=id_columns, **split_params["test"])
    else:
        train_data, valid_data, test_data = split_function(data, id_columns=id_columns, **split_params)

    return train_data, valid_data, test_data


def get_datasets(
    ts_preprocessor: TimeSeriesPreprocessor,
    dataset: Union[Dataset, pd.DataFrame],
    split_config: Dict[str, Union[List[Union[int, float]], float]] = {"train": 0.7, "test": 0.2},
    stride: int = 1,
    fewshot_fraction: Optional[float] = None,
    fewshot_location: str = FractionLocation.LAST.value,
    as_univariate: bool = False,
    use_frequency_token: bool = False,
) -> Tuple[Any]:
    """Creates the preprocessed pytorch datasets needed for training and evaluation
    using the HuggingFace trainer

    Args:
        dataset (Union[Dataset, pd.DataFrame]): Loaded pandas dataframe
        split_config (Dict[str, Union[List[Union[int, float]], float]]): Dictionary of dictionaries containing
            split parameters. Defaults to {"train": 0.7, "test": 0.2}. Two configurations are possible:
            1. Specify train/valid/test indices or relative fractions
                {
                    train: [0, 50],
                    valid: [50, 70],
                    test:  [70, 100]
                }
            end value is not inclusive
            2. Specify train/test fractions:
                {
                    train: 0.7
                    test: 0.2
                }
                A valid split should not be specified directly; the above implies valid = 0.1
        stride (int): Stride used for creating the datasets. It is applied to all of train, validation, and test.
            Defaults to 1.
        fewshot_fraction (float, optional): When non-null, return this percent of the original training
            dataset. This is done to support fewshot fine-tuning.
        fewshot_location (str): Determines where the fewshot data is chosen. Valid options are "first" and "last"
            as described in the enum FewshotLocation. Default is to choose the fewshot data at the end
            of the training dataset (i.e., "last").
        as_univariate (bool, optional): When True the datasets returned will contain only one target column. An
            additional ID is added to distinguish original column name. Only valid if there are no exogenous
            specified. Defaults to False.
        use_frequency_token (bool): If True, datasets are created that include the frequency token. Defaults to False.

    Returns:
        Tuple of pytorch datasets, including: train, validation, test.
    """

    if not ts_preprocessor.context_length:
        raise ValueError("TimeSeriesPreprocessor must be instantiated with non-null context_length")
    if not ts_preprocessor.prediction_length:
        raise ValueError("TimeSeriesPreprocessor must be instantiated with non-null prediction_length")

    data = ts_preprocessor._standardize_dataframe(dataset)

    train_data, valid_data, test_data = prepare_data_splits(
        data,
        id_columns=ts_preprocessor.id_columns,
        context_length=ts_preprocessor.context_length,
        split_config=split_config,
    )

    # data preprocessing
    ts_preprocessor.train(train_data)

    # specify columns
    column_specifiers = {
        "id_columns": ts_preprocessor.id_columns,
        "timestamp_column": ts_preprocessor.timestamp_column,
        "target_columns": ts_preprocessor.target_columns,
        "observable_columns": ts_preprocessor.observable_columns,
        "control_columns": ts_preprocessor.control_columns,
        "conditional_columns": ts_preprocessor.conditional_columns,
        "static_categorical_columns": ts_preprocessor.static_categorical_columns,
    }

    # handle fewshot operation
    if fewshot_fraction is not None:
        if not ((fewshot_fraction <= 1.0) and (fewshot_fraction > 0.0)):
            raise ValueError(f"Fewshot fraction should be between 0 and 1, received {fewshot_fraction}")

        train_data = select_by_fixed_fraction(
            train_data,
            id_columns=ts_preprocessor.id_columns,
            fraction=fewshot_fraction,
            location=fewshot_location,
            minimum_size=ts_preprocessor.context_length,
        )

    params = column_specifiers
    params["context_length"] = ts_preprocessor.context_length
    params["prediction_length"] = ts_preprocessor.prediction_length
    params["stride"] = stride
    if use_frequency_token:
        params["frequency_token"] = ts_preprocessor.get_frequency_token(ts_preprocessor.freq)

    # get torch datasets
    train_valid_test = [train_data, valid_data, test_data]
    train_valid_test_prep = [ts_preprocessor.preprocess(d) for d in train_valid_test]

    if as_univariate and len(ts_preprocessor.target_columns) > 1:
        if (
            ts_preprocessor.observable_columns
            or ts_preprocessor.control_columns
            or ts_preprocessor.conditional_columns
            or ts_preprocessor.static_categorical_columns
        ):
            raise ValueError("`as_univariate` option only allowed when there are no exogenous columns.")

        train_valid_test_prep = [
            convert_to_univariate(
                d,
                timestamp_column=ts_preprocessor.timestamp_column,
                id_columns=ts_preprocessor.id_columns,
                target_columns=ts_preprocessor.target_columns,
            )
            for d in train_valid_test_prep
        ]

        params["target_columns"] = ["value"]
        params["id_columns"] = params["id_columns"] + ["column_id"]

    return tuple([ForecastDFDataset(d, **params) for d in train_valid_test_prep])


def create_timestamps(
    last_timestamp: Union[datetime.datetime, pd.Timestamp],
    freq: Optional[Union[int, float, datetime.timedelta, pd.Timedelta, str]] = None,
    time_sequence: Optional[Union[List[int], List[float], List[datetime.datetime], List[pd.Timestamp]]] = None,
    periods: int = 1,
) -> List[pd.Timestamp]:
    """Simple utility to create a list of timestamps based on start, delta and number of periods

    Args:
        last_timestamp (Union[datetime.datetime, pd.Timestamp]): The last observed timestamp, new timestamps will be created
            after this timestamp.
        freq (Optional[Union[int, float, datetime.timedelta, pd.Timedelta, str]], optional): The frequency at which timestamps
            should be generated. Defaults to None.
        time_sequence (Optional[Union[List[int], List[float], List[datetime.datetime], List[pd.Timestamp]]], optional): A time sequence
            from which the frequency can be inferred. Defaults to None.
        periods (int, optional): The number of timestamps to generate. Defaults to 1.

    Raises:
        ValueError: If the frequency cannot be parsed from freq or inferred from time_sequence

    Returns:
        List[pd.Timestamp]: List of timestamps
    """

    if freq is None and time_sequence is None:
        raise ValueError("Neither `freq` nor `time_sequence` provided, cannot determine frequency.")

    if freq is None:
        # to do: make more robust
        freq = time_sequence[-1] - time_sequence[-2]

    # more complex logic is required to support all edge cases
    if isinstance(freq, (pd.Timedelta, datetime.timedelta, str)):
        try:
            # try date range directly
            return pd.date_range(
                last_timestamp,
                freq=freq,
                periods=periods + 1,
            ).tolist()[1:]
        except ValueError as e:
            # if it fails, we can try to compute a timedelta from the provided string
            if isinstance(freq, str):
                freq = pd._libs.tslibs.timedeltas.Timedelta(freq)
                return pd.date_range(
                    last_timestamp,
                    freq=freq,
                    periods=periods + 1,
                ).tolist()[1:]
            else:
                raise e
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
