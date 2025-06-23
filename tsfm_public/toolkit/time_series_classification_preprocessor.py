# Copyright contributors to the TSFM project
#
"""Classification-specific preprocessor for time series data preparation"""

import copy
import itertools
from collections import defaultdict
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from datasets import Dataset

from .time_series_preprocessor import (
    INTERNAL_ID_COLUMN,
    LabelEncoder,
    ScalerType,
    TimeSeriesProcessorBase,
    TimeSeriesTask,
)
from .util import check_nested_lengths, is_nested_dataframe, join_list_without_repeat


NESTED_ID_COLUMN = "__nested_series_id"


class TimeSeriesClassificationPreprocessor(TimeSeriesProcessorBase):
    """A preprocessor for supporting time series modeling tasks"""

    PROCESSOR_NAME = "preprocessor_config.json"

    def __init__(
        self,
        id_columns: List[str] = [],
        timestamp_column: Optional[str] = None,
        input_columns: List[str] = [],
        label_column: str = None,
        static_categorical_columns: List[str] = [],
        context_length: int = 64,
        scaling: bool = False,
        scaler_type: ScalerType = ScalerType.STANDARD.value,
        scaling_id_columns: Optional[List[str]] = None,
        encode_categorical: bool = True,
        time_series_task: str = TimeSeriesTask.CLASSIFICATION.value,
        **kwargs,
    ):
        """Multi-time series aware data preprocessor. Provides functions for scaling data and facilitates downstream
        operations on time series data, including model training and inference.

        Args:
            id_columns (List[str]): List of column names which identify different time series in a multi-time series input. Defaults to [].
            timestamp_column (Optional[str], optional): The name of the column containing the timestamp of the time series. Defaults to None.
            input_columns (List[str], optional): List of column names which identify the channels in the input, these are the
                columns that will be used to determine the classification. Defaults to [].
            static_categorical_columns (List[str], optional): List of column names which identify categorical-valued channels in the input
                which are fixed over time. Defaults to [].
            context_length (int, optional): The length of the input context window. Defaults to 64.
            prediction_length (Optional[int], optional): The length of the prediction window. Defaults to None.
            scaling (bool, optional): If True, data is scaled. Defaults to False.
            scaler_type (ScalerType, optional): The type of scaling to perform. See ScalerType for available scalers. Defaults to ScalerType.STANDARD.value.
            scaling_id_columns (Optional[List[str]], optional): In some cases we need to separate data by a different set of id_columns
                when determining scaling factors. For the purposes of determining scaling, data will be grouped by the provided columns.
                If None, the `id_columns` will be used. If and empty list ([]), the dataset will be treated as a single group for scaling.
                Defaults to None. This should be a subset of the id_columns.
            encode_categorical (bool, optional): If True any categorical columns will be encoded using ordinal encoding. Defaults to True.
            time_series_task (str, optional): Reserved for future use. Defaults to TimeSeriesTask.FORECASTING.value.

        Raises:
            ValueError: Raised if `id_columns` is not a list.
            ValueError: Raised if `timestamp_column` is not a scalar.
        """
        # note base class __init__ method sets all arguments as attributes

        if not isinstance(id_columns, list):
            raise ValueError(f"Invalid argument provided for `id_columns`: {id_columns}")

        if isinstance(timestamp_column, list):
            raise ValueError(f"`timestamp_column` should not be a list, received: {timestamp_column}")

        if label_column is None:
            raise ValueError("`label_column` must be specified")

        if isinstance(label_column, list):
            raise ValueError(f"`label_column` should not be a list, received: {label_column}")

        self.id_columns = id_columns
        self.timestamp_column = timestamp_column
        self.label_column = label_column
        self.input_columns = list(input_columns)

        self.static_categorical_columns = list(static_categorical_columns)

        self.context_length = context_length
        self.scaling = scaling
        self.encode_categorical = encode_categorical
        self.time_series_task = time_series_task

        self.scaler_type = scaler_type

        # check subset
        if scaling_id_columns is not None:
            if not set(scaling_id_columns).issubset(self.id_columns):
                raise ValueError("`scaling_id_columns` must be a subset of `id_columns`")
            self.scaling_id_columns = scaling_id_columns
        else:
            self.scaling_id_columns = copy.copy(id_columns)

        self.scaler_dict = {}
        self.categorical_encoder = None

        kwargs["processor_class"] = self.__class__.__name__

        super().__init__(**kwargs)

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

    def _validate_columns(self):
        """Check column specification parameters

        Raises:
            ValueError: Raised when a given column appears in multiple column specifiers.
        """

        counter = defaultdict(int)

        for c in self.input_columns + self.static_categorical_columns:
            counter[c] += 1

        if max(counter.values()) > 1:
            raise ValueError(
                "A column name should appear only once in `input_columns` and `static_categorical_columns`."
            )

    def _check_dataset(self, dataset: Union[Dataset, pd.DataFrame], check_nested: bool = True):
        super()._check_dataset(dataset)

        if check_nested:
            check_nested_lengths(dataset, self.input_columns)

    def _get_real_valued_dynamic_channels(
        self,
    ) -> List[str]:
        """Helper function to return list of the real-valued dynamic channels (columns)"""
        real_valued_dynamic_columns = join_list_without_repeat(
            self.input_columns,
        )
        return real_valued_dynamic_columns

    @property
    def num_input_channels(
        self,
    ) -> int:
        """Return the number of input channels

        Input channels are defined as those channels in:
            input_columns

        Future support for time-varying categorical may be addded later.
        """
        return len(self._get_real_valued_dynamic_channels())

    @property
    def exogenous_channel_indices(self) -> List[int]:
        """Return the indices of the exogenous columns

        Classification does not yet support additional exogenous columns.

        Probably remove this method.
        """
        return []

    @property
    def categorical_vocab_size_list(self) -> List[int]:
        """Return the static_categorical_column vocabulary sizes."""
        if not self.static_categorical_columns or not self.encode_categorical:
            return None

        if not self.categorical_encoder:
            raise RuntimeError(
                "Vocabulary sizes are only available after training the preprocessor. Please run the `train` method first."
            )

        sizes = []
        for feat, cats in zip(self.categorical_encoder.feature_names_in_, self.categorical_encoder.categories_):
            if feat in self.static_categorical_columns:
                sizes.append(len(cats))

        return sizes

    def _train_label_encoder(self, df: pd.DataFrame):
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(df[self.label_column])

    def _train_scaler(self, df: pd.DataFrame):
        scaler_class = self._get_scaler_class(self.scaler_type)
        columns_to_scale = self.input_columns

        if self.scaling:
            for name, g in self._get_groups(df):
                self.scaler_dict[name] = scaler_class()
                if self._is_nested:
                    # one scaler for this group, but requires wrangling for training
                    unnested = unnest_transform(df, columns=columns_to_scale)
                    self.scaler_dict[name].fit(unnested[columns_to_scale])
                else:
                    self.scaler_dict[name] = scaler_class()
                    # one scaler per group
                    self.scaler_dict[name].fit(g[columns_to_scale])

    def train(
        self,
        dataset: Union[Dataset, pd.DataFrame],
    ) -> "TimeSeriesClassificationPreprocessor":
        """Train data transformation operations

        Currently iterates over groups defined by id_columns to train the scaler, if enabled.
        This could be generalized to arbitrary sequence of operations to apply to each group.

        The fitted scalers and their parameters are saved in scaler_dict

        Returns: self

        """

        self._check_dataset(dataset)
        df = self._standardize_dataframe(dataset)
        self._validate_columns()

        self._is_nested = is_nested_dataframe(df, column=self.input_columns[0])

        # if self.freq is None:
        #     self._estimate_frequency(df)

        self._train_label_encoder(df)

        if self.encode_categorical:
            self._train_categorical_encoder(df)
            df = self._process_encoding(df.copy())

        if self.scaling:
            self._train_scaler(df)

        self._clean_up_dataframe(df)
        return self

    def _process_label_encoding(self, df: pd.DataFrame):
        cols_to_encode = self.label_column
        if not self.label_encoder:
            raise RuntimeError("Attempt to encode label column, but the encoder has not been trained yet.")
        df[cols_to_encode] = self.label_encoder.transform(df[cols_to_encode])
        return df

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
        df = self._process_label_encoding(df)
        df = self._process_encoding(df)

        if self.scaling:
            cols_to_scale = self.input_columns  # self._get_other_columns_to_scale()

            if self.scaling and len(self.scaler_dict) == 0:
                # trying to get output, but we never trained the scaler
                raise RuntimeError(
                    "Attempt to get scaled output, but scaler has not yet been trained. Please run the `train` method first."
                )

            # note, we might want an option to return a copy of the data rather than modifying in place
            def scale_func(grp, id_columns, nested=False):
                if isinstance(id_columns, list):
                    name = tuple(grp.iloc[0][id_columns].tolist())
                else:
                    name = grp.iloc[0][id_columns]

                if nested:
                    unnested = unnest_transform(df, columns=cols_to_scale)
                    unnested[cols_to_scale] = self.scaler_dict[name].transform(unnested[cols_to_scale])
                    grp[cols_to_scale] = nest_transform(unnested, columns=cols_to_scale)
                else:
                    grp[cols_to_scale] = self.scaler_dict[name].transform(grp[cols_to_scale])
                return grp

            if self.scaling_id_columns is not None and len(self.scaling_id_columns) > 0:
                id_columns = (
                    self.scaling_id_columns if len(self.scaling_id_columns) > 1 else self.scaling_id_columns[0]
                )
            else:
                id_columns = INTERNAL_ID_COLUMN

            df_out = df.groupby(id_columns, group_keys=False)[df.columns].apply(
                scale_func, id_columns=id_columns, nested=self._is_nested
            )
            df = df_out

        self._clean_up_dataframe(df)
        return df

    def inverse_transform_labels(self, dataset: pd.DataFrame, suffix: Optional[str] = None) -> pd.DataFrame:
        """Inverse transform the labels back to their original values.

        Args:
            dataset (pd.DataFrame): The input dataframe.

        Returns:
            pd.DataFrame: Dataframe with original values in the label_column
        """
        self._check_dataset(dataset, check_nested=False)
        df = self._standardize_dataframe(dataset)

        col_to_transform = self.label_column
        if suffix is not None:
            col_to_transform = f"{self.label_column}{suffix}"

        df[col_to_transform] = self.label_encoder.inverse_transform(df[col_to_transform])
        self._clean_up_dataframe(df)
        return df


def unnest_transform(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Unnest a dataframe that contains nested series.

    Args:
        df (pd.DataFrame): Original dataframe, should contain entries which are pd.Series.
        columns (List[str]): Columns which shoudl be unnested series.

    Raises:
        ValueError: Raised when the dataframe does not contain and id column referenced by NESTED_ID_COLUMN

    Returns:
        pd.DataFrame: Resulting dataframe, but with nested series entries.
    """

    # create a row_id column
    order_preserved_columns = [c for c in df.columns if c in columns]
    series_lengths = df[columns[0]].apply(len).to_list()
    unnested = [[i] * series_lengths[i] for i in df.index]  # range(len(df))]
    # flatten
    unnested = [np.asarray(list(itertools.chain.from_iterable(unnested)))]

    for c in order_preserved_columns:
        unnested.append(np.concatenate([d.values for d in df[c].to_list()]))

    unnested = pd.DataFrame(
        dict(
            zip(
                [
                    NESTED_ID_COLUMN,
                ]
                + order_preserved_columns,
                unnested,
            )
        ),
    )
    return unnested


def nest_transform(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Nest a dataframe by first splitting the data by group and creating series for each group.
    The new dataframe has these series as row entries. Original order of the columns in the dataframe
    are preserved.

    Args:
        df (pd.DataFrame): Original dataframe, must contain an id column resulting from the unnest_transformation.
        columns (List[str]): Columns for which to create nested series.

    Raises:
        ValueError: Raised when the dataframe does not contain and id column referenced by NESTED_ID_COLUMN

    Returns:
        pd.DataFrame: Resulting dataframe, but with nested series entries.
    """

    order_preserved_columns = [c for c in df.columns if c in columns]
    if NESTED_ID_COLUMN not in df.columns:
        raise ValueError(
            f"nest_transformation requires that the dataset have an existing nested id column named {NESTED_ID_COLUMN}"
        )
    groups = df.groupby(NESTED_ID_COLUMN)
    rows = []
    index = []
    for name, g in groups:
        rows.append([pd.Series(g[c].values) for c in order_preserved_columns])
        index.append(name)

    nested = pd.DataFrame(rows, columns=order_preserved_columns, index=index)
    return nested
