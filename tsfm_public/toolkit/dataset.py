# Copyright contributors to the TSFM project
#
"""Tools for building torch datasets"""

import copy
from itertools import starmap
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch

from .util import join_list_without_repeat


class BaseDFDataset(torch.utils.data.Dataset):
    """
    An abtract class representing a :class: `BaseDFDataset`.

    All the datasets that represents data frames should subclass it.
    All subclasses should overwrite :meth: `__get_item__`

    Args:
        data_df (DataFrame, required): input data
        datetime_col (str, optional): datetime column in the data_df. Defaults to None
        x_cols (list, optional): list of columns of X. If x_cols is an empty list, all the columns in the data_df is taken, except the datatime_col. Defaults to an empty list.
        y_cols (list, required): list of columns of y. Defaults to an empty list.
        seq_len (int, required): the sequence length. Defaults to 1
        pred_len (int, required): forecasting horizon. Defaults to 0.
        zero_padding (bool, optional): pad zero if the data_df is shorter than seq_len+pred_len
    """

    def __init__(
        self,
        data_df: pd.DataFrame,
        id_columns: List[str] = [],
        timestamp_column: Optional[str] = None,
        group_id: Optional[Union[List[int], List[str]]] = None,
        x_cols: list = [],
        y_cols: list = [],
        drop_cols: list = [],
        context_length: int = 1,
        prediction_length: int = 0,
        zero_padding: bool = True,
    ):
        super().__init__()
        if not isinstance(x_cols, list):
            x_cols = [x_cols]
        if not isinstance(y_cols, list):
            y_cols = [y_cols]

        if len(x_cols) > 0:
            assert is_cols_in_df(
                data_df, x_cols
            ), f"one or more {x_cols} is not in the list of data_df columns"

        if len(y_cols) > 0:
            assert is_cols_in_df(
                data_df, y_cols
            ), f"one or more {y_cols} is not in the list of data_df columns"

        if timestamp_column:
            assert timestamp_column in list(
                data_df.columns
            ), f"{timestamp_column} is not in the list of data_df columns"
            assert (
                timestamp_column not in x_cols
            ), f"{timestamp_column} should not be in the list of x_cols"

        self.data_df = data_df
        self.datetime_col = timestamp_column
        self.id_columns = id_columns
        self.x_cols = x_cols
        self.y_cols = y_cols
        self.drop_cols = drop_cols
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.zero_padding = zero_padding
        self.timestamps = None
        self.group_id = group_id

        # sort the data by datetime
        if timestamp_column in list(data_df.columns):
            data_df[timestamp_column] = pd.to_datetime(data_df[timestamp_column])
            data_df = data_df.sort_values(timestamp_column, ignore_index=True)

        # pad zero to the data_df if the len is shorter than seq_len+pred_len
        if zero_padding:
            data_df = self.pad_zero(data_df)

        if timestamp_column in list(data_df.columns):
            self.timestamps = data_df[timestamp_column].values

        # get the input data
        if len(x_cols) > 0:
            self.X = data_df[x_cols]
        else:
            drop_cols = self.drop_cols + y_cols
            if timestamp_column:
                drop_cols += [timestamp_column]
            self.X = data_df.drop(drop_cols, axis=1) if len(drop_cols) > 0 else data_df
            self.x_cols = list(self.X.columns)

        # get target data
        if len(y_cols) > 0:
            self.y = data_df[y_cols]
        else:
            self.y = None

        # get number of X variables
        self.n_vars = self.X.shape[1]
        # get number of target
        self.n_targets = len(y_cols) if len(y_cols) > 0 else 0

    def pad_zero(self, data_df):
        # return zero_padding_to_df(data_df, self.seq_len + self.pred_len)
        return ts_padding(
            data_df,
            timestamp_column=self.datetime_col,
            id_columns=self.id_columns,
            context_length=self.context_length + self.prediction_length,
        )

    def __len__(self):
        return len(self.X) - self.context_length - self.prediction_length + 1

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index
        Returns:
            (Any): Sample and meta data, optionally transformed by the respective transforms.
        """
        raise NotImplementedError


class BaseConcatDFDataset(torch.utils.data.ConcatDataset):
    """
    An abtract class representing a :class: `BaseConcatDFDataset`.

    Args:
        data_df (DataFrame, required): input data
        datetime_col (str, optional): datetime column in the data_df. Defaults to None
        x_cols (list, optional): list of columns of X. If x_cols is an empty list, all the columns in the data_df is taken, except the datatime_col. Defaults to an empty list.
        y_cols (list, required): list of columns of y. Defaults to an empty list.
        group_ids (list, optional): list of group_ids to split the data_df to different groups. If group_ids is defined, it will triggle the groupby method in DataFrame. If empty, entire data frame is treated as one group.
        seq_len (int, required): the sequence length. Defaults to 1
        num_workers (int, optional): the number if workers used for creating a list of dataset from group_ids. Defaults to 1.
        pred_len (int, required): forecasting horizon. Defaults to 0.
        cls (class, required): dataset class
    """

    def __init__(
        self,
        data_df: pd.DataFrame,
        id_columns: List[str] = [],
        timestamp_column: Optional[str] = None,
        context_length: int = 1,
        prediction_length: int = 1,
        num_workers: int = 1,
        cls=BaseDFDataset,
        **kwargs,
    ):
        if len(id_columns) > 0:
            assert is_cols_in_df(
                data_df, id_columns
            ), f"{id_columns} is not in the data_df columns"

        self.timestamp_column = timestamp_column
        self.id_columns = id_columns
        # self.x_cols = x_cols
        # self.y_cols = y_cols
        self.context_length = context_length
        self.num_workers = num_workers
        self.cls = cls
        self.prediction_length = prediction_length
        self.extra_kwargs = kwargs

        # create groupby object
        if len(id_columns) == 1:
            self.group_df = data_df.groupby(by=self.id_columns[0])
        elif len(id_columns) > 1:
            self.group_df = data_df.groupby(by=self.id_columns)
        else:
            data_df["group"] = 0  # create a artificial group
            self.group_df = data_df.groupby(by="group")

        # add group_ids to the drop_cols
        self.drop_cols = id_columns if len(id_columns) > 0 else ["group"]

        self.group_names = list(self.group_df.groups.keys())
        datasets = self.concat_dataset()
        super().__init__(datasets)
        self.n_vars = self.datasets[0].n_vars
        self.n_targets = self.datasets[0].n_targets

    def concat_dataset(self):
        """Create a list of Datasets

        Returns:
            List of datasets
        """
        group_df = self.group_df
        # print(f'group_df: {group_df}')
        # pool = mp.Pool(self.num_workers)
        # pool.starmap(
        list_dset = starmap(
            get_group_data,
            [
                (
                    self.cls,
                    group,
                    group_id,
                    self.id_columns,
                    self.timestamp_column,
                    self.context_length,
                    self.prediction_length,
                    self.drop_cols,
                    self.extra_kwargs,
                )
                for group_id, group in group_df
            ],
        )

        # pool.close()
        # del group_df
        return list_dset


def get_group_data(
    cls,
    group,
    group_id,
    id_columns: List[str] = [],
    timestamp_column: Optional[str] = None,
    context_length: int = 1,
    prediction_length: int = 1,
    drop_cols: Optional[List[str]] = None,
    extra_kwargs: Dict[str, Any] = {},
):
    return cls(
        data_df=group,
        group_id=group_id,
        id_columns=id_columns,
        timestamp_column=timestamp_column,
        context_length=context_length,
        prediction_length=prediction_length,
        drop_cols=drop_cols,
        **extra_kwargs,
    )


class PretrainDFDataset(BaseConcatDFDataset):
    """
    A :class: `PretrainDFDataset` is used for pretraining.

    To be updated
    Args:
        data_df (DataFrame, required): input data
        datetime_col (str, optional): datetime column in the data_df. Defaults to None
        x_cols (list, optional): list of columns of X. If x_cols is an empty list, all the columns in the data_df is taken, except the datatime_col. Defaults to an empty list.
        group_ids (list, optional): list of group_ids to split the data_df to different groups. If group_ids is defined, it will triggle the groupby method in DataFrame. If empty, entire data frame is treated as one group.
        seq_len (int, required): the sequence length. Defaults to 1
        num_workers (int, optional): the number if workers used for creating a list of dataset from group_ids. Defaults to 1.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        id_columns: List[str] = [],
        timestamp_column: Optional[str] = None,
        target_columns: List[str] = [],
        context_length: int = 1,
        num_workers: int = 1,
    ):
        super().__init__(
            data_df=data,
            id_columns=id_columns,
            timestamp_column=timestamp_column,
            num_workers=num_workers,
            context_length=context_length,
            prediction_length=0,
            cls=self.BasePretrainDFDataset,
            target_columns=target_columns,
        )
        self.n_inp = 1

    class BasePretrainDFDataset(BaseDFDataset):
        def __init__(
            self,
            data_df: pd.DataFrame,
            group_id: Optional[Union[List[int], List[str]]] = None,
            context_length: int = 1,
            prediction_length: int = 0,
            drop_cols: list = [],
            id_columns: List[str] = [],
            timestamp_column: Optional[str] = None,
            target_columns: List[str] = [],
        ):
            self.target_columns = target_columns

            x_cols = target_columns
            y_cols = []

            super().__init__(
                data_df=data_df,
                id_columns=id_columns,
                timestamp_column=timestamp_column,
                x_cols=x_cols,
                y_cols=y_cols,
                context_length=context_length,
                prediction_length=prediction_length,
                group_id=group_id,
                drop_cols=drop_cols,
            )

        def __getitem__(self, time_id):
            seq_x = self.X[time_id : time_id + self.context_length].values
            ret = {"past_values": np_to_torch(seq_x)}
            if self.datetime_col:
                ret["timestamp"] = self.timestamps[time_id + self.context_length - 1]
            if self.group_id:
                ret["id"] = self.group_id

            return ret


class ForecastDFDataset(BaseConcatDFDataset):
    """
    A :class: `ForecastDFDataset` used for forecasting.

    Args:
        data_df (DataFrame, required): input data
        datetime_col (str, optional): datetime column in the data_df. Defaults to None
        x_cols (list, optional): list of columns of X. If x_cols is an empty list, all the columns in the data_df is taken, except the datatime_col. Defaults to an empty list.
        group_ids (list, optional): list of group_ids to split the data_df to different groups. If group_ids is defined, it will triggle the groupby method in DataFrame. If empty, entire data frame is treated as one group.
        seq_len (int, required): the sequence length. Defaults to 1
        num_workers (int, optional): the number if workers used for creating a list of dataset from group_ids. Defaults to 1.
        pred_len (int, required): forecasting horizon. Defaults to 0.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        id_columns: List[str] = [],
        timestamp_column: Optional[str] = None,
        target_columns: List[str] = [],
        observable_columns: List[str] = [],
        control_columns: List[str] = [],
        conditional_columns: List[str] = [],
        static_categorical_columns: List[str] = [],
        context_length: int = 1,
        prediction_length: int = 1,
        num_workers: int = 1,
        frequency_token: Optional[int] = None,
        autoregressive_modeling: bool = True,
        training: bool = True,
    ):
        # output_columns_tmp = input_columns if output_columns == [] else output_columns

        super().__init__(
            data_df=data,
            id_columns=id_columns,
            timestamp_column=timestamp_column,
            num_workers=num_workers,
            context_length=context_length,
            prediction_length=prediction_length,
            cls=self.BaseForecastDFDataset,
            # extra_args
            target_columns=target_columns,
            observable_columns=observable_columns,
            control_columns=control_columns,
            conditional_columns=conditional_columns,
            static_categorical_columns=static_categorical_columns,
            frequency_token=frequency_token,
            autoregressive_modeling=autoregressive_modeling,
            training=training,
        )
        self.n_inp = 2
        # for forecasting, the number of targets is the same as number of X variables
        self.n_targets = self.n_vars

    class BaseForecastDFDataset(BaseDFDataset):
        """
        X_{t+1,..., t+p} = f(X_{:t})
        """

        def __init__(
            self,
            data_df: pd.DataFrame,
            group_id: Optional[Union[List[int], List[str]]] = None,
            context_length: int = 1,
            prediction_length: int = 1,
            drop_cols: list = [],
            id_columns: List[str] = [],
            timestamp_column: Optional[str] = None,
            target_columns: List[str] = [],
            observable_columns: List[str] = [],
            control_columns: List[str] = [],
            conditional_columns: List[str] = [],
            static_categorical_columns: List[str] = [],
            frequency_token: Optional[int] = None,
            autoregressive_modeling: bool = True,
            training: bool = True,
        ):
            self.frequency_token = frequency_token
            self.target_columns = target_columns
            self.observable_columns = observable_columns
            self.control_columns = control_columns
            self.conditional_columns = conditional_columns
            self.static_categorical_columns = static_categorical_columns
            self.autoregressive_modeling = autoregressive_modeling
            self.training = training

            x_cols = join_list_without_repeat(
                target_columns,
                observable_columns,
                control_columns,
                conditional_columns,
            )
            y_cols = copy.copy(x_cols)

            # check non-autoregressive case
            if len(target_columns) == len(x_cols) and not self.autoregressive_modeling:
                raise ValueError(
                    "Non-autoregressive modeling was chosen, but there are no input columns for prediction."
                )

            # masking for conditional values which are not observed during future period
            self.y_mask_conditional = np.array(
                [(c in conditional_columns) for c in y_cols]
            )

            # create a mask of x which masks targets
            self.x_mask_targets = np.array([(c in target_columns) for c in x_cols])

            super().__init__(
                data_df=data_df,
                id_columns=id_columns,
                timestamp_column=timestamp_column,
                x_cols=x_cols,
                y_cols=y_cols,
                context_length=context_length,
                prediction_length=prediction_length,
                group_id=group_id,
                drop_cols=drop_cols,
            )

        def __getitem__(self, time_id):
            # seq_x: batch_size x seq_len x num_x_cols
            seq_x = self.X[time_id : time_id + self.context_length].values
            if not self.autoregressive_modeling:
                seq_x[:, self.x_mask_targets] = 0

            # seq_y: batch_size x pred_len x num_x_cols
            seq_y = self.y[
                time_id
                + self.context_length : time_id
                + self.context_length
                + self.prediction_length
            ].values

            seq_y[:, self.y_mask_conditional] = 0

            ret = {
                "past_values": np_to_torch(seq_x),
                "future_values": np_to_torch(seq_y),
            }
            if self.datetime_col:
                ret["timestamp"] = self.timestamps[time_id + self.context_length - 1]

            if self.group_id:
                ret["id"] = self.group_id

            if self.frequency_token is not None:
                ret["freq_token"] = torch.tensor(self.frequency_token, dtype=torch.int)

            if self.static_categorical_columns:
                categorical_values = self.data_df[
                    self.static_categorical_columns
                ].values[0, :]
                ret["static_categorical_values"] = np_to_torch(categorical_values)

            return ret

        def __len__(self):
            return len(self.X) - self.context_length - self.prediction_length + 1


class RegressionDFDataset(BaseConcatDFDataset):
    """
    A :class: `RegressionDFDataset` used for regression.

    Args:
        data_df (DataFrame, required): input data
        datetime_col (str, optional): datetime column in the data_df. Defaults to None
        input_columns (list, optional): list of columns of X. If x_cols is an empty list, all the columns in the data_df is taken, except the datatime_col. Defaults to an empty list.
        output_columns (list, required): list of columns of y. Defaults to an empty list.
        id_columns (list, optional): List of columns that specify ids in the dataset. list of group_ids to split the data_df to different groups. If group_ids is defined, it will triggle the groupby method in DataFrame. If empty, entire data frame is treated as one group.
        context_length (int, required): the sequence length. Defaults to 1
        num_workers (int, optional): the number if workers used for creating a list of dataset from group_ids. Defaults to 1.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        id_columns: List[str] = [],
        timestamp_column: Optional[str] = None,
        input_columns: List[str] = [],
        target_columns: List[str] = [],
        static_categorical_columns: List[str] = [],
        context_length: int = 1,
        num_workers: int = 1,
    ):
        # self.y_cols = y_cols

        super().__init__(
            data_df=data,
            id_columns=id_columns,
            timestamp_column=timestamp_column,
            num_workers=num_workers,
            context_length=context_length,
            cls=self.BaseRegressionDFDataset,
            input_columns=input_columns,
            target_columns=target_columns,
            static_categorical_columns=static_categorical_columns,
        )

        self.n_inp = 2

    class BaseRegressionDFDataset(BaseDFDataset):
        """
        y_{t} = f(X_{:t})
        """

        def __init__(
            self,
            data_df: pd.DataFrame,
            group_id: Optional[Union[List[int], List[str]]] = None,
            context_length: int = 1,
            prediction_length: int = 0,
            drop_cols: list = [],
            id_columns: List[str] = [],
            timestamp_column: Optional[str] = None,
            target_columns: List[str] = [],
            input_columns: List[str] = [],
            static_categorical_columns: List[str] = [],
        ):

            self.target_columns = target_columns
            self.input_columns = input_columns
            self.static_categorical_columns = static_categorical_columns

            x_cols = input_columns
            y_cols = target_columns

            super().__init__(
                data_df=data_df,
                id_columns=id_columns,
                timestamp_column=timestamp_column,
                x_cols=x_cols,
                y_cols=y_cols,
                context_length=context_length,
                prediction_length=prediction_length,
                group_id=group_id,
                drop_cols=drop_cols,
            )

        def __getitem__(self, time_id):
            # seq_x: batch_size x seq_len x num_x_cols
            seq_x = self.X[time_id : time_id + self.context_length].values
            seq_y = self.y[
                time_id + self.context_length - 1 : time_id + self.context_length
            ].values.ravel()
            # return _torch(seq_x, seq_y)

            ret = {
                "past_values": np_to_torch(seq_x),
                "target_values": np_to_torch(seq_y),
            }
            if self.datetime_col:
                ret["timestamp"] = self.timestamps[time_id + self.context_length - 1]

            if self.group_id:
                ret["id"] = self.group_id

            if self.static_categorical_columns:
                categorical_values = self.data_df[
                    self.static_categorical_columns
                ].values[0, :]
                ret["static_categorical_values"] = np_to_torch(categorical_values)

            return ret


def np_to_torch(data: np.array, float_type=np.float32):
    if data.dtype == "float":
        return torch.from_numpy(data.astype(float_type))
    elif data.dtype == "int":
        return torch.from_numpy(data)
    return torch.from_numpy(data)


def _torch(*nps):
    return tuple(np_to_torch(x) for x in nps)


def zero_padding_to_df(df: pd.DataFrame, seq_len: int) -> pd.DataFrame:
    """
    check if df has length > seq_len.
    If not, then fill in zero
    Args:
        df (_type_): data frame
        seq_len (int): sequence length
    Returns:
        data frame
    """
    if len(df) >= seq_len:
        return df
    fill_len = seq_len - len(df) + 1
    # add zeros dataframe
    zeros_df = pd.DataFrame(np.zeros([fill_len, df.shape[1]]), columns=df.columns)
    # combine the data
    new_df = pd.concat([zeros_df, df])
    return new_df


def ts_padding(
    df: pd.DataFrame,
    id_columns: Optional[List[str]] = None,
    timestamp_column: Optional[str] = None,
    context_length: int = 1,
) -> pd.DataFrame:
    """
    Pad a dataframe, which is aware of time series conventions.

    Check if df has length >= context_length.
    If not, then fill (prepending) while preserving types and properly handling IDs and dates/timestamps. When
    prepending dates, the sampling interval will be estimated, to create proper preceeding dates.

    The assumption is the provided data contains only one id across the provided ID columns, the value will be
    replicated in the prepended rows.

    Args:
        df (_type_): data frame
        id_columns: List of strings representing columns containing ID information.
        timestamp_column: str for column name containing timestamps.
        context_length (int): required length

    Returns:
        Padded data frame
    """
    l = len(df)
    if l >= context_length:
        return df
    fill_length = context_length - l  # why did we previously have + 1 here?

    # create dataframe
    pad_df = pd.DataFrame(np.zeros([fill_length, df.shape[1]]), columns=df.columns)

    for c in df.columns:
        if (id_columns and c in id_columns) or (c == timestamp_column):
            continue
        pad_df[c] = pad_df[c].astype(df.dtypes[c], copy=False)

    if timestamp_column:
        if (df[timestamp_column].dtype.type == np.datetime64) or (
            df[timestamp_column].dtype == int
        ):
            last_timestamp = df.iloc[0][timestamp_column]
            period = df.iloc[1][timestamp_column] - df.iloc[0][timestamp_column]
            prepended_timestamps = [
                last_timestamp + offset * period for offset in range(-fill_length, 0)
            ]
            pad_df[timestamp_column] = prepended_timestamps
        else:
            pad_df[timestamp_column] = None
        # Ensure same type
        pad_df[timestamp_column] = pad_df[timestamp_column].astype(
            df[timestamp_column].dtype
        )

    if id_columns:
        id_values = df.iloc[0][id_columns].to_list()
        for id_column_name, id_column_value in zip(id_columns, id_values):
            pad_df[id_column_name] = id_column_value

    # combine the data
    new_df = pd.concat([pad_df, df])
    return new_df


def is_cols_in_df(df: pd.DataFrame, cols: List[str]) -> bool:
    """
    Args:
        df:
        cols:

    Returns:
        bool
    """
    for col in cols:
        if col not in list(df.columns):
            return False
    return True


if __name__ == "__main__":
    df = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5, 6, 7, 8],
            "B": [4, 5, 6, 7, 8, 9, 10, 11],
            "C": [7, 8, 9, 10, 11, 12, 13, 14],
            "g1": [0, 1, 1, 1, 0, 0, 0, 0],
        }
    )
    print(df)

    d6 = PretrainDFDataset(data_df=df, x_cols=["A", "B"], group_ids=["g1"], seq_len=2)
    print(f"d6: {d6}")

    d7 = ForecastDFDataset(
        data_df=df, x_cols=["A", "B"], group_ids=["g1"], seq_len=2, pred_len=2
    )
