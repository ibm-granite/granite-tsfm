# Copyright contributors to the TSFM project
#
"""Tools for building torch datasets"""

import copy
import enum
import logging
from itertools import starmap
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from .util import check_nested_lengths, is_nested_dataframe, join_list_without_repeat


LOGGER = logging.getLogger(__file__)


class ImputeMethod(enum.Enum):
    """`Enum` for the different imputation methods ."""

    FORWARD_FILL = "forward_fill"
    LINEAR = "linear"


class BaseDFDataset(torch.utils.data.Dataset):
    """Base dataset for time series models built upon a pandas dataframe

    Args:
        data_df (pd.DataFrame): Underlying pandas dataframe.
        id_columns (List[str], optional): List of columns which contain id information to separate distinct time series. Defaults to [].
        timestamp_column (Optional[str], optional): Name of the timestamp column. Defaults to None.
        group_id (Optional[Union[List[int], List[str]]], optional): _description_. Defaults to None.
        x_cols (list, optional): Columns to treat as inputs. If an empty list ([]) all the columns in the data_df are taken, except the timestamp column. Defaults to [].
        y_cols (list, optional): Columns to treat as outputs. Defaults to [].
        drop_cols (list, optional): List of columns that are dropped to form the X matrix (input). Defaults to [].
        context_length (int, optional): Length of historical data used when creating individual examples in the torch dataset. Defaults to 1.
        prediction_length (int, optional): Length of prediction (future values). Defaults to 0.
        zero_padding (bool, optional): If True, windows of context_length+prediction_length which are too short are padded with zeros. Defaults to True.
        stride (int, optional): Stride at which windows are produced. Defaults to 1.
        fill_value (Union[float, int], optional): Value used to fill any missing values. Defaults to 0.0.
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
        stride: int = 1,
        fill_value: Union[float, int] = 0.0,
        full_series=False,
    ):
        super().__init__()
        if not isinstance(x_cols, list):
            x_cols = [x_cols]
        if not isinstance(y_cols, list):
            y_cols = [y_cols]

        if len(x_cols) > 0:
            there, missing = is_cols_in_df(data_df, x_cols)
            assert there, f"{missing} given in {x_cols} is not a valid column identifier in the data."

        if len(y_cols) > 0:
            there, missing = is_cols_in_df(data_df, y_cols)
            assert there, f"{missing} given in {y_cols} is not a valid column identifier in the data."

        if timestamp_column:
            assert timestamp_column in list(data_df.columns), (
                f"{timestamp_column} is not in the list of data column names provided {data_df.columns}"
            )
            assert timestamp_column not in x_cols, (
                f"{timestamp_column} can not be used as a timestamp column as it also appears in provided collection:{x_cols}."
            )

        self.data_df = data_df
        self.datetime_col = timestamp_column
        self.id_columns = id_columns
        self.x_cols = x_cols
        self.y_cols = y_cols
        self.drop_cols = drop_cols
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.zero_padding = zero_padding
        self.fill_value = fill_value
        self.timestamps = None
        self.group_id = group_id
        self.stride = stride

        # sort the data by datetime
        if timestamp_column in list(data_df.columns):
            # if not isinstance(data_df[timestamp_column].iloc[0], pd.Timestamp):
            #     data_df[timestamp_column] = pd.to_datetime(data_df[timestamp_column])
            data_df = data_df.sort_values(timestamp_column, ignore_index=True)

        # pad zero to the data_df if the len is shorter than seq_len+pred_len
        # currently no padding when in full_series mode
        if not full_series:
            if zero_padding:
                data_df = self.pad_zero(data_df)
            elif len(data_df) < self.context_length + self.prediction_length:
                LOGGER.warning(
                    f"Padding is disabled and input data is shorter than required length. Received {len(data_df)} time point, but require at least {self.context_length + self.prediction_length} time points."
                )

        if timestamp_column in list(data_df.columns):
            self.timestamps = data_df[timestamp_column].to_list()  # .values coerces timestamps
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
        return max((len(self.X) - self.context_length - self.prediction_length) // self.stride + 1, 0)

    def _check_index(self, index: int) -> int:
        if index >= len(self):
            raise IndexError("Index exceeds dataset length")

        if index < 0:
            if -index > len(self):
                raise ValueError("Absolute value of index should not exceed dataset length")
            index = len(self) + index
        return index

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index
        Returns:
            (Any): Sample and meta data, optionally transformed by the respective transforms.
        """
        raise NotImplementedError


class BaseConcatDFDataset(torch.utils.data.ConcatDataset):
    """A dataset consisting of a concatenation of other datasets, based on torch ConcatDataset.

    Args:
        data_df (pd.DataFrame): Underlying pandas dataframe.
        id_columns (List[str], optional): List of columns which contain id information to separate distinct time series. Defaults to [].
        timestamp_column (Optional[str], optional): Name of the timestamp column. Defaults to None.
        context_length (int, optional): Length of historical data used when creating individual examples in the torch dataset. Defaults to 1.
        prediction_length (int, optional): Length of prediction (future values). Defaults to 0.
        num_workers (int, optional): (Currently not used) Number of workers. Defaults to 1.
        fill_value (Union[float, int], optional): Value used to fill any missing values. Defaults to 0.0.
        cls (_type_, optional): The dataset class used to create the underlying datasets. Defaults to BaseDFDataset.
        stride (int, optional): Stride at which windows are produced. Defaults to 1.
    """

    def __init__(
        self,
        data_df: pd.DataFrame,
        id_columns: List[str] = [],
        timestamp_column: Optional[str] = None,
        context_length: int = 1,
        prediction_length: int = 1,
        num_workers: int = 1,
        fill_value: Union[float, int] = 0.0,
        cls=BaseDFDataset,
        stride: int = 1,
        **kwargs,
    ):
        if len(id_columns) > 0:
            there, missing = is_cols_in_df(data_df, id_columns)
            assert there, f"{missing} given in {id_columns} is not a valid column in the data."

        self.timestamp_column = timestamp_column
        self.id_columns = id_columns
        # self.x_cols = x_cols
        # self.y_cols = y_cols
        self.context_length = context_length
        self.num_workers = num_workers
        self.cls = cls
        self.prediction_length = prediction_length
        self.stride = stride
        self.extra_kwargs = kwargs
        self.fill_value = fill_value
        self.cls = cls

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
                    self.stride,
                    self.fill_value,
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
    stride: int = 1,
    fill_value: Union[float, int] = 0.0,
    extra_kwargs: Dict[str, Any] = {},
):
    return cls(
        data_df=group,
        group_id=group_id if isinstance(group_id, tuple) else (group_id,),
        id_columns=id_columns,
        timestamp_column=timestamp_column,
        context_length=context_length,
        prediction_length=prediction_length,
        drop_cols=drop_cols,
        stride=stride,
        fill_value=fill_value,
        **extra_kwargs,
    )


class PretrainDFDataset(BaseConcatDFDataset):
    """A dataset used for masked pre-training.

    Args:
        data (pd.DataFrame): Underlying pandas dataframe.
        id_columns (List[str], optional): List of columns which contain id information to separate distinct time series. Defaults to [].
        timestamp_column (Optional[str], optional): Name of the timestamp column. Defaults to None.
        target_columns (List[str], optional): List of column names which identify the target channels in the input, these are the
            columns that will be predicted. Defaults to [].
        context_length (int, optional): Length of historical data used when creating individual examples in the torch dataset. Defaults to 1.
        num_workers (int, optional): (Currently not used) Number of workers. Defaults to 1.
        stride (int, optional): Stride at which windows are produced. Defaults to 1.
        fill_value (Union[float, int], optional): Value used to fill any missing values. Defaults to 0.0.
        enable_padding (bool, optional): If True, windows of context_length+prediction_length which are too short are padded with zeros. Defaults to True.
            If False, a warning is issued when the input data does not contain sufficient records to create a non-empty dataset.



    The resulting dataset returns records (dictionaries) containing:
        past_values: tensor of past values of the target columns of length equal to context length
        past_observed_mask: tensor indicating which values are observed in the past values tensor
        timestamp: the timestamp of the end of the context window
        id: a tuple of id values (taken from the id columns) containing the id information of the time series segment
    """

    def __init__(
        self,
        data: pd.DataFrame,
        id_columns: List[str] = [],
        timestamp_column: Optional[str] = None,
        target_columns: List[str] = [],
        context_length: int = 1,
        num_workers: int = 1,
        stride: int = 1,
        fill_value: Union[float, int] = 0.0,
        enable_padding: bool = True,
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
            stride=stride,
            fill_value=fill_value,
            enable_padding=enable_padding,
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
            stride: int = 1,
            fill_value: Union[float, int] = 0.0,
            enable_padding: bool = True,
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
                stride=stride,
                fill_value=fill_value,
                zero_padding=enable_padding,
            )

        def __getitem__(self, index):
            index = self._check_index(index)

            time_id = index * self.stride
            seq_x = self.X.iloc[time_id : time_id + self.context_length].values
            ret = {
                "past_values": np_to_torch(np.nan_to_num(seq_x, nan=self.fill_value)),
                "past_observed_mask": np_to_torch(~np.isnan(seq_x)),
            }
            if self.datetime_col:
                ret["timestamp"] = self.timestamps[time_id + self.context_length - 1]
            if self.group_id:
                ret["id"] = self.group_id

            return ret


class ForecastDFDataset(BaseConcatDFDataset):
    """A dataset used for forecasting pretraing and inference

    Args:
        data (pd.DataFrame): Underlying pandas dataframe.
        id_columns (List[str], optional): List of columns which contain id information to separate distinct time series. Defaults
            to [].
        timestamp_column (Optional[str], optional): Name of the timestamp column. Defaults to None.
        target_columns (List[str], optional): List of column names which identify the target channels in the input, these are the
            columns that will be predicted. Defaults to [].
        observable_columns (List[str], optional): List of column names which identify the observable channels in the input.
            Observable channels are channels which we have knowledge about in the past and future. For example, weather
            conditions such as temperature or precipitation may be known or estimated in the future, but cannot be
            changed. Defaults to [].
        control_columns (List[str], optional): List of column names which identify the control channels in the input. Control
            channels are similar to observable channels, except that future values may be controlled. For example, discount
            percentage of a particular product is known and controllable in the future. Defaults to [].
        conditional_columns (List[str], optional): List of column names which identify the conditional channels in the input.
            Conditional channels are channels which we know in the past, but do not know in the future. Defaults to [].
        categorical_columns (List[str]): List of column names which identify time-varying categorical-valued channels in the input.
            Defaults to [].
        static_categorical_columns (List[str], optional): List of column names which identify categorical-valued channels in the
            input which are fixed over time. Defaults to [].
        context_length (int, optional): Length of historical data used when creating individual examples in the torch dataset.
            Defaults to 1.
        prediction_length (int, optional): Length of the future forecast. Defaults to 1.
        num_workers (int, optional): (Currently not used) Number of workers. Defaults to 1.
        frequency_token (Optional[int], optional): An integer representing the frequency of the data. Please see for an example of
            frequency token mappings. Defaults to None.
        autoregressive_modeling (bool, optional): (Experimental) If False, any target values in the context window are masked and
            replaced by 0. If True, the context window contains all the historical target information. Defaults to True.
        stride (int, optional): Stride at which windows are produced. Defaults to 1.
        fill_value (Union[float, int], optional): Value used to fill any missing values. Defaults to 0.0.
        masking_specification (List[Tuple[str, Union[int, Tuple[int, int]]]], optional): Allow masking the history (past values) of
            specific columns. The complete specification is a list of individual column masking specifications. A column masking
            specification is a 2-tuple where the first index specifies a column name. The second index specifies and index/indices to
            mask in each of the the context windows (past_values) generated for that column. If a single index is provided, masking
            will begin at that index and continue to the end of the context window. If a tuple of two values is provded, they are
            treated as python list indices; the values given by these indices will be masked.
        enable_padding (bool, optional): If True, windows of context_length+prediction_length which are too short are padded with zeros. Defaults to True.
            If False, a warning is issued when the input data does not contain sufficient records to create a non-empty dataset.
        metadata_columns (List[str], optional): List of columns to be passed through to the output of the dataset as `metadata`.
            Defaults to [].
        impute_method (str, optional): Enables imputation in the past_values of the dataset output. Possible options are in the
            ImputeMethod enum, "forward_fill" fills the values using previous values. Any values which cannot be filled are filled with
            `fill_value`. "linear" use linear interpolation to imput the missing values. None: use regular filling with `fill_value`.
            Defaults to None.

    The resulting dataset returns records (dictionaries) containing:
        past_values: tensor of past values of the target columns of length equal to context length (context_length x number of features)
        past_observed_mask: tensor indicating which values are observed in the past values tensor (context_length x number of features)
        future_values: tensor of future values of the target columns of length equal to prediction length (prediction_length x number of features)
        future_observed_mask: tensor indicating which values are observed in the future values tensor (prediction_length x number of features)
        freq_token: tensor containing the frequency token (scalar)
        static_categorical_features: tensor of static categorical features (1 x len(static_categorical_columns))
        timestamp: the timestamp of the end of the context window
        id: a tuple of id values (taken from the id columns) containing the id information of the time series segment

        where number of features is the total number of columns specified in target_columns, observable_columns, control_columns,
            conditional_columns
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
        categorical_columns: List[str] = [],
        static_categorical_columns: List[str] = [],
        context_length: int = 1,
        prediction_length: int = 1,
        num_workers: int = 1,
        frequency_token: Optional[int] = None,
        autoregressive_modeling: bool = True,
        stride: int = 1,
        fill_value: Union[float, int] = 0.0,
        masking_specification: Optional[List[Tuple[str, Union[int, Tuple[int, int]]]]] = None,
        enable_padding: bool = True,
        metadata_columns: List[str] = [],
        impute_method: Optional[str] = None,
    ):
        # output_columns_tmp = input_columns if output_columns == [] else output_columns

        if not (
            impute_method is None
            or impute_method == ImputeMethod.FORWARD_FILL.value
            or impute_method == ImputeMethod.LINEAR.value
        ):
            raise ValueError(f"Unknown impute_method {self.impute_method}.")

        super().__init__(
            data_df=data,
            id_columns=id_columns,
            timestamp_column=timestamp_column,
            num_workers=num_workers,
            context_length=context_length,
            prediction_length=prediction_length,
            fill_value=fill_value,
            cls=self.BaseForecastDFDataset,
            stride=stride,
            enable_padding=enable_padding,
            # extra_args
            target_columns=target_columns,
            observable_columns=observable_columns,
            control_columns=control_columns,
            conditional_columns=conditional_columns,
            categorical_columns=categorical_columns,
            static_categorical_columns=static_categorical_columns,
            frequency_token=frequency_token,
            autoregressive_modeling=autoregressive_modeling,
            masking_specification=masking_specification,
            metadata_columns=metadata_columns,
            impute_method=impute_method,
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
            categorical_columns: List[str] = [],
            static_categorical_columns: List[str] = [],
            frequency_token: Optional[int] = None,
            autoregressive_modeling: bool = True,
            stride: int = 1,
            fill_value: Union[float, int] = 0.0,
            masking_specification: Optional[List[Tuple[str, Union[int, Tuple[int, int]]]]] = None,
            enable_padding: bool = True,
            metadata_columns: List[str] = [],
            impute_method: Optional[str] = ImputeMethod.FORWARD_FILL.value,
        ):
            self.frequency_token = frequency_token
            self.target_columns = target_columns
            self.observable_columns = observable_columns
            self.control_columns = control_columns
            self.conditional_columns = conditional_columns
            self.categorical_columns = categorical_columns
            self.static_categorical_columns = static_categorical_columns
            self.autoregressive_modeling = autoregressive_modeling
            self.masking_specification = masking_specification
            self.metadata_columns = metadata_columns
            self.impute_method = impute_method

            x_cols = join_list_without_repeat(
                target_columns,
                observable_columns,
                control_columns,
                conditional_columns,
                categorical_columns,
            )
            y_cols = copy.copy(x_cols)

            self.column_name_to_index_map = {k: v for v, k in enumerate(x_cols)}

            # check non-autoregressive case
            if len(target_columns) == len(x_cols) and not self.autoregressive_modeling:
                raise ValueError(
                    "Non-autoregressive modeling was chosen, but there are no input columns for prediction."
                )

            # masking for conditional values which are not observed during future period
            self.y_mask_conditional = np.array([(c in conditional_columns) for c in y_cols])

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
                stride=stride,
                fill_value=fill_value,
                zero_padding=enable_padding,
            )

        def __getitem__(self, index):
            # seq_x: batch_size x seq_len x num_x_cols
            index = self._check_index(index)

            time_id = index * self.stride

            seq_x = self.X.iloc[time_id : time_id + self.context_length].values.astype(np.float32)
            if not self.autoregressive_modeling:
                seq_x[:, self.x_mask_targets] = 0

            if self.masking_specification is not None:
                seq_x = apply_masking_specification(
                    seq_x,
                    masking_specification=self.masking_specification,
                    column_name_to_index_map=self.column_name_to_index_map,
                )

            if self.impute_method == ImputeMethod.FORWARD_FILL.value:
                seq_x_imputed = impute_forward_fill(seq_x, self.fill_value)
            elif self.impute_method == ImputeMethod.LINEAR.value:
                # interpolate for each channel
                seq_x_imputed = np.apply_along_axis(interpolate_by_var, 0, seq_x)
            elif self.impute_method is None:
                seq_x_imputed = np.nan_to_num(seq_x, nan=self.fill_value)
            else:
                raise ValueError(f"Unknown impute_method {self.impute_method}.")

            # seq_y: batch_size x pred_len x num_x_cols
            seq_y = self.y.iloc[
                time_id + self.context_length : time_id + self.context_length + self.prediction_length
            ].values
            seq_y[:, self.y_mask_conditional] = 0

            ret = {
                "past_values": np_to_torch(seq_x_imputed),
                "future_values": np_to_torch(np.nan_to_num(seq_y, nan=self.fill_value)),
                "past_observed_mask": np_to_torch(~np.isnan(seq_x)),
                "future_observed_mask": np_to_torch(~np.isnan(seq_y)),
            }

            if self.datetime_col:
                ret["timestamp"] = self.timestamps[time_id + self.context_length - 1]

            if self.group_id:
                ret["id"] = self.group_id

            if self.frequency_token is not None:
                ret["freq_token"] = torch.tensor(self.frequency_token, dtype=torch.int)

            if self.static_categorical_columns:
                categorical_values = self.data_df[self.static_categorical_columns].values[0, :]
                ret["static_categorical_values"] = np_to_torch(categorical_values)

            if self.metadata_columns:
                ret["metadata"] = self.data_df[self.metadata_columns].values[
                    time_id : time_id + self.context_length + self.prediction_length, :
                ]

            return ret

        def __len__(self):
            return max((len(self.X) - self.context_length - self.prediction_length) // self.stride + 1, 0)


class ImputeForecastDFDataset(BaseConcatDFDataset):
    """A dataset used for forecasting pretraing and inference

    Args:
        data (pd.DataFrame): Underlying pandas dataframe.
        id_columns (List[str], optional): List of columns which contain id information to separate distinct time series. Defaults
            to [].
        timestamp_column (Optional[str], optional): Name of the timestamp column. Defaults to None.
        target_columns (List[str], optional): List of column names which identify the target channels in the input, these are the
            columns that will be predicted. Defaults to [].
        observable_columns (List[str], optional): List of column names which identify the observable channels in the input.
            Observable channels are channels which we have knowledge about in the past and future. For example, weather
            conditions such as temperature or precipitation may be known or estimated in the future, but cannot be
            changed. Defaults to [].
        control_columns (List[str], optional): List of column names which identify the control channels in the input. Control
            channels are similar to observable channels, except that future values may be controlled. For example, discount
            percentage of a particular product is known and controllable in the future. Defaults to [].
        conditional_columns (List[str], optional): List of column names which identify the conditional channels in the input.
            Conditional channels are channels which we know in the past, but do not know in the future. Defaults to [].
        static_categorical_columns (List[str], optional): List of column names which identify categorical-valued channels in the
            input which are fixed over time. Defaults to [].
        context_length (int, optional): Length of historical data used when creating individual examples in the torch dataset.
            Defaults to 1.
        prediction_length (int, optional): Length of the future forecast. Defaults to 1.
        num_workers (int, optional): (Currently not used) Number of workers. Defaults to 1.
        frequency_token (Optional[int], optional): An integer representing the frequency of the data. Please see for an example of
            frequency token mappings. Defaults to None.
        stride (int, optional): Stride at which windows are produced. Defaults to 1.
        fill_value (Union[float, int], optional): Value used to fill any missing values. Defaults to 0.0.
        masking_specification (List[Tuple[str, Union[int, Tuple[int, int]]]], optional): Allow masking the history (past values) of
            specific columns. The complete specification is a list of individual column masking specifications. A column masking
            specification is a 2-tuple where the first index specifies a column name. The second index specifies and index/indices to
            mask in each of the the context windows (past_values) generated for that column. If a single index is provided, masking
            will begin at that index and continue to the end of the context window. If a tuple of two values is provded, they are
            treated as python list indices; the values given by these indices will be masked.
        artificial_missing_rate (Float, optional): Rate at which points are selected to be marked as artificially missing. Defaults
            to 0.0 (no artificial missing).
        artificial_missing_columns (List[str], optional): Column names on which to apply artificial missing. Defaults to None.
        artificial_missing_at_time_t (bool, optional): If True the point at time t (last point in the context window) is marked as
            missing. Defaults to False.
        metadata_columns (List[str], optional): List of columns to be passed through to the output of the dataset as `metadata`.
            Defaults to [].
        impute_method (str, optional): Enables imputation in the past_values of the dataset output. Possible options are in the
            ImputeMethod enum, "forward_fill" fills the values using previous values. Any values which cannot be filled are filled with
            `fill_value`. "linear" use linear interpolation to imput the missing values. None: use regular filling with `fill_value`.
            Defaults to None.

    The resulting dataset returns records (dictionaries) containing:
        past_values: tensor of past values of the target columns of length equal to context length (context_length x number of features)
        past_observed_mask: tensor indicating which values are observed in the past values tensor (context_length x number of features)
        future_values: tensor of future values of the target columns of length equal to prediction length (prediction_length x number of features)
        future_observed_mask: tensor indicating which values are observed in the future values tensor (prediction_length x number of features)
        freq_token: tensor containing the frequency token (scalar)
        static_categorical_features: tensor of static categorical features (1 x len(static_categorical_columns))
        timestamp: the timestamp of the end of the context window
        id: a tuple of id values (taken from the id columns) containing the id information of the time series segment

        where number of features is the total number of columns specified in target_columns, observable_columns, control_columns,
            conditional_columns
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
        stride: int = 1,
        fill_value: Union[float, int] = 0.0,
        masking_specification: Optional[List[Tuple[str, Union[int, Tuple[int, int]]]]] = None,
        artificial_missing_rate: float = 0.0,
        artificial_missing_columns: Optional[List[str]] = None,
        artificial_missing_at_time_t: bool = False,
        random_seed: int = 0,
        impute_method: Optional[str] = ImputeMethod.FORWARD_FILL.value,
    ):
        # output_columns_tmp = input_columns if output_columns == [] else output_columns

        super().__init__(
            data_df=data,
            id_columns=id_columns,
            timestamp_column=timestamp_column,
            num_workers=num_workers,
            context_length=context_length,
            prediction_length=prediction_length,
            fill_value=fill_value,
            cls=self.BaseImputeForecastDFDataset,
            stride=stride,
            # extra_args
            target_columns=target_columns,
            observable_columns=observable_columns,
            control_columns=control_columns,
            conditional_columns=conditional_columns,
            static_categorical_columns=static_categorical_columns,
            frequency_token=frequency_token,
            masking_specification=masking_specification,
            artificial_missing_rate=artificial_missing_rate,
            artificial_missing_columns=artificial_missing_columns,
            artificial_missing_at_time_t=artificial_missing_at_time_t,
            rng=np.random.default_rng(seed=random_seed),
            impute_method=impute_method,
        )
        self.n_inp = 2
        # for forecasting, the number of targets is the same as number of X variables
        self.n_targets = self.n_vars

    class BaseImputeForecastDFDataset(BaseDFDataset):
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
            stride: int = 1,
            fill_value: Union[float, int] = 0.0,
            masking_specification: Optional[List[Tuple[str, Union[int, Tuple[int, int]]]]] = None,
            artificial_missing_rate: float = 0.0,
            artificial_missing_columns: Optional[List[str]] = None,
            artificial_missing_at_time_t: bool = False,
            rng=np.random.default_rng(),
            impute_method: Optional[str] = ImputeMethod.FORWARD_FILL.value,
        ):
            self.frequency_token = frequency_token
            self.target_columns = target_columns
            self.observable_columns = observable_columns
            self.control_columns = control_columns
            self.conditional_columns = conditional_columns
            self.static_categorical_columns = static_categorical_columns
            self.masking_specification = masking_specification
            self.artificial_missing_rate = artificial_missing_rate
            self.artificial_missing_at_time_t = artificial_missing_at_time_t
            self.rng = rng
            self.impute_method = impute_method

            if artificial_missing_columns is None:
                self.artificial_missing_columns = copy.copy(target_columns)
            else:
                self.artificial_missing_columns = artificial_missing_columns

            x_cols = join_list_without_repeat(
                target_columns,
                observable_columns,
                control_columns,
                conditional_columns,
            )
            y_cols = copy.copy(x_cols)

            # masking for conditional values which are not observed during future period
            self._y_mask_conditional = np.array([(c in conditional_columns) for c in y_cols])

            self.column_name_to_index_map = {k: v for v, k in enumerate(x_cols)}

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
                stride=stride,
                fill_value=fill_value,
            )

            self._X_imputed = self.X.ffill().fillna(self.fill_value)
            # create a artificially_observed_mask
            # target columns only
            # we don't mask what is already missing

            self._artificial_missing_column_mask = np.array([(c in self.artificial_missing_columns) for c in x_cols])
            artificial_observed_mask = np.ones_like(self.X.values[:, self._artificial_missing_column_mask], dtype=bool)

            if artificial_missing_rate > 0:
                # obs_x = np.where(~np.isnan(self.X.values[:, self._artificial_missing_mask]))
                obs_x = np.nonzero(~np.isnan(self.X.values[:, self._artificial_missing_column_mask]))

                # process columnwise
                for col_idx in range(artificial_observed_mask.shape[1]):
                    # get a column
                    obs_x_tmp = obs_x[0][obs_x[1] == col_idx]

                    # select missing, and generate 2-tuple of indices
                    # k = int(artificial_observed_mask.shape[0] * self.artificial_missing_rate)
                    # if k >= len(obs_x_tmp):
                    k = int(len(obs_x_tmp) * self.artificial_missing_rate)

                    artificial_missing_indices = (
                        self.rng.choice(
                            obs_x_tmp,
                            k,
                            replace=False,
                        ),
                        np.array([col_idx] * k, dtype=int),
                    )
                    artificial_observed_mask[artificial_missing_indices] = False

            self._artificial_past_observed_mask = artificial_observed_mask

        def __getitem__(self, index):
            # seq_x: batch_size x seq_len x num_x_cols

            time_id = index * self.stride

            seq_x = self.X[time_id : time_id + self.context_length].values

            if self.masking_specification is not None:
                seq_x = apply_masking_specification(
                    seq_x,
                    masking_specification=self.masking_specification,
                    column_name_to_index_map=self.column_name_to_index_map,
                )

            if self.impute_method == ImputeMethod.FORWARD_FILL.value:
                seq_x_imputed = impute_forward_fill(seq_x, self.fill_value)
            elif self.impute_method == ImputeMethod.LINEAR.value:
                # interpolate for each channel
                seq_x_imputed = np.apply_along_axis(interpolate_by_var, 0, seq_x)
            elif self.impute_method is None:
                seq_x_imputed = np.nan_to_num(seq_x, nan=self.fill_value)

            # mask only on seq_x
            artificial_past_observed_seq = np.ones_like(seq_x, dtype=bool)
            artificial_past_observed_seq[:, self._artificial_missing_column_mask] = (
                self._artificial_past_observed_mask[time_id : time_id + self.context_length]
            )

            # enforce missing at time t
            if self.artificial_missing_at_time_t:
                # get columns where time t value is observed
                col_obs = ~np.isnan(seq_x[-1, self._artificial_missing_column_mask])
                artificial_past_observed_seq[-1, self._artificial_missing_column_mask] = ~col_obs

            # seq_y: batch_size x pred_len x num_x_cols
            seq_y = self.y[
                time_id + self.context_length : time_id + self.context_length + self.prediction_length
            ].values

            seq_y[:, self._y_mask_conditional] = 0

            ret = {
                "past_values": np_to_torch(seq_x_imputed),
                "future_values": np_to_torch(np.nan_to_num(seq_y, nan=self.fill_value)),
                "past_observed_mask": np_to_torch(~np.isnan(seq_x)),
                "future_observed_mask": np_to_torch(~np.isnan(seq_y)),
                "artificial_past_observed_mask": np_to_torch(artificial_past_observed_seq),
            }

            if self.datetime_col:
                ret["timestamp"] = self.timestamps[time_id + self.context_length - 1]

            if self.group_id:
                ret["id"] = self.group_id

            if self.frequency_token is not None:
                ret["freq_token"] = torch.tensor(self.frequency_token, dtype=torch.int)

            if self.static_categorical_columns:
                categorical_values = self.data_df[self.static_categorical_columns].values[0, :]
                ret["static_categorical_values"] = np_to_torch(categorical_values)

            return ret

        def __len__(self):
            return (len(self.X) - self.context_length - self.prediction_length) // self.stride + 1


class RegressionDFDataset(BaseConcatDFDataset):
    """A dataset used for forecasting pretraing and inference

    Args:
        data (pd.DataFrame): Underlying pandas dataframe.
        id_columns (List[str], optional): List of columns which contain id information to separate distinct time series. Defaults
            to [].
        timestamp_column (Optional[str], optional): Name of the timestamp column. Defaults to None.
        input_columns (List[str], optional): List of columns to use as inputs to the regression
        target_columns (List[str], optional): List of column names which identify the target channels in the input, these are the
            columns that will be predicted. Defaults to [].
        context_length (int, optional): Length of historical data used when creating individual examples in the torch dataset.
            Defaults to 1.
        num_workers (int, optional): (Currently not used) Number of workers. Defaults to 1.
        stride (int, optional): Stride at which windows are produced. Defaults to 1.
        fill_value (Union[float, int], optional): Value used to fill any missing values. Defaults to 0.0.
        enable_padding (bool, optional): If True, windows of context_length+prediction_length which are too short are padded with zeros. Defaults to True.
            If False, a warning is issued when the input data does not contain sufficient records to create a non-empty dataset.


    The resulting dataset returns records (dictionaries) containing:
        past_values: tensor of past values of the target columns of length equal to context length (context_length x len(input_columns))
        past_observed_mask: tensor indicating which values are observed in the past values tensor (context_length x len(input_columns))
        target_values: tensor of future values of the target columns of length equal to prediction length (prediction_length x len(target_columns))
        static_categorical_features: tensor of static categorical features (1 x len(static_categorical_columns))
        timestamp: the timestamp of the end of the context window
        id: a tuple of id values (taken from the id columns) containing the id information of the time series segment
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
        stride: int = 1,
        fill_value: Union[float, int] = 0.0,
        enable_padding: bool = True,
    ):
        # self.y_cols = y_cols

        super().__init__(
            data_df=data,
            id_columns=id_columns,
            timestamp_column=timestamp_column,
            num_workers=num_workers,
            context_length=context_length,
            prediction_length=0,
            cls=self.BaseRegressionDFDataset,
            input_columns=input_columns,
            target_columns=target_columns,
            static_categorical_columns=static_categorical_columns,
            stride=stride,
            fill_value=fill_value,
            enable_padding=enable_padding,
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
            stride: int = 1,
            fill_value: Union[float, int] = 0.0,
            enable_padding: bool = True,
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
                stride=stride,
                fill_value=fill_value,
                zero_padding=enable_padding,
            )

        def __getitem__(self, index):
            # seq_x: batch_size x seq_len x num_x_cols
            index = self._check_index(index)

            time_id = index * self.stride
            seq_x = self.X.iloc[time_id : time_id + self.context_length].values
            seq_y = self.y.iloc[time_id + self.context_length - 1 : time_id + self.context_length].values.ravel()
            # return _torch(seq_x, seq_y)

            ret = {
                "past_values": np_to_torch(np.nan_to_num(seq_x, nan=self.fill_value)),
                "target_values": np_to_torch(np.nan_to_num(seq_y, nan=self.fill_value)),
                "past_observed_mask": np_to_torch(~np.isnan(seq_x)),
            }
            if self.datetime_col:
                ret["timestamp"] = self.timestamps[time_id + self.context_length - 1]

            if self.group_id:
                ret["id"] = self.group_id

            if self.static_categorical_columns:
                categorical_values = self.data_df[self.static_categorical_columns].values[0, :]
                ret["static_categorical_values"] = np_to_torch(categorical_values)

            return ret


class ClassificationDFDataset(BaseConcatDFDataset):
    """A dataset used for forecasting pretraing and inference

    Args:
        data (pd.DataFrame): Underlying pandas dataframe.
        id_columns (List[str], optional): List of columns which contain id information to separate distinct time series. Defaults
            to [].
        timestamp_column (Optional[str], optional): Name of the timestamp column. Defaults to None.
        input_columns (List[str], optional): List of columns to use as inputs to the regression
        label_column (str, optional): List of column names which identify the label of the time series. Defaults to "label".
        context_length (int, optional): Length of historical data used when creating individual examples in the torch dataset.
            Defaults to 1.
        num_workers (int, optional): (Currently not used) Number of workers. Defaults to 1.
        stride (int, optional): Stride at which windows are produced. Defaults to 1.
        fill_value (Union[float, int], optional): Value used to fill any missing values. Defaults to 0.0.
        enable_padding (bool, optional): If True, windows of context_length+prediction_length which are too short are padded with zeros. Defaults to True.
            If False, a warning is issued when the input data does not contain sufficient records to create a non-empty dataset. Only applies
            when full_series==False.
        full_series (bool, optional): If True, each series is used when creating a single entry in the classification dataset. In this case
            the series is interpolated to match the desired context_length. No padding is done. If False, each series is windowed acording to
            the provided context_length. Defaults to False.


    The resulting dataset returns records (dictionaries) containing:
        past_values: tensor of past values of the target columns of length equal to context length (context_length x len(input_columns))
        past_observed_mask: tensor indicating which values are observed in the past values tensor (context_length x len(input_columns))
        target_values: tensor containing the label (scalar)
        static_categorical_features: tensor of static categorical features (1 x len(static_categorical_columns))
        timestamp: the timestamp of the end of the context window
        id: a tuple of id values (taken from the id columns) containing the id information of the time series segment
    """

    def __init__(
        self,
        data: pd.DataFrame,
        id_columns: List[str] = [],
        timestamp_column: Optional[str] = None,
        input_columns: List[str] = [],
        label_column: str = "label",
        static_categorical_columns: List[str] = [],
        context_length: int = 1,
        num_workers: int = 1,
        stride: int = 1,
        fill_value: Union[float, int] = 0.0,
        enable_padding: bool = True,
        full_series: bool = True,
    ):
        if full_series:
            if not is_nested_dataframe(data, column=input_columns[0]):
                raise ValueError(
                    "The provided data does not appear to contain row entries which contain a full time series."
                )

            check_nested_lengths(data, columns=input_columns)
            if enable_padding:
                enable_padding = False

            super().__init__(
                data_df=data,
                id_columns=id_columns,
                timestamp_column=timestamp_column,
                num_workers=num_workers,
                context_length=context_length,
                prediction_length=0,
                cls=_FullSeriesClassificationDFDataset,
                input_columns=input_columns,
                label_column=label_column,
                static_categorical_columns=static_categorical_columns,
                stride=stride,
                fill_value=fill_value,
                enable_padding=enable_padding,
            )

        else:
            super().__init__(
                data_df=data,
                id_columns=id_columns,
                timestamp_column=timestamp_column,
                num_workers=num_workers,
                context_length=context_length,
                prediction_length=0,
                cls=_WindowedSeriesClassificationDFDataset,
                input_columns=input_columns,
                label_column=label_column,
                static_categorical_columns=static_categorical_columns,
                stride=stride,
                fill_value=fill_value,
                enable_padding=enable_padding,
            )

        self.n_inp = 2
        self.full_series = full_series


def np_to_torch(data: np.array, float_type=np.float32):
    if data.dtype == "float":
        return torch.from_numpy(data.astype(float_type))
    elif data.dtype == "int":
        return torch.from_numpy(data)
    elif data.dtype == "bool":
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
        if len(df) < 2:
            pad_df[timestamp_column] = None
        elif (
            (df[timestamp_column].dtype.type == np.datetime64)
            or (df[timestamp_column].dtype == int)
            or (df[timestamp_column].dtype == np.int64)
        ):
            last_timestamp = df.iloc[0][timestamp_column]
            period = df.iloc[1][timestamp_column] - df.iloc[0][timestamp_column]
            prepended_timestamps = [last_timestamp + offset * period for offset in range(-fill_length, 0)]
            pad_df[timestamp_column] = prepended_timestamps
        else:
            pad_df[timestamp_column] = None
        pad_df[timestamp_column] = pad_df[timestamp_column].astype(df[timestamp_column].dtype)

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
            return False, col
    return True, None


def apply_masking_specification(
    past_values_tensor: np.ndarray,
    masking_specification: List[Tuple[str, Union[int, Tuple[int, int]]]],
    column_name_to_index_map: Dict[str, int],
) -> np.ndarray:
    """Apply the desired mask defined by masking_specification.

    Args:
        past_values_tensor (np.ndarray): Tensor of past values, should have shape (context_length, num_channels)
        masking_specification (List[Tuple[str, Union[int, Tuple[int, int]]]], optional): Allow masking the history (past values) of
            specific columns. The complete specification is a list of individual column masking specifications. A column masking
            specification is a 2-tuple where the first index specifies a column name. The second index specifies and index/indices to
            mask in each of the the context windows (past_values) generated for that column. If a single index is provided, masking
            will begin at that index and continue to the end of the context window. If a tuple of two values is provded, they are
            treated as python list indices; the values given by these indices will be masked.
        column_name_to_index_map (Dict[str, int]): A mapping of string (column names) to the indice of that column

    Returns:
        np.ndarry: Tensor with values masked
    """

    past_values_tensor = past_values_tensor.copy()
    for col_name, spec in masking_specification:
        col_idx = column_name_to_index_map[col_name]
        if isinstance(spec, (tuple, list)) and len(spec) == 2:
            past_values_tensor[spec[0] : spec[1], col_idx] = np.nan
        else:
            past_values_tensor[spec:, col_idx] = np.nan
    return past_values_tensor


def interpolate_by_var(miss_seq: np.ndarray) -> np.ndarray:
    """Interpolate each column of the input numpy array using np.interpolate.

    Args:
        miss_seq (np.ndarray): Sequence potentially containing missing values.

    Returns:
        np.ndarray: Imputed sequence.
    """
    nans = np.isnan(miss_seq)
    if nans.sum() == 0:
        return miss_seq

    imputed = np.interp(np.where(nans)[0], np.where(~nans)[0], miss_seq[~nans])
    miss_seq_copy = miss_seq.copy()
    miss_seq_copy[nans] = imputed
    return miss_seq_copy


def impute_forward_fill(miss_seq: np.ndarray, fill_value: float) -> np.ndarray:
    """Impute each column of the input numpy array using forward filling.

    Args:
        miss_seq (np.ndarray): Sequence potentially containing missing values.
        fill_value (float): Value used to impute any values that cannot be filled

    Returns:
        np.ndarray: Imputed sequence.
    """
    # impute the numpy matrix, time is the first dimension
    # based on https://stackoverflow.com/questions/41190852/most-efficient-way-to-forward-fill-nan-values-in-numpy-array
    mask = np.isnan(miss_seq)
    idx = np.where(~mask, np.expand_dims(np.arange(mask.shape[0]), 1), 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    miss_seq_copy = miss_seq.copy()
    seq_x_imputed = miss_seq_copy[idx, np.arange(idx.shape[1])[None, :]]

    # fill any remaining nan
    seq_x_imputed = np.nan_to_num(seq_x_imputed, nan=fill_value)
    return seq_x_imputed


class _WindowedSeriesClassificationDFDataset(BaseDFDataset):
    """Standard windowed classification dataset.

    Windows of size context_length are extracted from the input time series. The labels are chosen from the label_column
    at the index of the end of the window.
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
        label_column: str = "label",
        input_columns: List[str] = [],
        static_categorical_columns: List[str] = [],
        stride: int = 1,
        fill_value: Union[float, int] = 0.0,
        enable_padding: bool = True,
    ):
        self.label_column = label_column
        self.input_columns = input_columns
        self.static_categorical_columns = static_categorical_columns

        x_cols = input_columns
        y_cols = label_column

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
            stride=stride,
            fill_value=fill_value,
            zero_padding=enable_padding,
            full_series=False,
        )

    def __getitem__(self, index):
        # seq_x: batch_size x seq_len x num_x_cols
        index = self._check_index(index)

        time_id = index * self.stride
        seq_x = self.X.iloc[time_id : time_id + self.context_length].values
        # seq_y = self.y[time_id + self.context_length - 1 : time_id + self.context_length].values.ravel()
        seq_y = self.y.iloc[time_id + self.context_length - 1].values[0]

        ret = {
            "past_values": np_to_torch(np.nan_to_num(seq_x, nan=self.fill_value)),
            "target_values": torch.tensor(np.nan_to_num(seq_y, nan=self.fill_value), dtype=torch.int64),
            "past_observed_mask": np_to_torch(~np.isnan(seq_x)),
        }
        if self.datetime_col:
            ret["timestamp"] = self.timestamps[time_id + self.context_length - 1]

        if self.group_id:
            ret["id"] = self.group_id

        if self.static_categorical_columns:
            categorical_values = self.data_df[self.static_categorical_columns].values[0, :]
            ret["static_categorical_values"] = np_to_torch(categorical_values)

        return ret


class _FullSeriesClassificationDFDataset(BaseDFDataset):
    """Full series classification dataset.

    Works with nested input dataframes (i.e., dataframes which contain pd.Series as entries). In this case, each series
    is interpolated to the desired context_length.
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
        label_column: str = "label",
        input_columns: List[str] = [],
        static_categorical_columns: List[str] = [],
        stride: int = 1,
        fill_value: Union[float, int] = 0.0,
        enable_padding: bool = True,
    ):
        self.label_column = label_column
        self.input_columns = input_columns
        self.static_categorical_columns = static_categorical_columns

        x_cols = input_columns
        y_cols = label_column

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
            stride=stride,
            fill_value=fill_value,
            zero_padding=enable_padding,
            full_series=True,
        )

        # after init
        # self.X should contain the data including all inputs (i.e. input columns)
        # X is in the nexted format where X has series entries in some places

    def __len__(self):
        return len(self.X) // self.stride

    def __getitem__(self, index):
        index = self._check_index(index)

        ind = index * self.stride

        # each row has entries which are series
        # assume also that static categorical are static for each row (not necessarily across rows)

        # first convert rows of series to full tensor: len x features
        # assumes consistent length
        X = np.asarray([s.to_list() for s in self.X.iloc[ind].values])
        X = np_to_torch(np.nan_to_num(X, nan=self.fill_value))
        y = np_to_torch(self.y.iloc[ind].values)

        X = X.unsqueeze(dim=0)  # transpose(1, 0). c l --->  1 c l
        X = torch.nn.functional.interpolate(
            X, self.context_length, mode="linear"
        )  # need a 3D tensor of shape B C L    # 1 c l ---> 1 c context_points
        X = X.squeeze(dim=0).transpose(0, 1)  # 1 c cp ---> cp c

        y = torch.squeeze(y)

        ret = {
            "past_values": X,
            "target_values": y,
            # omit or we could do: "past_observed_mask": np_to_torch(np.ones(X.shape)),
        }

        if self.datetime_col:
            if isinstance(self.timestamps[ind], pd.Series):
                ret["timestamp"] = self.timestamps[ind].iloc[-1]
            else:
                ret["timestamp"] = self.timestamps[ind]

        # concatenate the row to the id
        ret["id"] = self.group_id + (ind,) if self.group_id else (ind,)

        if self.static_categorical_columns:
            categorical_values = self.data_df[self.static_categorical_columns].iloc[ind].values
            ret["static_categorical_values"] = np_to_torch(categorical_values)

        return ret


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

    d7 = ForecastDFDataset(data_df=df, x_cols=["A", "B"], group_ids=["g1"], seq_len=2, pred_len=2)
