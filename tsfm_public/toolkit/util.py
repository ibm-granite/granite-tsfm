# Copyright contributors to the TSFM project
#
"""Basic functions and utilities"""

import copy
from datetime import datetime
from distutils.util import strtobool
from typing import Any, Dict, List, Optional, Union

import pandas as pd


def select_by_timestamp(
    df: pd.DataFrame,
    timestamp_column: str = "timestamp",
    start_timestamp: Optional[Union[str, datetime]] = None,
    end_timestamp: Optional[Union[str, datetime]] = None,
) -> pd.DataFrame:
    """Select a portion of a dataset based on timestamps.
    Note that the range selected is inclusive of the starting timestamp.

    Args:
        df (pd.DataFrame): Input dataframe.
        timestamp_column (str, optional): Timestamp column in the dataset. Defaults to "timestamp".
        start_timestamp (Optional[Union[str, datetime]], optional): Timestamp of the starting point.
            Defaults to None. Use None to specify the start of the data.
        end_timestamp (Optional[Union[str, datetime]], optional): Timestamp of the ending point.
            Use None to specify the end of the data. Defaults to None.

    Raises:
        ValueError: User must specify either start_timestamp or end_timestamp.

    Returns:
        pd.DataFrame: Subset of the dataframe.
    """

    if not start_timestamp and not end_timestamp:
        raise ValueError("At least one of start_timestamp or end_timestamp must be specified.")

    if not start_timestamp:
        return df[df[timestamp_column] < end_timestamp]

    if not end_timestamp:
        return df[df[timestamp_column] >= start_timestamp]

    return df[(df[timestamp_column] >= start_timestamp) & (df[timestamp_column] < end_timestamp)]


def select_by_index(
    df: pd.DataFrame,
    id_columns: Optional[List[str]] = None,
    start_index: Optional[int] = None,
    end_index: Optional[int] = None,
) -> pd.DataFrame:
    """Select a portion of a dataset based on integer indices into the data.
    Note that the range selected is inclusive of the starting index. When ID columns are specified
    the selection is done per-time series (i.e., the indices are used relative to each time series).

    Args:
        df (pd.DataFrame): Input dataframe.
        id_columns (List[str], optional): Columns which specify the IDs in the dataset. Defaults to None.
        start_index (Optional[int], optional): Index of the starting point.
            Defaults to None. Use None to specify the start of the data.
        end_index (Optional[Union[str, datetime]], optional): Index of the ending point.
            Use None to specify the end of the data. Defaults to None.

    Raises:
        ValueError: User must specify either start_index or end_index.

    Returns:
        pd.DataFrame: Subset of the dataframe.
    """
    if not start_index and not end_index:
        raise ValueError("At least one of start_index or end_index must be specified.")

    if not id_columns:
        return _split_group_by_index(df, start_index=start_index, end_index=end_index).copy()

    groups = df.groupby(_get_groupby_columns(id_columns))
    result = []
    for name, group in groups:
        result.append(_split_group_by_index(group, name=name, start_index=start_index, end_index=end_index))

    return pd.concat(result)


def select_by_relative_fraction(
    df: pd.DataFrame,
    id_columns: Optional[List[str]] = None,
    start_fraction: Optional[float] = None,
    start_offset: Optional[int] = 0,
    end_fraction: Optional[float] = None,
) -> pd.DataFrame:
    """Select a portion of a dataset based on relative fractions of the data.
    Note that the range selected is inclusive of the starting index. When ID columns are specified
    the selection is done per-time series (i.e., the fractions are used relative to each time series length).

    The indices are computed as:
    index_start_i = floor(length_i * start_fraction) - start_offset
    index_end_i = floor(length_i * end_fraction)

    Args:
        df (pd.DataFrame): Input dataframe.
        id_columns (List[str], optional): Columns which specify the IDs in the dataset. Defaults to None.
        start_fraction (Optional[float], optional): The fraction to specify the start of the selection. Use None to specify the start of the dataset. Defaults to None.
        start_offset (Optional[int], optional): An optional offset to apply to the starting point of
            each subseries. A non-negative value should be used. Defaults to 0.
        end_fraction (Optional[float], optional): The fraction to specify the end of the selection.
            Use None to specify the end of the dataset. Defaults to None.

    Raises:
        ValueError: Raised when the user does not specify either start_index or end_index. Also raised
            when a negative value of start_offset is provided.

    Returns:
        pd.DataFrame: Subset of the dataframe.
    """
    if not start_fraction and not end_fraction:
        raise ValueError("At least one of start_fraction or end_fraction must be specified.")

    if start_offset < 0:
        raise ValueError("The value of start_offset should ne non-negative.")

    if not id_columns:
        return _split_group_by_fraction(
            df,
            start_fraction=start_fraction,
            end_fraction=end_fraction,
            start_offset=start_offset,
        ).copy()

    groups = df.groupby(_get_groupby_columns(id_columns))
    result = []
    for name, group in groups:
        result.append(
            _split_group_by_fraction(
                group,
                name=name,
                start_fraction=start_fraction,
                end_fraction=end_fraction,
                start_offset=start_offset,
            )
        )

    return pd.concat(result)


def _get_groupby_columns(id_columns: List[str]) -> Union[List[str], str]:
    if not isinstance(id_columns, (List)):
        raise ValueError("id_columns must be a list")

    if len(id_columns) == 1:
        return id_columns[0]

    return id_columns


def _split_group_by_index(
    group_df: pd.DataFrame,
    name: Optional[str] = None,
    start_index: Optional[int] = None,
    end_index: Optional[int] = None,
) -> pd.DataFrame:
    if start_index and (start_index >= len(group_df)):
        msg = "Selection would result in an empty time series, please check start_index and time series length"
        msg = msg + f" (id = {name})" if name else msg
        raise ValueError(msg)

    # Also check that end_index <= len(group_df)?

    if not start_index:
        return group_df.iloc[:end_index,]

    if not end_index:
        return group_df.iloc[start_index:,]

    return group_df.iloc[start_index:end_index, :]


def _split_group_by_fraction(
    group_df: pd.DataFrame,
    name: Optional[str] = None,
    start_fraction: Optional[float] = None,
    start_offset: Optional[int] = 0,
    end_fraction: Optional[float] = None,
) -> pd.DataFrame:
    length = len(group_df)

    if start_fraction is not None:
        start_index = int(length * start_fraction) - start_offset
        if start_index < 0:
            if name:
                msg = f"Computed starting_index for id={name} is negative, please check individual time series lengths, start_fraction, and start_offset."
            else:
                msg = "Computed starting_index is negative, please check time series length, start_fraction, and start_offset."
            raise ValueError(msg)
    else:
        start_index = None

    if end_fraction is not None:
        end_index = int(length * end_fraction)
    else:
        end_index = None

    return _split_group_by_index(group_df=group_df, start_index=start_index, end_index=end_index)


def convert_tsf_to_dataframe(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
    value_column_name="series_value",
):
    col_names = []
    col_types = []
    all_data = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    with open(full_file_path_and_name, "r", encoding="cp1252") as file:
        for line in file:
            # Strip white space from start/end of line
            line = line.strip()

            if line:
                if line.startswith("@"):  # Read meta-data
                    if not line.startswith("@data"):
                        line_content = line.split(" ")
                        if line.startswith("@attribute"):
                            if len(line_content) != 3:  # Attributes have both name and type
                                raise Exception("Invalid meta-data specification.")

                            col_names.append(line_content[1])
                            col_types.append(line_content[2])
                        else:
                            if len(line_content) != 2:  # Other meta-data have only values
                                raise Exception("Invalid meta-data specification.")

                            if line.startswith("@frequency"):
                                frequency = line_content[1]
                            elif line.startswith("@horizon"):
                                forecast_horizon = int(line_content[1])
                            elif line.startswith("@missing"):
                                contain_missing_values = bool(strtobool(line_content[1]))
                            elif line.startswith("@equallength"):
                                contain_equal_length = bool(strtobool(line_content[1]))

                    else:
                        if len(col_names) == 0:
                            raise Exception("Missing attribute section. Attribute section must come before data.")

                        found_data_tag = True
                elif not line.startswith("#"):
                    if len(col_names) == 0:
                        raise Exception("Missing attribute section. Attribute section must come before data.")
                    elif not found_data_tag:
                        raise Exception("Missing @data tag.")
                    else:
                        if not started_reading_data_section:
                            started_reading_data_section = True
                            found_data_section = True
                            all_series = []

                            for col in col_names:
                                all_data[col] = []

                        full_info = line.split(":")

                        if len(full_info) != (len(col_names) + 1):
                            raise Exception("Missing attributes/values in series.")

                        series = full_info[len(full_info) - 1]
                        series = series.split(",")

                        if len(series) == 0:
                            raise Exception(
                                "A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series. Missing values should be indicated with ? symbol"
                            )

                        numeric_series = []

                        for val in series:
                            if val == "?":
                                numeric_series.append(replace_missing_vals_with)
                            else:
                                numeric_series.append(float(val))

                        if numeric_series.count(replace_missing_vals_with) == len(numeric_series):
                            raise Exception(
                                "All series values are missing. A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series."
                            )

                        all_series.append(pd.Series(numeric_series).array)

                        for i in range(len(col_names)):
                            att_val = None
                            if col_types[i] == "numeric":
                                att_val = int(full_info[i])
                            elif col_types[i] == "string":
                                att_val = str(full_info[i])
                            elif col_types[i] == "date":
                                att_val = datetime.strptime(full_info[i], "%Y-%m-%d %H-%M-%S")
                            else:
                                raise Exception(
                                    "Invalid attribute type."
                                )  # Currently, the code supports only numeric, string and date types. Extend this as required.

                            if att_val is None:
                                raise Exception("Invalid attribute value.")
                            else:
                                all_data[col_names[i]].append(att_val)

                line_count = line_count + 1

        if line_count == 0:
            raise Exception("Empty file.")
        if len(col_names) == 0:
            raise Exception("Missing attribute section.")
        if not found_data_section:
            raise Exception("Missing series information under data section.")

        all_data[value_column_name] = all_series
        loaded_data = pd.DataFrame(all_data)

        return (
            loaded_data,
            frequency,
            forecast_horizon,
            contain_missing_values,
            contain_equal_length,
        )


def get_split_params(
    split_config: Dict[str, List[int]], context_length=None
) -> Dict[str, Dict[str, Union[int, float]]]:
    """_summary_

    Args:
        split_config (Dict[str, List[int]]): _description_
        context_length (_type_, optional): _description_. Defaults to None.

    Returns:
        Dict[str, Dict[str, Union[int, float]]]: _description_
    """

    split_params = {}
    split_function = {}

    for group in ["train", "test", "valid"]:
        if split_config[group][1] < 1:
            split_params[group] = {
                "start_fraction": split_config[group][0],
                "end_fraction": split_config[group][1],
                "start_offset": (context_length if (context_length and group != "train") else 0),
            }
            split_function[group] = select_by_relative_fraction
        else:
            split_params[group] = {
                "start_index": (
                    split_config[group][0] - (context_length if (context_length and group != "train") else 0)
                ),
                "end_index": split_config[group][1],
            }
            split_function[group] = select_by_index

    return split_params, split_function


def convert_tsf(filename: str) -> pd.DataFrame:
    """Converts a tsf format file into a pandas dataframe.
    Returns the result in canonical multi-time series format, with an ID column, and timestamp.

    Args:
        filename (str): Input file name.

    Returns:
        pd.DataFrame: Converted time series
    """
    (
        loaded_data,
        frequency,
        forecast_horizon,
        contain_missing_values,
        contain_equal_length,
    ) = convert_tsf_to_dataframe(filename)

    dfs = []
    for index, item in loaded_data.iterrows():
        # todo: use actual dates for timestamp
        dfs.append(
            pd.DataFrame(
                {
                    "id": item.series_name,
                    "timestamp": range(len(item.series_value)),
                    "value": item.series_value,
                }
            )
        )

    df = pd.concat(dfs)
    return df


def join_list_without_repeat(*lists: List[List[Any]]) -> List[Any]:
    """Join multiple lists in sequence without repeating

    Returns:
        List[Any]: Combined list.
    """

    final = None
    final_set = set()
    for alist in lists:
        if final is None:
            final = copy.copy(alist)
        else:
            final = final + [item for item in alist if item not in final_set]
        final_set = set(final)
    return final
