# Copyright contributors to the TSFM project
#
"""Basic functions and utilities"""

import copy
import enum
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from pandas.api.types import is_datetime64_dtype


def strtobool(val):
    """Convert a string representation of truth to true (1) or false (0).

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.

    Source: setuptools/_distutils/util.py
    """
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return 1
    elif val in ("n", "no", "f", "false", "off", "0"):
        return 0
    else:
        raise ValueError(f"invalid truth value {val!r}")


class FractionLocation(enum.Enum):
    """`Enum` for the different locations where a fraction of data can be chosen."""

    FIRST = "first"
    LAST = "last"


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
    Note that the range selected is inclusive of the starting index, but exclusive of the end index. When ID
    columns are specified the selection is done per-time series (i.e., the indices are used relative to each
    time series). The indexing is intended to be similar to python-style indexing into lists, where the end
    of the specified range is not included.

    Args:
        df (pd.DataFrame): Input dataframe.
        id_columns (List[str], optional): Columns which specify the IDs in the dataset. Defaults to None.
        start_index (Optional[int], optional): Index of the starting point.
            Defaults to None. Use None to specify the start of the data.
        end_index (Optional[Union[str, datetime]], optional): Index for the end of the selection (not inclusive).
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


def select_by_fixed_fraction(
    df: pd.DataFrame,
    id_columns: Optional[List[str]] = None,
    fraction: float = 1.0,
    location: str = FractionLocation.FIRST.value,
    minimum_size: Optional[int] = 0,
) -> pd.DataFrame:
    """Select a portion of a dataset based on a fraction of the data.
    Fraction can either be located at the start (location = FractionLocation.FIRST) or at the end (location = FractionLocation.LAST)

    Args:
        df (pd.DataFrame): Input dataframe.
        id_columns (List[str], optional): Columns which specify the IDs in the dataset. Defaults to None.
        fraction (float): The fraction to select.
        location (str): Location of where to select the fraction Defaults to FractionLocation.FIRST.value.
        minimum_size (int, optional): Minimum size of the split. Defaults to None.

    Raises:
        ValueError: Raised when the fraction is not within the range [0,1].

    Returns:
        pd.DataFrame: Subset of the dataframe.
    """

    if fraction < 0 or fraction > 1:
        raise ValueError("The value of fraction should be between 0 and 1.")

    if not id_columns:
        return _split_group_by_fixed_fraction(
            df, fraction=fraction, location=location, minimum_size=minimum_size
        ).copy()

    groups = df.groupby(_get_groupby_columns(id_columns))
    result = []
    for name, group in groups:
        result.append(
            _split_group_by_fixed_fraction(
                group,
                name=name,
                fraction=fraction,
                location=location,
                minimum_size=minimum_size,
            )
        )

    return pd.concat(result)


def train_test_split(
    df: pd.DataFrame,
    id_columns: Optional[List[str]] = None,
    train: Union[int, float] = 0.7,
    test: Union[int, float] = 0.2,
    valid_test_offset: int = 0,
):
    # to do: add validation

    if not id_columns:
        return tuple(
            [
                tmp.copy()
                for tmp in _split_group_train_test(df, train=train, test=test, valid_test_offset=valid_test_offset)
            ]
        )

    groups = df.groupby(_get_groupby_columns(id_columns))
    result = []
    for name, group in groups:
        result.append(
            _split_group_train_test(
                group,
                name=name,
                train=train,
                test=test,
                valid_test_offset=valid_test_offset,
            )
        )

    result_train, result_valid, result_test = zip(*result)
    return pd.concat(result_train), pd.concat(result_valid), pd.concat(result_test)


def _split_group_train_test(
    group_df: pd.DataFrame,
    name: Optional[str] = None,
    train: Union[int, float] = 0.7,
    test: Union[int, float] = 0.2,
    valid_test_offset: int = 0,
):
    l = len(group_df)

    train_size = int(l * train)
    test_size = int(l * test)

    valid_size = l - train_size - test_size

    train_df = _split_group_by_index(group_df, name, start_index=0, end_index=train_size)

    valid_df = _split_group_by_index(
        group_df,
        name,
        start_index=train_size - valid_test_offset,
        end_index=train_size + valid_size,
    )

    test_df = _split_group_by_index(group_df, name, start_index=train_size + valid_size - valid_test_offset)

    return train_df, valid_df, test_df


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
    """Helper function for splitting by index."""
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
    """Helper function for splitting by relative fraction."""
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

    return _split_group_by_index(group_df=group_df, name=name, start_index=start_index, end_index=end_index)


def _split_group_by_fixed_fraction(
    group_df: pd.DataFrame,
    name: Optional[str] = None,
    fraction: float = 1.0,
    location: Optional[str] = None,
    minimum_size: Optional[int] = 0,
):
    """Helper function for splitting by fixed fraction."""
    l = len(group_df)
    fraction_size = int(fraction * (l - minimum_size)) + minimum_size

    if location == FractionLocation.FIRST.value:
        start_index = 0
        end_index = fraction_size
    elif location == FractionLocation.LAST.value:
        start_index = l - fraction_size
        end_index = l
    else:
        raise ValueError(
            f"`location` should be either `{FractionLocation.FIRST.value}` or `{FractionLocation.LAST.value}`"
        )

    return _split_group_by_index(group_df=group_df, name=name, start_index=start_index, end_index=end_index)


def convert_tsf_to_dataframe(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
    value_column_name="series_value",
):
    """Read a .tsf file into a pandas dataframe.

    Args:
        full_file_path_and_name (_type_): _description_
        replace_missing_vals_with (str, optional): _description_. Defaults to "NaN".
        value_column_name (str, optional): _description_. Defaults to "series_value".

    This code adopted from the Monash forecasting repository github:
    https://github.com/rakshitha123/TSForecasting/blob/master/utils/data_loader.py

    """

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


def convert_tsfile_to_dataframe(
    full_file_path_and_name,
    return_separate_X_and_y=True,
    replace_missing_vals_with="NaN",
):
    """Load data from a .ts file into a Pandas DataFrame.
    Parameters
    ----------
    full_file_path_and_name: str
        The full pathname of the .ts file to read.
    return_separate_X_and_y: bool
        true if X and Y values should be returned as separate Data Frames (
        X) and a numpy array (y), false otherwise.
        This is only relevant for data that
    replace_missing_vals_with: str
       The value that missing values in the text file should be replaced
       with prior to parsing.
    Returns
    -------
    DataFrame (default) or ndarray (i
        If return_separate_X_and_y then a tuple containing a DataFrame and a
        numpy array containing the relevant time-series and corresponding
        class values.
    DataFrame
        If not return_separate_X_and_y then a single DataFrame containing
        all time-series and (if relevant) a column "class_vals" the
        associated class values.


    This code adopted from sktime:
    https://github.com/sktime/sktime/blob/v0.30.0/sktime/datasets/_readers_writers/ts.py#L32-L615

    """
    # Initialize flags and variables used when parsing the file
    metadata_started = False
    data_started = False

    has_problem_name_tag = False
    has_timestamps_tag = False
    has_univariate_tag = False
    has_class_labels_tag = False
    has_data_tag = False

    previous_timestamp_was_int = None
    prev_timestamp_was_timestamp = None
    num_dimensions = None
    is_first_case = True
    instance_list = []
    class_val_list = []
    line_num = 0
    # Parse the file
    with open(full_file_path_and_name, "r", encoding="utf-8") as file:
        for line in file:
            # Strip white space from start/end of line and change to
            # lowercase for use below
            line = line.strip().lower()
            # Empty lines are valid at any point in a file
            if line:
                # Check if this line contains metadata
                # Please note that even though metadata is stored in this
                # function it is not currently published externally
                if line.startswith("@problemname"):
                    # Check that the data has not started
                    if data_started:
                        raise IOError("metadata must come before data")
                    # Check that the associated value is valid
                    tokens = line.split(" ")
                    token_len = len(tokens)
                    if token_len == 1:
                        raise IOError("problemname tag requires an associated value")
                    # problem_name = line[len("@problemname") + 1:]
                    has_problem_name_tag = True
                    metadata_started = True
                elif line.startswith("@timestamps"):
                    # Check that the data has not started
                    if data_started:
                        raise IOError("metadata must come before data")
                    # Check that the associated value is valid
                    tokens = line.split(" ")
                    token_len = len(tokens)
                    if token_len != 2:
                        raise IOError("timestamps tag requires an associated Boolean " "value")
                    elif tokens[1] == "true":
                        timestamps = True
                    elif tokens[1] == "false":
                        timestamps = False
                    else:
                        raise IOError("invalid timestamps value")
                    has_timestamps_tag = True
                    metadata_started = True
                elif line.startswith("@univariate"):
                    # Check that the data has not started
                    if data_started:
                        raise IOError("metadata must come before data")
                    # Check that the associated value is valid
                    tokens = line.split(" ")
                    token_len = len(tokens)
                    if token_len != 2:
                        raise IOError("univariate tag requires an associated Boolean  " "value")
                    elif tokens[1] == "true":
                        # univariate = True
                        pass
                    elif tokens[1] == "false":
                        # univariate = False
                        pass
                    else:
                        raise IOError("invalid univariate value")
                    has_univariate_tag = True
                    metadata_started = True
                elif line.startswith("@classlabel"):
                    # Check that the data has not started
                    if data_started:
                        raise IOError("metadata must come before data")
                    # Check that the associated value is valid
                    tokens = line.split(" ")
                    token_len = len(tokens)
                    if token_len == 1:
                        raise IOError("classlabel tag requires an associated Boolean  " "value")
                    if tokens[1] == "true":
                        class_labels = True
                    elif tokens[1] == "false":
                        class_labels = False
                    else:
                        raise IOError("invalid classLabel value")
                    # Check if we have any associated class values
                    if token_len == 2 and class_labels:
                        raise IOError("if the classlabel tag is true then class values " "must be supplied")
                    has_class_labels_tag = True
                    class_label_list = [token.strip() for token in tokens[2:]]
                    metadata_started = True
                    classification_dataset = True
                elif line.startswith("@targetlabel"):
                    if data_started:
                        raise IOError("metadata must come before data")
                    tokens = line.split(" ")
                    token_len = len(tokens)
                    if token_len == 1:
                        raise IOError("targetlabel tag requires an associated Boolean value")
                    if tokens[1] == "true":
                        class_labels = True
                    elif tokens[1] == "false":
                        class_labels = False
                    else:
                        raise IOError("invalid targetlabel value")
                    if token_len > 2:
                        raise IOError(
                            "targetlabel tag should not be accompanied with info "
                            "apart from true/false, but found "
                            f"{tokens}"
                        )
                    has_class_labels_tag = True
                    metadata_started = True
                    classification_dataset = False
                # Check if this line contains the start of data
                elif line.startswith("@data"):
                    if line != "@data":
                        raise IOError("data tag should not have an associated value")
                    if data_started and not metadata_started:
                        raise IOError("metadata must come before data")
                    else:
                        has_data_tag = True
                        data_started = True
                # If the 'data tag has been found then metadata has been
                # parsed and data can be loaded
                elif data_started:
                    # Check that a full set of metadata has been provided
                    if (
                        not has_problem_name_tag
                        or not has_timestamps_tag
                        or not has_univariate_tag
                        or not has_class_labels_tag
                        or not has_data_tag
                    ):
                        raise IOError("a full set of metadata has not been provided " "before the data")
                    # Replace any missing values with the value specified
                    line = line.replace("?", replace_missing_vals_with)
                    # Check if we are dealing with data that has timestamps
                    if timestamps:
                        # We're dealing with timestamps so cannot just split
                        # line on ':' as timestamps may contain one
                        has_another_value = False
                        has_another_dimension = False
                        timestamp_for_dim = []
                        values_for_dimension = []
                        this_line_num_dim = 0
                        line_len = len(line)
                        char_num = 0
                        while char_num < line_len:
                            # Move through any spaces
                            while char_num < line_len and str.isspace(line[char_num]):
                                char_num += 1
                            # See if there is any more data to read in or if
                            # we should validate that read thus far
                            if char_num < line_len:
                                # See if we have an empty dimension (i.e. no
                                # values)
                                if line[char_num] == ":":
                                    if len(instance_list) < (this_line_num_dim + 1):
                                        instance_list.append([])
                                    instance_list[this_line_num_dim].append(pd.Series(dtype="object"))
                                    this_line_num_dim += 1
                                    has_another_value = False
                                    has_another_dimension = True
                                    timestamp_for_dim = []
                                    values_for_dimension = []
                                    char_num += 1
                                else:
                                    # Check if we have reached a class label
                                    if line[char_num] != "(" and class_labels:
                                        class_val = line[char_num:].strip()
                                        if classification_dataset:
                                            if class_val not in class_label_list:
                                                raise IOError(
                                                    "the class value '"
                                                    + class_val
                                                    + "' on line "
                                                    + str(line_num + 1)
                                                    + " is not "
                                                    "valid"
                                                )
                                        class_val_list.append(class_val)
                                        char_num = line_len
                                        has_another_value = False
                                        has_another_dimension = False
                                        timestamp_for_dim = []
                                        values_for_dimension = []
                                    else:
                                        # Read in the data contained within
                                        # the next tuple
                                        if line[char_num] != "(" and not class_labels:
                                            raise IOError(
                                                "dimension "
                                                + str(this_line_num_dim + 1)
                                                + " on line "
                                                + str(line_num + 1)
                                                + " does "
                                                "not "
                                                "start "
                                                "with a "
                                                "'('"
                                            )
                                        char_num += 1
                                        tuple_data = ""
                                        while char_num < line_len and line[char_num] != ")":
                                            tuple_data += line[char_num]
                                            char_num += 1
                                        if char_num >= line_len or line[char_num] != ")":
                                            raise IOError(
                                                "dimension "
                                                + str(this_line_num_dim + 1)
                                                + " on line "
                                                + str(line_num + 1)
                                                + " does "
                                                "not end"
                                                " with a "
                                                "')'"
                                            )
                                        # Read in any spaces immediately
                                        # after the current tuple
                                        char_num += 1
                                        while char_num < line_len and str.isspace(line[char_num]):
                                            char_num += 1

                                        # Check if there is another value or
                                        # dimension to process after this tuple
                                        if char_num >= line_len:
                                            has_another_value = False
                                            has_another_dimension = False
                                        elif line[char_num] == ",":
                                            has_another_value = True
                                            has_another_dimension = False
                                        elif line[char_num] == ":":
                                            has_another_value = False
                                            has_another_dimension = True
                                        char_num += 1
                                        # Get the numeric value for the
                                        # tuple by reading from the end of
                                        # the tuple data backwards to the
                                        # last comma
                                        last_comma_index = tuple_data.rfind(",")
                                        if last_comma_index == -1:
                                            raise IOError(
                                                "dimension "
                                                + str(this_line_num_dim + 1)
                                                + " on line "
                                                + str(line_num + 1)
                                                + " contains a tuple that has "
                                                "no comma inside of it"
                                            )
                                        try:
                                            value = tuple_data[last_comma_index + 1 :]
                                            value = float(value)
                                        except ValueError:
                                            raise IOError(
                                                "dimension "
                                                + str(this_line_num_dim + 1)
                                                + " on line "
                                                + str(line_num + 1)
                                                + " contains a tuple that does "
                                                "not have a valid numeric "
                                                "value"
                                            )
                                        # Check the type of timestamp that
                                        # we have
                                        timestamp = tuple_data[0:last_comma_index]
                                        try:
                                            timestamp = int(timestamp)
                                            timestamp_is_int = True
                                            timestamp_is_timestamp = False
                                        except ValueError:
                                            timestamp_is_int = False
                                        if not timestamp_is_int:
                                            try:
                                                timestamp = timestamp.strip()
                                                timestamp_is_timestamp = True
                                            except ValueError:
                                                timestamp_is_timestamp = False
                                        # Make sure that the timestamps in
                                        # the file (not just this dimension
                                        # or case) are consistent
                                        if not timestamp_is_timestamp and not timestamp_is_int:
                                            raise IOError(
                                                "dimension "
                                                + str(this_line_num_dim + 1)
                                                + " on line "
                                                + str(line_num + 1)
                                                + " contains a tuple that "
                                                "has an invalid timestamp '" + timestamp + "'"
                                            )
                                        if (
                                            previous_timestamp_was_int is not None
                                            and previous_timestamp_was_int
                                            and not timestamp_is_int
                                        ):
                                            raise IOError(
                                                "dimension "
                                                + str(this_line_num_dim + 1)
                                                + " on line "
                                                + str(line_num + 1)
                                                + " contains tuples where the "
                                                "timestamp format is "
                                                "inconsistent"
                                            )
                                        if (
                                            prev_timestamp_was_timestamp is not None
                                            and prev_timestamp_was_timestamp
                                            and not timestamp_is_timestamp
                                        ):
                                            raise IOError(
                                                "dimension "
                                                + str(this_line_num_dim + 1)
                                                + " on line "
                                                + str(line_num + 1)
                                                + " contains tuples where the "
                                                "timestamp format is "
                                                "inconsistent"
                                            )
                                        # Store the values
                                        timestamp_for_dim += [timestamp]
                                        values_for_dimension += [value]
                                        #  If this was our first tuple then
                                        #  we store the type of timestamp we
                                        #  had
                                        if prev_timestamp_was_timestamp is None and timestamp_is_timestamp:
                                            prev_timestamp_was_timestamp = True
                                            previous_timestamp_was_int = False

                                        if previous_timestamp_was_int is None and timestamp_is_int:
                                            prev_timestamp_was_timestamp = False
                                            previous_timestamp_was_int = True
                                        # See if we should add the data for
                                        # this dimension
                                        if not has_another_value:
                                            if len(instance_list) < (this_line_num_dim + 1):
                                                instance_list.append([])

                                            if timestamp_is_timestamp:
                                                timestamp_for_dim = pd.DatetimeIndex(timestamp_for_dim)

                                            instance_list[this_line_num_dim].append(
                                                pd.Series(
                                                    index=timestamp_for_dim,
                                                    data=values_for_dimension,
                                                )
                                            )
                                            this_line_num_dim += 1
                                            timestamp_for_dim = []
                                            values_for_dimension = []
                            elif has_another_value:
                                raise IOError(
                                    "dimension " + str(this_line_num_dim + 1) + " on "
                                    "line " + str(line_num + 1) + " ends with a ',' that "
                                    "is not followed by "
                                    "another tuple"
                                )
                            elif has_another_dimension and class_labels:
                                raise IOError(
                                    "dimension " + str(this_line_num_dim + 1) + " on "
                                    "line " + str(line_num + 1) + " ends with a ':' while "
                                    "it should list a class "
                                    "value"
                                )
                            elif has_another_dimension and not class_labels:
                                if len(instance_list) < (this_line_num_dim + 1):
                                    instance_list.append([])
                                instance_list[this_line_num_dim].append(pd.Series(dtype=np.float32))
                                this_line_num_dim += 1
                                num_dimensions = this_line_num_dim
                            # If this is the 1st line of data we have seen
                            # then note the dimensions
                            if not has_another_value and not has_another_dimension:
                                if num_dimensions is None:
                                    num_dimensions = this_line_num_dim
                                if num_dimensions != this_line_num_dim:
                                    raise IOError(
                                        "line " + str(line_num + 1) + " does not have the "
                                        "same number of "
                                        "dimensions as the "
                                        "previous line of "
                                        "data"
                                    )
                        # Check that we are not expecting some more data,
                        # and if not, store that processed above
                        if has_another_value:
                            raise IOError(
                                "dimension "
                                + str(this_line_num_dim + 1)
                                + " on line "
                                + str(line_num + 1)
                                + " ends with a ',' that is "
                                "not followed by another "
                                "tuple"
                            )
                        elif has_another_dimension and class_labels:
                            raise IOError(
                                "dimension "
                                + str(this_line_num_dim + 1)
                                + " on line "
                                + str(line_num + 1)
                                + " ends with a ':' while it "
                                "should list a class value"
                            )
                        elif has_another_dimension and not class_labels:
                            if len(instance_list) < (this_line_num_dim + 1):
                                instance_list.append([])
                            instance_list[this_line_num_dim].append(pd.Series(dtype="object"))
                            this_line_num_dim += 1
                            num_dimensions = this_line_num_dim
                        # If this is the 1st line of data we have seen then
                        # note the dimensions
                        if not has_another_value and num_dimensions != this_line_num_dim:
                            raise IOError(
                                "line " + str(line_num + 1) + " does not have the same "
                                "number of dimensions as the "
                                "previous line of data"
                            )
                        # Check if we should have class values, and if so
                        # that they are contained in those listed in the
                        # metadata
                        if class_labels and len(class_val_list) == 0:
                            raise IOError("the cases have no associated class values")
                    else:
                        dimensions = line.split(":")
                        # If first row then note the number of dimensions (
                        # that must be the same for all cases)
                        if is_first_case:
                            num_dimensions = len(dimensions)
                            if class_labels:
                                num_dimensions -= 1
                            for _dim in range(0, num_dimensions):
                                instance_list.append([])
                            is_first_case = False
                        # See how many dimensions that the case whose data
                        # in represented in this line has
                        this_line_num_dim = len(dimensions)
                        if class_labels:
                            this_line_num_dim -= 1
                        # All dimensions should be included for all series,
                        # even if they are empty
                        if this_line_num_dim != num_dimensions:
                            raise IOError(
                                "inconsistent number of dimensions. "
                                "Expecting " + str(num_dimensions) + " but have read " + str(this_line_num_dim)
                            )
                        # Process the data for each dimension
                        for dim in range(0, num_dimensions):
                            dimension = dimensions[dim].strip()

                            if dimension:
                                data_series = dimension.split(",")
                                data_series = [float(i) for i in data_series]
                                instance_list[dim].append(pd.Series(data_series))
                            else:
                                instance_list[dim].append(pd.Series(dtype="object"))
                        if class_labels:
                            class_val_list.append(dimensions[num_dimensions].strip())
            line_num += 1
    # Check that the file was not empty
    if line_num:
        # Check that the file contained both metadata and data
        if metadata_started and not (
            has_problem_name_tag
            and has_timestamps_tag
            and has_univariate_tag
            and has_class_labels_tag
            and has_data_tag
        ):
            raise IOError("metadata incomplete")

        elif metadata_started and not data_started:
            raise IOError("file contained metadata but no data")

        elif metadata_started and data_started and len(instance_list) == 0:
            raise IOError("file contained metadata but no data")
        # Create a DataFrame from the data parsed above
        data = pd.DataFrame(dtype=np.float32)
        for dim in range(0, num_dimensions):
            data["dim_" + str(dim)] = instance_list[dim]
        # Check if we should return any associated class labels separately
        if class_labels:
            if return_separate_X_and_y:
                return data, np.asarray(class_val_list)
            else:
                data["class_vals"] = pd.Series(class_val_list)
                return data
        else:
            return data
    else:
        raise IOError("empty file")


def get_split_params(
    split_config: Dict[str, Union[float, List[Union[int, float]]]],
    context_length: Optional[int] = None,
) -> Tuple[Dict[str, Dict[str, Union[int, float]]], Dict[str, Callable]]:
    """Get split parameters

    Args:
        split_config ( Dict[str, Union[float, List[Union[int, float]]]]): Dictionary containing keys which
            define the splits. Two options are possible:
            1. Specifiy train, valid, test. Each value consists of a list of length two, indicating
            the boundaries of a split.
            2. Specify train, test. Each value consists of a single floating point number specifying the
            fraction of data to use. Valid is populated using the remaining data.

        context_length (int, optional): Context length, used only when offseting
            the split so predictions can be made for all elements of split. Defaults to None.

    Returns:
        Tuple[Dict[str, Dict[str, Union[int, float]]], Dict[str, Callable]]: Tuple of split parameters
        and split functions to use to split the data.
    """

    split_params = {}
    split_function = {}

    if "valid" in split_config:
        for group in ["train", "test", "valid"]:
            if ((split_config[group][0] < 1) and (split_config[group][0] != 0)) or (split_config[group][1] < 1):
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

    # no valid, assume train/test split
    split_function = train_test_split
    split_params = {
        "train": split_config["train"],
        "test": split_config["test"],
        "valid_test_offset": context_length if context_length else 0,
    }
    return split_params, split_function


def convert_tsf(filename: str) -> pd.DataFrame:
    """Converts a tsf format file into a pandas dataframe.
    Returns the result in canonical multi-time series format, with an ID column, timestamp, and one or more
    value columns. Attemps to map frequency information given in the input file to pandas equivalents.


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
    ) = convert_tsf_to_dataframe(filename, replace_missing_vals_with=np.NaN)

    id_column_name = "id"
    timestamp_column_name = "timestamp"
    value_column_name = "value"

    tsf_to_pandas_freq_map = {
        "daily": "d",
        "hourly": "h",
        "half_hourly": "30min",
        "seconds": "s",
        "minutes": "min",
        "minutely": "min",
        "weekly": "W",
        "monthly": "MS",
        "yearly": "YS",
        "quarterly": "QS",
    }

    if frequency:
        try:
            freq = tsf_to_pandas_freq_map[frequency]
        except KeyError:
            try:
                freq_val, freq_unit = frequency.split("_")
                freq = freq_val + tsf_to_pandas_freq_map[freq_unit]
            except ValueError:
                freq = tsf_to_pandas_freq_map[frequency]
            except KeyError:
                raise ValueError(f"Input file contains an unknown frequency unit {freq_unit}")
    else:
        freq = None

    # determine presence of timestamp column and name
    default_start_timestamp = datetime(1900, 1, 1)
    datetimes = [is_datetime64_dtype(d) for d in loaded_data.dtypes]
    source_timestamp_column = None
    if any(datetimes):
        source_timestamp_column = loaded_data.columns[datetimes][0]

    dfs = []
    for index, item in loaded_data.iterrows():
        if freq and source_timestamp_column:
            timestamps = pd.date_range(item[source_timestamp_column], periods=len(item.series_value), freq=freq)
        elif freq:
            timestamps = pd.date_range(default_start_timestamp, periods=len(item.series_value), freq=freq)
        else:
            timestamps = range(len(item.series_value))

        dfs.append(
            pd.DataFrame(
                {
                    id_column_name: item.series_name,
                    timestamp_column_name: timestamps,
                    value_column_name: item.series_value,
                }
            )
        )

    df = pd.concat(dfs)
    df.reset_index(inplace=True, drop=True)
    return df


def convert_to_univariate(
    data: pd.DataFrame,
    timestamp_column: str,
    id_columns: List[str],
    target_columns: List[str],
    var_name: str = "column_id",
    value_name: str = "value",
) -> pd.DataFrame:
    """Converts a dataframe in canonical format to a univariate dataset. Adds an additional id column to
    indicate the original target column to which a given value corresponds.

    Args:
        data (pd.DataFrame): Input data frame containing multiple target columns.
        timestamp_column (str): String representing the timestamp column.
        id_columns (List[str]): List of columns representing the ids in the data. Use empty list (`[]`) if there
            are no id columns.
        target_columns (List[str]): The target columns in the data.
        var_name (str): Name of new id column used to identify original column name. Defaults to "column_id".
        value_name (str): Name of new value column in the resulting univariate datset. Defaults to "value".

    Returns:
        pd.DataFrame: Converted dataframe.
    """

    if len(target_columns) < 2:
        raise ValueError("`target_columns` should be a non-empty list of two or more elements.")

    return pd.melt(
        data,
        id_vars=[
            timestamp_column,
        ]
        + id_columns,
        value_vars=target_columns,
        var_name=var_name,
        value_name=value_name,
    )


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


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model

    Args:
        model (torch.nn.Module): The model.

    Returns:
        int: Number of parameters requiring gradients.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def convert_tsfile(filename: str) -> pd.DataFrame:
    """Converts a .ts file into a pandas dataframe.
    Returns the result in canonical multi-time series format, with an ID column, and timestamp.

    Args:
        filename (str): Input file name.

    Returns:
        pd.DataFrame: Converted time series


    To do:
    - address renaming of columns
    - check that we catch all timestamp column types

    """

    dfs = []
    df = convert_tsfile_to_dataframe(filename, return_separate_X_and_y=False)

    # rows, columns = df.shape
    value_columns = [c for c in df.columns if c != "class_vals"]

    for row in df.itertuples():
        l = len(row.dim_0)
        temp_df = pd.DataFrame({"id": [row.Index] * l})

        for j, c in enumerate(value_columns):
            c_data = getattr(row, c)
            if isinstance(c_data.index, pd.Timestamp) and "timestamp" not in temp_df.columns:
                ## include timestamp columns if data includes timestamps
                temp_df["timestamp"] = c_data.index
            temp_df[f"value_{j}"] = c_data.values

        temp_df["target"] = row.class_vals

        dfs.append(temp_df)

    final_df = pd.concat(dfs, ignore_index=True)

    # to be moved to a preprocessor
    # ## convert targets to floats or integers
    # ## non-numeric classification labels will be converted to integers as well
    # try:
    #     final_df["target"] = pd.to_numeric(final_df["target"])
    # except KeyError:
    #     string_labels = final_df["target"].unique()
    #     label_to_int_map = {str_label: num for num, str_label in enumerate(string_labels)}
    #     final_df["target"] = final_df["target"].map(label_to_int_map)

    # ## make sure labels are 0 indexed if classification
    # if classification and final_df["target"].min() != 0:
    #     final_df["target"] = final_df["target"] - 1

    return final_df
