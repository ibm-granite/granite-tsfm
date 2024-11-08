"""Light weight checks on data frame contents to keep Q/A happy."""

from typing import Any, Dict, Tuple

import pandas as pd


def id_cols_are_strings(data: pd.DataFrame, schema: Dict[str, Any]) -> Tuple[int, str]:
    # did user specify id columns
    id_cols = schema["id_columns"] if "id_columns" in schema else None
    if id_cols:
        dtypes = data.dtypes
        for id in id_cols:
            if not isinstance(dtypes[id], object):
                return 1, f"data for identifier column {id} must not be a character or string type"
    return 0, None


def columns_referenced_are_there(data: pd.DataFrame, schema: Dict[str, Any]) -> Tuple[int, str]:
    cols_df = list(data.columns)
    tocheck = ["id_columns", "timestamp_column"]
    for s in tocheck:
        v = schema.get(s, None)
        if v:
            if isinstance(v, list):
                for ch in v:
                    if ch not in cols_df:
                        return 1, f"schema and data mismatch, '{ch}' is not in {','.join(cols_df)}"
            elif isinstance(v, str):
                if v not in cols_df:
                    return 1, f"schema and data mismatch, '{v}' is not in {','.join(cols_df)}"
            else:
                return 1, f"unexpected type {type(v)} given for schema element {s}"
    return 0, None


# append you checks here. these should be as light weight as possible
# avoid copying data, for example.
ALLCHECKS = [columns_referenced_are_there, id_cols_are_strings]


def check(data: pd.DataFrame, schema: Dict[str, Any]) -> Tuple[int, str]:
    if len(data) < 1:
        return 0, None
    for f in ALLCHECKS:
        rc, msg = f(data, schema)
        if rc != 0:
            return rc, msg
    return 0, None
