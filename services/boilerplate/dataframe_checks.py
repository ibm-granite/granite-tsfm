"""Light weight checks on data frame contents to keep Q/A happy."""

import itertools
from typing import Any, Dict, Tuple

import pandas as pd


SCHEMA_KEYS = [
    "id_columns",
    "timestamp_column",
    "target_columns",
    "observable_columns",
    "control_columns",
    "conditional_columns",
    "static_categorical_columns",
]  # todo can we dynamically generated this from inference_payloads?


def disjoint_sets(data: pd.DataFrame, schema: Dict[str, Any]) -> Tuple[int, str]:
    #
    items = []
    for ent in SCHEMA_KEYS:
        it = schema[ent] if ent in schema else None
        if isinstance(it, str):
            it = [it]
        if it:
            items.append(it)
    all_combinations = []
    for r in range(1, len(items) + 1):
        combinations = itertools.combinations(items, r)
        all_combinations.extend(combinations)

    # make sure that we form empty sets
    for combo in all_combinations:
        if len(combo) < 2:
            continue
        s = set(combo[0])
        for it in combo[1:]:
            if len(set(it).intersection(s)) > 0:
                return 1, f"you may not specify f{it} whilst it also appears in {s}"
            s = s.union(set(it))

    return 0, None


def id_cols_are_int_or_string_types(data: pd.DataFrame, schema: Dict[str, Any]) -> Tuple[int, str]:
    # did user specify id columns
    id_cols = schema["id_columns"] if "id_columns" in schema else None
    if id_cols:
        dtypes = data.dtypes
        for id in id_cols:
            if not (pd.api.types.is_string_dtype(dtypes[id]) or pd.api.types.is_integer_dtype(dtypes[id])):
                return 1, f"data for identifier column {id} must be a string or integer type."
    return 0, None


def columns_referenced_are_there(data: pd.DataFrame, schema: Dict[str, Any]) -> Tuple[int, str]:
    cols_df = list(data.columns)

    # freq
    for s in SCHEMA_KEYS:
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
ALLCHECKS = [columns_referenced_are_there, id_cols_are_int_or_string_types, disjoint_sets]


def check(data: pd.DataFrame, schema: Dict[str, Any]) -> Tuple[int, str]:
    if len(data) < 1:
        return 0, None
    for f in ALLCHECKS:
        rc, msg = f(data, schema)
        if rc != 0:
            return rc, msg
    return 0, None
