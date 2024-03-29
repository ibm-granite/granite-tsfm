# %%
import pandas as pd

from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor


split_config = {"train": [0, 8640], "valid": [8640, 11520], "test": [11520, 14400]}


dataset_path = (
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
)

timestamp_column = "date"

df = pd.read_csv(
    dataset_path,
    parse_dates=[timestamp_column],
)

tsp = TimeSeriesPreprocessor(
    id_columns=[],
    timestamp_column=timestamp_column,
    target_columns=["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"],
    observable_columns=[],
    control_columns=[],
    conditional_columns=[],
    static_categorical_columns=[],
    scaling=True,
    scaler_type="standard",
    encode_categorical=False,
    prediction_length=10,
    context_length=96,
)

train, valid, test = tsp.get_datasets(df, split_config)

# %%
split_config = {"train": [0, 0.7], "valid": [0.7, 0.9], "test": [0.9, 1]}

train, valid, test = tsp.get_datasets(df, split_config)


# %%

df = pd.read_csv("/Users/wmgifford/Downloads/weather.csv", parse_dates=["date"])

tsp = TimeSeriesPreprocessor(
    timestamp_column="date",
    id_columns=[],
    target_columns=[],
    prediction_length=96,
    context_length=512,
)

a, b, c = tsp.get_datasets(
    df,
    split_config={
        "train": 0.7,
        "test": 0.2,
    },
)
