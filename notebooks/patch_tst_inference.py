# %%[markdown]
# # Channel Independence Patch Time Series Transformer
# Inference (forecasting)


# %%
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from IPython.display import Image

from transformers.models.patchtst import PatchTSTForPrediction


from tsfmservices.toolkit.time_series_preprocessor import TimeSeriesPreprocessor
from tsfmservices.toolkit.time_series_forecasting_pipeline import (
    TimeSeriesForecastingPipeline,
)
from tsfmservices.toolkit.util import select_by_index


# %%[markdown]
# ## Load model and construct forecasting pipeline
#
# Please adjust the following parameters to suit your application:
# - timestamp_column: column name containing timestamp information, use None if there is no such column
# - id_columns: List of column names specifying the IDs of different time series. If no ID column exists, use []
# - forecast_columns: List of columns to be modeled
#
# Using the parameters above load the data, divide it into train and eval portions, and create torch datasets.

# %%

timestamp_column = "date"
id_columns = []
forecast_columns = ["OT"]

model = PatchTSTForPrediction.from_pretrained("model/forecasting")
tsp = TimeSeriesPreprocessor.from_pretrained("preprocessor")
forecast_pipeline = TimeSeriesForecastingPipeline(
    model=model,
    timestamp_column=timestamp_column,
    id_columns=id_columns,
    input_columns=forecast_columns,
)
context_length = model.config.context_length

# %%[markdown]
# ## Load and prepare test dataset
#

# %%
data = pd.read_csv(
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
    parse_dates=[timestamp_column],
)

test_data = select_by_index(
    data,
    id_columns=id_columns,
    start_index=12 * 30 * 24 + 4 * 30 * 24 - context_length,
    end_index=12 * 30 * 24 + 8 * 30 * 24,
)
test_data = tsp.preprocess(test_data)


# %%[markdown]
# ## Generate Forecasts
#
# Note that the ouput will consist of a Pandas dataframe with the following structure.
# If you have specified timestamp and/or ID columns they will be included. The forecast
# columns will be named `{forecast column}_prediction`, for each `{forecast column}` that was
# specified.
# Each forecast column will be a vector of values with length equal to the prediction horizon
# that was specified when the model was trained.

# %%
forecasts = forecast_pipeline(test_data)
forecasts.head()

# %%[markdown]
# ## Evaluate performance
#
# %%
from tsevaluate.multivalue_timeseries_evaluator import CrossTimeSeriesEvaluator


labels_ = forecasts[id_columns + [timestamp_column] + forecast_columns]
forecasts_ = forecasts.drop(columns=forecast_columns)

eval = CrossTimeSeriesEvaluator(
    timestamp_column=timestamp_column,
    prediction_columns=[f"{c}_prediction" for c in forecast_columns],
    label_columns=forecast_columns,
    metrics_spec=["mse", "smape", "rmse", "mae"],
)
eval.evaluate(labels_, forecasts_)

# %%[markdown]
# ## Plot results
#
#
# %%

plot_start = 2850
plot_stride = model.config.prediction_length
periodicity = "1h"
prediction_length = model.config.prediction_length
plot_index = range(plot_start, forecasts_.shape[0], plot_stride)


fig = make_subplots(specs=[[{"secondary_y": True}]])
# for id in id_list:
#     data = train[train.id == id]
#     fig.add_trace(go.Scatter(x=data["date"], y=data["val"], name=f"{id}(train)"))
for c in forecast_columns:
    fig.add_trace(
        go.Scatter(
            x=test_data[timestamp_column].iloc[plot_start:],
            y=test_data[c].iloc[plot_start:],
            name=f"{c}",
        )
    )

for c in forecast_columns:
    forecast_name = f"{c}_prediction"
    for i in plot_index:
        start = forecasts_.loc[i, timestamp_column]

        timestamps = pd.date_range(
            start, freq=periodicity, periods=prediction_length + 1
        )
        timestamp = timestamps[1:]
        forecast_val = forecasts_.loc[i, forecast_name]

        fig.add_trace(go.Scatter(x=timestamps, y=forecast_val, name=forecast_name))

fig["layout"].update(height=600, width=800, title="graph", xaxis=dict(tickangle=-45))
Image(fig.to_image(format="png"))
# %%
