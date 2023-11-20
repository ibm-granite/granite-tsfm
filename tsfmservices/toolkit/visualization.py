"""Utilities for plotting time series data"""

# Standard
from datetime import timedelta
import logging

# Third Party
from IPython.display import Image
from plotly.subplots import make_subplots
import pandas as pd
import plotly.graph_objs as go

try:
    # Third Party
    import matplotlib.pyplot as plt
    import seaborn as sns

    HAVE_SEABORN = True
except ImportError:
    HAVE_SEABORN = False


def explode_and_add_timestamps(df, value_column, timestamp_column, time_unit="h"):
    # create df with all time predictions
    df_exploded = df.explode(value_column, ignore_index=True)
    df_exploded["periods"] = df_exploded.groupby(timestamp_column).cumcount()
    df_exploded["offset"] = pd.to_timedelta(df_exploded["periods"], unit=time_unit)
    df_exploded["new_timestamp"] = df_exploded[timestamp_column] + df_exploded["offset"]
    df_exploded = df_exploded.drop(columns=["periods", "offset"])
    return df_exploded


def plot_line(x, y, plot_type: str, plot_id: str, data=None, fig=None, **kwargs):
    # plot data lines based on plot type
    if plot_type == "plotly":
        fig.add_trace(go.Scatter(x=x, y=y, name=plot_id, **kwargs))
    else:
        sns.lineplot(data=data, x=x, y=y, label=plot_id, **kwargs)


def plot_ts_forecasting(
    test_data: pd.DataFrame,
    forecast_data: pd.DataFrame,
    forecast_columns: list,
    timestamp_column: str,
    periodicity: str,
    prediction_length: int,
    context_length: int,
    plot_stride: int = None,
    plot_start: int = 0,
    plot_end: int = -1,
    title: str = "Forecasting plot",
    fig_size: tuple = (15, 8),
    plot_type: str = "seaborn",
    return_image: bool = True,
):
    test_data_updated = test_data.reset_index()
    # plot true data

    if not HAVE_SEABORN and plot_type == "seaborn":
        raise ValueError(
            "Please install the seaborn package if seaborn plots are needed."
        )

    if plot_start > len(test_data_updated):
        logging.warning(
            f"The start index is out of range. "
            f"Please choose the start index within the range of the test data to see the true values."
        )

    if not periodicity:
        logging.warning(f"Please specify periodicity.")

    if not context_length:
        logging.warning(f"Please specify context_length.")

    if not prediction_length:
        logging.warning(f"Please specify prediction_length.")

    if plot_end < context_length:
        logging.warning(
            f"To see the plot lines of the prediction and true values, "
            f"choose the plot_end larger than {context_length}."
        )

    # plot_end = min(plot_end, len(test_data_updated))
    # plot_start = max(0, plot_start)

    if plot_stride is None:
        plot_stride = prediction_length

    # if plot_start > context_length:
    #     logging.warning(
    #         f"To see the plot lines of the prediction and true values, "
    #         f"choose the plot_start less than {context_length}."
    #     )

    if plot_type == "plotly":
        fig = make_subplots(specs=[[{"secondary_y": True}]])
    else:
        fig = plt.figure(figsize=(fig_size[0], fig_size[1]))
    for c in forecast_columns:
        x = test_data_updated[timestamp_column].loc[plot_start:plot_end]
        y = test_data_updated[c].loc[plot_start:plot_end]
        plot_line(x=x, y=y, plot_type=plot_type, plot_id=c, fig=fig)

    # plot prediction data

    predictions_start = max(0, plot_start - context_length)
    # predictions_end = max(plot_end - context_length, 0)

    predictions_end = plot_end - context_length - prediction_length

    plot_index = range(predictions_start, predictions_end, plot_stride)
    # print(test_data_updated[timestamp_column].loc[plot_start:])
    # print(forecast_data.loc[plot_start, timestamp_column])
    for c in forecast_columns:
        forecast_name = f"{c}_prediction"
        if plot_type == "plotly":
            for i in plot_index:
                start = forecast_data.loc[i, timestamp_column]
                timestamps = pd.date_range(
                    start, freq=periodicity, periods=prediction_length + 1
                )
                timestamp = timestamps[1:]
                forecast_val = forecast_data.loc[i, forecast_name]
                plot_line(
                    x=timestamp,
                    y=forecast_val,
                    plot_type=plot_type,
                    plot_id=forecast_name,
                    fig=fig,
                )
        else:
            test_forecast = explode_and_add_timestamps(
                forecast_data[[timestamp_column, forecast_name]],
                forecast_name,
                timestamp_column,
                periodicity,
            )
            plot_line(
                data=test_forecast,
                x="new_timestamp",
                y=forecast_name,
                plot_type=plot_type,
                plot_id=forecast_name,
                estimator="median",
                errorbar=("ci", 95),
            )

    # final plot formatting
    if plot_type == "plotly":
        fig["layout"].update(
            height=fig_size[1] * 100,
            width=fig_size[0] * 100,
            title=title,
            xaxis=dict(tickangle=-45),
        )
        if return_image:
            return Image(fig.to_image(format="png"))
        else:
            return fig
    else:
        plt.title(title)
        plt.xticks(rotation=45)
        plt.close()
        return fig
