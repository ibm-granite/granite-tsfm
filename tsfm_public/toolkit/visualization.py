# Copyright contributors to the TSFM project
#
"""Utilities for plotting time series data"""

import logging

import pandas as pd
import plotly.graph_objs as go
from IPython.display import Image
from plotly.subplots import make_subplots


try:
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
    num_predictions: int = None,
    plot_start: int = 0,
    plot_end: int = -1,
    title: str = "Forecasting plot",
    fig_size: tuple = (15, 8),
    plot_type: str = "seaborn",
    return_image: bool = True,
):
    """_summary_

    Args:
        test_data (pd.DataFrame): _description_
        forecast_data (pd.DataFrame): _description_
        forecast_columns (list): _description_
        timestamp_column (str): _description_
        periodicity (str): _description_
        prediction_length (int): _description_
        context_length (int): _description_
        plot_stride (int, optional): _description_. Defaults to None.
        plot_start (int, optional): _description_. Defaults to 0.
        plot_end (int, optional): _description_. Defaults to -1.
        title (str, optional): _description_. Defaults to "Forecasting plot".
        fig_size (tuple, optional): _description_. Defaults to (15, 8).
        plot_type (str, optional): _description_. Defaults to "seaborn".
        return_image (bool, optional): _description_. Defaults to True.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """

    test_data_updated = test_data  # .reset_index()
    # plot true data

    if not HAVE_SEABORN and plot_type == "seaborn":
        raise ValueError(
            "Please install the seaborn package if seaborn plots are needed."
        )

    # if plot_start > len(test_data_updated):
    #     logging.warning(
    #         f"The start index is out of range. "
    #         f"Please choose the start index within the range of the test data to see the true values."
    #     )

    if not periodicity:
        logging.warning("Please specify periodicity.")

    if not context_length:
        logging.warning("Please specify context_length.")

    if not prediction_length:
        logging.warning("Please specify prediction_length.")

    plot_range = range(test_data_updated.shape[0])[plot_start:plot_end]

    # if plot_range[0] < context_length:
    #     logging.warning(
    #         f"To see the plot lines of the prediction and ground truth "
    #         f"choose the plot_start larger than {context_length}."
    #     )

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
        x = test_data_updated[timestamp_column].iloc[plot_start:plot_end]
        y = test_data_updated[c].iloc[plot_start:plot_end]
        plot_line(x=x, y=y, plot_type=plot_type, plot_id=c, fig=fig)

    # plot prediction data

    predictions_start = max(0, plot_start - context_length)
    # predictions_end = max(plot_end - context_length, 0)

    # index into the predictions so that the end of the prediction coincides with the end of the ground truth
    #
    predictions_end = (
        plot_range[-1] - prediction_length - context_length + 1
    )  #  - context_length - prediction_length

    predictions_start = plot_range[0] - context_length

    plot_index = range(predictions_end, predictions_start - 1, -plot_stride)

    if num_predictions and num_predictions < len(plot_index):
        plot_index = plot_index[:num_predictions]

    for c in forecast_columns:
        forecast_name = f"{c}_prediction"
        if plot_type == "plotly":
            for i in plot_index:
                start = forecast_data.iloc[i][timestamp_column]
                timestamps = pd.date_range(
                    start, freq=periodicity, periods=prediction_length + 1
                )
                timestamp = timestamps[1:]
                forecast_val = forecast_data.iloc[i][forecast_name]
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
            height=fig_size[1],
            width=fig_size[0],
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
