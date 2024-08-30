# Copyright contributors to the TSFM project
#
"""Utilities for plotting time series data"""

import logging
import os
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import torch
from IPython.display import Image
from plotly.subplots import make_subplots
from torch.utils.data.dataset import Dataset
from transformers import PreTrainedModel

from .time_series_preprocessor import create_timestamps


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
        raise ValueError("Please install the seaborn package if seaborn plots are needed.")

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
    predictions_end = plot_range[-1] - prediction_length - context_length + 1  #  - context_length - prediction_length

    predictions_start = plot_range[0] - context_length

    plot_index = range(predictions_end, predictions_start - 1, -plot_stride)

    if num_predictions and num_predictions < len(plot_index):
        plot_index = plot_index[:num_predictions]

    for c in forecast_columns:
        forecast_name = f"{c}_prediction"
        if plot_type == "plotly":
            for i in plot_index:
                start = forecast_data.iloc[i][timestamp_column]
                timestamps = pd.date_range(start, freq=periodicity, periods=prediction_length + 1)
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
            xaxis={"tickangle": -45},
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


def plot_predictions(
    input_df: Optional[pd.DataFrame] = None,
    predictions_df: Optional[pd.DataFrame] = None,
    exploded_predictions_df: Optional[pd.DataFrame] = None,
    dset: Optional[Dataset] = None,
    model: Optional[PreTrainedModel] = None,
    freq: Optional[str] = None,
    timestamp_column: Optional[str] = None,
    id_columns: Optional[List[str]] = None,
    plot_context: Optional[None] = None,
    plot_dir: str = None,
    num_plots: int = 10,
    plot_prefix: str = "valid",
    channel: Union[int, str] = None,
    indices: List[int] = None,
):
    """Utility for plotting forecasts along with context and test data.

    Args:
        input_df: The input dataframe from which the predictions are generated, containing timestamp and target columns.
        predictions_df: The predictions dataframe, where each row contains starting timestamp and a list of predictions for each target column.
        exploded_predictions_df: The predictions dataframe, containing timestamp and predicted target columns.
        dset: Dataset.
        context_df: Context dataframe, containing timestamp and target columns.
        model: The pre-trained TimeseriesModel.
        freq: Frequency of the time series data
        timestamp_column: Name of timestamp column in the dataframe.
        id_columns: List of id columns in the dataframe.
        plot_context: If True, plot context data along with forecasts.
        plot_dir: Directory where plots are saved.
        num_plots: Number of subplots to plot in the figure.
        plot_prefix: Prefix to put on the plot file names.
        channel: Channel (target column or its index) to plot.
        indices: List of indices to plot.
    """
    if indices is not None:
        num_plots = len(indices)

    # possible operations:
    if input_df is not None and exploded_predictions_df is not None:
        # 1) This is a zero-shot prediction, so no test data. We have context data for the channel (target column).
        # We expect the context and predictions to contain the channel
        pchannel = f"{channel}_prediction"
        if pchannel not in exploded_predictions_df.columns:
            raise ValueError(f"Predictions dataframe does not contain target column '{pchannel}'.")
        if channel not in input_df.columns:
            raise ValueError(f"Context dataframe does not contain target column '{channel}'.")

        num_plots = 1
        prediction_length = len(exploded_predictions_df)
        plot_context = len(input_df)
        using_pipeline = True
        plot_test_data = False
    elif input_df is not None and predictions_df is not None:
        # 2) input_df and predictions plus column information is provided

        if indices is None:
            l = len(predictions_df)
            indices = np.random.choice(l, size=num_plots, replace=False)
        predictions_subset = [predictions_df.iloc[i] for i in indices]

        gt_df = input_df.copy()
        gt_df = gt_df.set_index(timestamp_column)  # add id column logic here

        prediction_length = len(predictions_subset[0][channel])
        using_pipeline = True
        plot_test_data = True
    elif model is not None and dset is not None:
        # 3) model and dataset are provided
        device = model.device

        with torch.no_grad():
            if indices is None:
                indices = np.random.choice(len(dset), size=num_plots, replace=False)
            random_samples = torch.stack([dset[i]["past_values"] for i in indices]).to(device=device)

            output = model(random_samples)
            predictions_subset = output.prediction_outputs[:, :, channel].squeeze().cpu().numpy()
            prediction_length = predictions_subset.shape[1]
        using_pipeline = False
        plot_test_data = True
    else:
        raise RuntimeError(
            "You must provide either input_df and predictions_df, or dset and model, or input_df and exploded_predictions_df."
        )

    if plot_context is None:
        plot_context = 2 * prediction_length

    # Set a more beautiful style
    plt.style.use("seaborn-v0_8-whitegrid")

    # Adjust figure size and subplot spacing
    assert num_plots >= 1
    fig, axs = plt.subplots(num_plots, 1, figsize=(10, 2 * num_plots))
    if num_plots == 1:
        axs = [axs]

    for i, index in enumerate(indices):
        if using_pipeline and plot_test_data:
            ts_y_hat = create_timestamps(predictions_subset[i][timestamp_column], freq=freq, periods=prediction_length)
            y_hat = (
                predictions_subset[i][f"{channel}_prediction"]
                if f"{channel}_prediction" in predictions_subset[i]
                else predictions_subset[i][channel]
            )

            # get ground truth
            loc = gt_df.index.get_loc(predictions_subset[i][timestamp_column])
            ts_index = gt_df.index[loc - plot_context + 1 : loc + 1 + prediction_length]
            y = gt_df.loc[ts_index][channel]
            ts_y = y.index
            y = y.values
            border = ts_y[-prediction_length]
            plot_title = f"Example {indices[i]}"

        elif using_pipeline:
            ts_y_hat = create_timestamps(
                exploded_predictions_df[timestamp_column].iloc[0], freq=freq, periods=prediction_length
            )
            y_hat = exploded_predictions_df[f"{channel}_prediction"]

            # get context
            # ts_y = create_timestamps(context_df[timestamp_column].iloc[0], freq=freq, periods=len(context_df))
            ts_y = input_df[timestamp_column].values
            y = input_df[channel].values
            border = None
            plot_title = f"Forecast for {channel}"

        else:
            batch = dset[index]
            ts_y_hat = np.arange(plot_context, plot_context + prediction_length)
            y_hat = predictions_subset[i]

            ts_y = np.arange(plot_context + prediction_length)
            y = batch["future_values"][:, channel].squeeze().numpy()
            x = batch["past_values"][-plot_context:, channel].squeeze().numpy()
            y = np.concatenate((x, y), axis=0)
            border = plot_context
            plot_title = f"Example {indices[i]}"

        # Plot predicted values with a dashed line
        axs[i].plot(ts_y_hat, y_hat, label="Predicted", linestyle="--", color="orange", linewidth=2)

        # Plot true values with a solid line
        axs[i].plot(ts_y, y, label="True", linestyle="-", color="blue", linewidth=2)

        # Plot horizon border
        if border is not None:
            axs[i].axvline(x=border, color="r", linestyle="-")

        axs[i].set_title(plot_title)
        axs[i].legend()

    # Adjust overall layout
    plt.tight_layout()

    # Save the plot
    if plot_dir is not None:
        plot_filename = f"{plot_prefix}_ch_{str(channel)}.pdf"
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, plot_filename))
