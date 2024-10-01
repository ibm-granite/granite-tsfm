# Copyright contributors to the TSFM project
#
"""Utilities for the demo app"""

import copy
import logging
import os
from typing import List, Type

import pandas as pd
import streamlit as st
import transformers
from plotly.graph_objs import graph_objs
from transformers import AutoConfig
from tsevaluate.multivalue_timeseries_evaluator import CrossTimeSeriesEvaluator

# Local
from tsfm_public.toolkit.time_series_forecasting_pipeline import (
    TimeSeriesForecastingPipeline,
)
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor
from tsfm_public.toolkit.util import select_by_index
from tsfm_public.toolkit.visualization import plot_ts_forecasting


# A dictionary containing datasets mapped to their location
# Note that keys of this dictionary get rendered in the UI

LOGGER = logging.getLogger(__file__)


@st.cache_data(persist="disk")
def get_performance(metrics: List[str] = None, **kwargs) -> pd.DataFrame:
    """Return a dataframe of performance results

    Args:
        metainfo (dict): _description_

    Returns:
        pd.DataFrame: _description_
    """

    print("perf: ", kwargs)
    # we want to remove some parameters that
    # do not get used within our forecast method
    # because passing them will break the cache from
    # call to call
    # cache_breakers = ["channel"]
    # cachable_args = copy.deepcopy(kwargs)
    # _ = [cachable_args.pop(arg) for arg in cache_breakers]
    timestamp_column = kwargs["timestamp_column"]
    id_columns = kwargs["id_columns"]
    forecast_columns = kwargs["forecast_columns"]

    _, forecasts = forecast(**kwargs)

    labels_ = forecasts[id_columns + [timestamp_column] + forecast_columns]
    forecasts_ = forecasts.drop(columns=forecast_columns)

    tseval = CrossTimeSeriesEvaluator(
        timestamp_column=timestamp_column,
        prediction_columns=[f"{c}_prediction" for c in forecast_columns],
        label_columns=forecast_columns,
        metrics_spec=metrics,
        multioutput="raw",
    )
    df_eval = tseval.evaluate(labels_, forecasts_)

    # df_eval.columns = [c.split("_")[1] for c in df_eval.columns]

    dfm = df_eval.melt()

    dfm["channel"] = dfm["variable"].apply(lambda x: x.split("_")[0].upper())
    dfm["metric"] = dfm["variable"].apply(lambda x: x.split("_")[-1].upper())
    dfm = dfm.pivot(columns=["metric"], index=["channel"], values=["value"])
    dfm.columns = dfm.columns.droplevel(0)

    tseval = CrossTimeSeriesEvaluator(
        timestamp_column=timestamp_column,
        prediction_columns=[f"{c}_prediction" for c in forecast_columns],
        label_columns=forecast_columns,
        metrics_spec=metrics,
        multioutput="uniform_average",
    )
    df_eval = tseval.evaluate(labels_, forecasts_)

    df_eval.columns = [c.split("_")[1].upper() for c in df_eval.columns]
    df_eval.index = ["Average"]

    return pd.concat([dfm, df_eval], axis=0)


@st.cache_data(persist="disk")
def csv_to_df(metainfo: dict) -> pd.DataFrame:
    """Return a pandas dataframe for a given entry in DATASETS metainfo.

    Args:
        metainfo (dict): meta information about the dataset. This is the values in the key->value
        mapping in the DATASETS dictionary.

    Returns:
        pd.DataFrame: _description_
    """
    return pd.read_csv(metainfo["uri"], parse_dates=[metainfo["timestamp_column"]])


def get_model_class(model_path: str) -> Type:
    conf = AutoConfig.from_pretrained(model_path)
    model_class = getattr(transformers, conf.architectures[0])

    return model_class


def get_model_path(**kwargs) -> str:
    if kwargs["approach"] == "zero_shot":
        model_path = os.path.join(kwargs["pretrained_model_path"], "pretrain")
    else:
        model_path = os.path.join(
            kwargs["pretrained_model_path"],
            "transfer",
            kwargs["dataset_name"],
            "model",
            kwargs["approach"],
        )
    return model_path


def get_preprocessor_path(**kwargs) -> str:
    return os.path.join(
        kwargs["pretrained_model_path"],
        "transfer",
        kwargs["dataset_name"],
        "preprocessor",
    )


@st.cache_data(persist="disk")
def forecast(**kwargs) -> pd.DataFrame:
    LOGGER.debug("in forecast with:")
    LOGGER.debug(kwargs)

    timestamp_column = kwargs["timestamp_column"]
    id_columns = kwargs["id_columns"]
    forecast_columns = kwargs["forecast_columns"]
    # finetuned_model_path = kwargs["finetuned_model_path"]
    # pretrained_model_path = kwargs["pretrained_model_path"]

    model_path = get_model_path(**kwargs)
    prep_path = get_preprocessor_path(**kwargs)

    model_class = get_model_class(model_path)
    model = model_class.from_pretrained(model_path, num_input_channels=len(forecast_columns))

    forecast_pipeline = TimeSeriesForecastingPipeline(
        model=model,
        timestamp_column=timestamp_column,
        id_columns=id_columns,
        target_columns=forecast_columns,
    )

    context_length = model.config.context_length
    tsp = TimeSeriesPreprocessor.from_pretrained(prep_path)
    test_start_index = kwargs["test_start_index"] - context_length
    test_end_index = kwargs["test_end_index"]
    data = csv_to_df(kwargs)

    test_data = select_by_index(
        data,
        id_columns=id_columns,
        start_index=test_start_index,
        end_index=test_end_index,
    )
    test_data = tsp.preprocess(test_data)
    # tsp = TimeSeriesPreprocessor.from_pretrained(kwargs["preprocessor_path"])
    forecasts = forecast_pipeline(test_data)
    return test_data, forecasts


def create_figure(**kwargs) -> graph_objs.Figure:
    """Create a figure with with parameters given in kwargs (we do this for maximum implemntation flexibilty).

    At a minimum kwargs should contain the following:

    Returns:
        graph_objs.Figure: The figure that will get displayed in the UI
    """
    # finetuned_model_path = kwargs["finetuned_model_path"]
    forecast_columns = kwargs["forecast_columns"]
    model_path = get_model_path(**kwargs)
    # print("MODEL PATH", model_path)
    timestamp_column = kwargs["timestamp_column"]

    model_class = get_model_class(model_path)

    model = model_class.from_pretrained(model_path, num_input_channels=len(forecast_columns))
    context_length = model.config.context_length
    periodicity = kwargs["periodicity"]
    channel = kwargs["channel"]

    # we want to remove some parameters that
    # do not get used within our forecast method
    # because passing them will break the cache from
    # call to call
    cache_breakers = ["channel"]
    cachable_args = copy.deepcopy(kwargs)
    _ = [cachable_args.pop(arg) for arg in cache_breakers]

    test_data, forecasts = forecast(**cachable_args)

    answer = plot_ts_forecasting(
        test_data,
        forecasts,
        forecast_columns=[channel],
        timestamp_column=timestamp_column,
        periodicity=periodicity,
        prediction_length=model.config.prediction_length,
        context_length=context_length,
        plot_start=0,  # -4 * 24 * 2 - 1,
        plot_end=context_length + model.config.prediction_length * 3,
        plot_stride=model.config.prediction_length,
        num_predictions=3,
        title=channel,
        fig_size=(1600, 200),
        plot_type="plotly",
        return_image=False,
    )

    answer.update_layout(autosize=False, width=500, height=500)
    return answer
