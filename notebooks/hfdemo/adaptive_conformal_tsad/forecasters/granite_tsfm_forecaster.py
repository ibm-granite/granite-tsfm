# Copyright contributors to the TSFM project
#
import numpy as np
import torch

from tsfm_public import FlowStateForPrediction
from tsfm_public.toolkit import get_model
from tsfm_public.toolkit.time_series_forecasting_pipeline import (
    TimeSeriesForecastingPipeline,
)
from tsfm_public.toolkit.time_series_preprocessor import (
    TimeSeriesPreprocessor,
)


GRANITE_TSFM_CHECKPOINTS = {
    "ttm": "ibm-granite/granite-timeseries-ttm-r2",
    "flowstate": "ibm-granite/granite-timeseries-flowstate-r1",
}


def granite_tsfm_forecaster(
    df,
    timestamp_column,
    target_columns,
    model_name="ttm",
    model_checkpoint=None,
    context_length=50,
    prediction_length=15,
    fixed_context=False,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_checkpoint is None:
        model_checkpoint = GRANITE_TSFM_CHECKPOINTS[model_name]
    if model_name == "ttm":
        model = get_model(
            model_checkpoint,
            context_length=context_length,
            prediction_length=prediction_length,
        )
    if model_name == "flowstate":
        model = FlowStateForPrediction.from_pretrained(model_checkpoint)

    ## TimeSeriesPreprocessor
    tsp = TimeSeriesPreprocessor(
        timestamp_column=timestamp_column,
        target_columns=target_columns,
        prediction_length=prediction_length,
        context_length=context_length,
        scaling=False,
    )
    tsp.train(df)
    ## Forecast Pipeline
    fpipe = TimeSeriesForecastingPipeline(model, feature_extractor=tsp, device=device)
    forecast = fpipe(df)
    y_pred = np.array(
        [np.stack(z) for z in forecast[[f"{target_column}_prediction" for target_column in target_columns]].values]
    ).transpose(0, 2, 1)
    y_true = np.array([np.stack(z) for z in forecast[list(target_columns)].values]).transpose(0, 2, 1)

    return {"y_pred": y_pred, "y_true": y_true[:, : y_pred.shape[1], :]}
