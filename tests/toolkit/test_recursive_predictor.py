# Copyright contributors to the TSFM project
#

"""Tests recursive prediction functions"""

import tempfile

import pytest
from transformers import Trainer, TrainingArguments

from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
from tsfm_public.toolkit.dataset import (
    ForecastDFDataset,
)
from tsfm_public.toolkit.recursive_predictor import RecursivePredictor, RecursivePredictorConfig
from tsfm_public.toolkit.util import select_by_index


@pytest.fixture(scope="module")
def ttm_model():
    model_path = "ibm/test-ttm-v1"
    model = TinyTimeMixerForPrediction.from_pretrained(model_path)

    return model


def test_simple_prediction(ttm_model, etth_data_base):
    ROLLING_PREDICTION_LENGTH = 192
    context_length = 512
    SEED = 42
    timestamp_column = "date"
    id_columns = []
    target_columns = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]

    data = etth_data_base.copy()

    test_end_index = 12 * 30 * 24 + 8 * 30 * 24
    test_start_index = test_end_index - context_length - ROLLING_PREDICTION_LENGTH

    test_data = select_by_index(
        data,
        id_columns=id_columns,
        start_index=test_start_index,
        end_index=test_end_index,
    )

    test_dataset = ForecastDFDataset(
        test_data,
        timestamp_column=timestamp_column,
        target_columns=target_columns,
        prediction_length=ROLLING_PREDICTION_LENGTH,
        context_length=context_length,
    )

    # base_model_context_length = ttm_model.config.context_length
    base_model_forecast_length = ttm_model.config.prediction_length

    rec_config = RecursivePredictorConfig(
        model=ttm_model,
        requested_forecast_length=ROLLING_PREDICTION_LENGTH,
        model_forecast_length=base_model_forecast_length,
        loss=ttm_model.config.loss,
    )
    rolling_model = RecursivePredictor(rec_config)

    temp_dir = tempfile.mkdtemp()
    # zeroshot_trainer
    zeroshot_trainer = Trainer(
        model=rolling_model,
        args=TrainingArguments(
            output_dir=temp_dir, per_device_eval_batch_size=32, seed=SEED, label_names=["future_values"]
        ),
    )

    out = zeroshot_trainer.predict(test_dataset)

    predictions = out[0][0]

    assert predictions.shape[1] == ROLLING_PREDICTION_LENGTH
