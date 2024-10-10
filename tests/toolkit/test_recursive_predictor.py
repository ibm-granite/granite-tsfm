# Copyright contributors to the TSFM project
#

"""Tests recursive prediction functions"""

import tempfile

from transformers import Trainer, TrainingArguments

from tsfm_public.toolkit.dataset import (
    ForecastDFDataset,
)
from tsfm_public.toolkit.recursive_predictor import RecursivePredictor, RecursivePredictorConfig
from tsfm_public.toolkit.util import select_by_index


def get_dataset_for_rolling_prediction(
    df, rolling_prediction_length, target_columns, conditional_columns=[], control_columns=[]
):
    # ROLLING_PREDICTION_LENGTH = 192
    context_length = 512
    timestamp_column = "date"
    id_columns = []
    # target_columns = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]

    data = df.copy()

    test_end_index = 12 * 30 * 24 + 8 * 30 * 24
    test_start_index = test_end_index - context_length - rolling_prediction_length

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
        control_columns=control_columns,
        conditional_columns=conditional_columns,
        prediction_length=rolling_prediction_length,
        context_length=context_length,
    )
    return test_dataset


def test_simple_rolling_prediction(ttm_model, etth_data_base):
    SEED = 42
    rolling_prediction_length = 192
    target_columns = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]

    test_dataset = get_dataset_for_rolling_prediction(
        etth_data_base, rolling_prediction_length=rolling_prediction_length, target_columns=target_columns
    )

    model = ttm_model()
    # base_model_context_length = ttm_model.config.context_length
    base_model_prediction_length = model.config.prediction_length

    rec_config = RecursivePredictorConfig(
        model=model,
        requested_prediction_length=rolling_prediction_length,
        model_prediction_length=base_model_prediction_length,
        loss=model.config.loss,
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

    predictions = out.predictions[0]

    assert predictions.shape[2] == len(target_columns)
    assert predictions.shape[1] == rolling_prediction_length


def test_rolling_prediction_with_exogenous(ttm_model, etth_data_base):
    SEED = 42
    rolling_prediction_length = 192
    control_columns = [
        "HUFL",
        "HULL",
        "MUFL",
        "MULL",
        "LUFL",
        "LULL",
    ]
    target_columns = ["OT"]

    # simulate an exogenous model by setting some parameters
    model = ttm_model(prediction_channel_indices=[0], num_input_channels=len(control_columns) + len(target_columns))

    test_dataset = get_dataset_for_rolling_prediction(
        etth_data_base,
        rolling_prediction_length=rolling_prediction_length,
        target_columns=target_columns,
        control_columns=control_columns,
    )

    # base_model_context_length = ttm_model.config.context_length
    base_model_prediction_length = model.config.prediction_length

    rec_config = RecursivePredictorConfig(
        model=model,
        requested_prediction_length=rolling_prediction_length,
        model_prediction_length=base_model_prediction_length,
        loss=model.config.loss,
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

    predictions = out.predictions[0]

    assert predictions.shape[2] == len(target_columns)
    assert predictions.shape[1] == rolling_prediction_length
