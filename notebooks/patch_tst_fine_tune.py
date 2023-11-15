# %%[markdown]
# # Channel Independence Patch Time Series Transformer
# Fine tuning for forecasting
#
# Maybe add a picture of the PatchTST with forecasting head?

# %%
import pandas as pd

from tsfmservices.toolkit.dataset import ForecastDFDataset
from tsfmservices.toolkit.time_series_preprocessor import TimeSeriesPreprocessor
from tsfmservices.toolkit.util import select_by_index
from transformers import (
    PatchTSTConfig,
    PatchTSTForPrediction,
    Trainer,
    TrainingArguments,
)

# %%[markdown]
# ## Load and prepare datasets
#
# In the next cell, please adjust the following parameters to suit your application:
# - dataset_path: path to local .csv file, or web address to a csv file for the data of interest. Data is loaded with pandas, so anything supported by
#   `pd.read_csv` is supported: (https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html).
# - timestamp_column: column name containing timestamp information, use None if there is no such column
# - id_columns: List of column names specifying the IDs of different time series. If no ID column exists, use []
# - forecast_columns: List of columns to be modeled
# - prediction_length: Specifies how many timepoints should be forecasted
# - context_length: The amount of historical data used as input to the model. Windows of the input time series data with length equal to
#   context_length will be extracted from the input dataframe. In the case of a multi-time series dataset, the context windows will be created
#   so that they are contained within a single time series (i.e., a single ID).
# - train_start_index, train_end_index: the start and end indices in the loaded data which delineate the training data.
# - eval_start_index, eval_end_index: the start and end indices in the loaded data which delineate the evaluation data.
#
# The data is first loaded into a Pandas dataframe and split into training and evaluation parts. Then the pandas dataframes are converted
# to the appropriate torch dataset needed for training.

# %%
dataset_path = (
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
)
timestamp_column = "date"
id_columns = []
forecast_columns = ["OT"]

prediction_length = 24

pretrained_model_path = "model/pretrained"

# load pretrained model config, to access some previously defined parameters
pretrained_config = PatchTSTConfig.from_pretrained(pretrained_model_path)
context_length = (
    pretrained_config.context_length
)  # use pretrained_config.context_length to match pretrained model

train_start_index = None  # None indicates beginning of dataset
train_end_index = 12 * 30 * 24

# we shift the start of the evaluation period back by context length so that
# the first evaluation timestamp is immediately following the training data
eval_start_index = 12 * 30 * 24 - context_length
eval_end_index = 12 * 30 * 24 + 4 * 30 * 24


# %%
data = pd.read_csv(
    dataset_path,
    parse_dates=[timestamp_column],
)

train_data = select_by_index(
    data,
    id_columns=id_columns,
    start_index=train_start_index,
    end_index=train_end_index,
)
eval_data = select_by_index(
    data,
    id_columns=id_columns,
    start_index=eval_start_index,
    end_index=eval_end_index,
)

print(data.head())

# %%
tsp = TimeSeriesPreprocessor(
    timestamp_column=timestamp_column,
    id_columns=id_columns,
    input_columns=forecast_columns,
    output_columns=forecast_columns,
    scaling=True,
)
tsp.train(train_data)
train_dataset = ForecastDFDataset(
    tsp.preprocess(train_data),
    timestamp_column=timestamp_column,
    id_columns=id_columns,
    input_columns=forecast_columns,
    context_length=context_length,
    prediction_length=prediction_length,
)
eval_dataset = ForecastDFDataset(
    tsp.preprocess(eval_data),
    timestamp_column=timestamp_column,
    id_columns=id_columns,
    input_columns=forecast_columns,
    context_length=context_length,
    prediction_length=prediction_length,
)

# %%[markdown]
# ## Configure the PatchTST model
#
# The PatchTSTConfig is created in the next cell. This leverages the configuration that
# is already present in the pretrained model, and adds the parameters necessary for the
# forecasting task. This includes:
# - context_length: As described above, the amount of historical data used as input to the model.
# - num_input_channels: The number of input channels. In this case, it is set equal to the n
#   number of dimensions we intend to forecast.
# - prediction_length: Prediction horizon for the forecasting task, as set above.

# %%
pred_config = PatchTSTConfig.from_pretrained(
    pretrained_model_path,
    context_length=context_length,
    num_input_channels=len(forecast_columns),
    prediction_length=prediction_length,
)

# %%[markdown]
# ## Load model and freeze base model parameters
#
# The follwoing cell loads the pretrained model and then freezes parameters in the base
# model. You will likely see a warning about weights not being initialized from the model
# checkpoint; this message is expected since the forecasting model has a head with weights
# which have not yet been trained.

# %%
forecasting_model = PatchTSTForPrediction.from_pretrained(
    "model/pretrained",
    config=pred_config,
    ignore_mismatched_sizes=True,
)
# This freezes the base model parameters
for param in forecasting_model.base_model.parameters():
    param.requires_grad = False


# %%[markdown]
# ## Fine-tune the model
#
# Fine-tunes the PatchTST model using the pretrained base model loaded above. We recommend that the user keep the settings
# as they are below, with the exception of:
#  - num_train_epochs: The number of training epochs. This may need to be adjusted to ensure sufficient training.
#

# %%
training_args = TrainingArguments(
    output_dir="./checkpoint/forecast",
    # logging_steps = 100,
    # per_device_train_batch_size = 64, #defaults to 8
    # per_device_eval_batch_size = 64, #defaults to 8
    num_train_epochs=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    # eval_steps = 100,
    save_total_limit=5,
    logging_strategy="epoch",
    load_best_model_at_end=True,
    label_names=["future_values"],
)


forecasting_trainer = Trainer(
    model=forecasting_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    # compute_metrics=compute_metrics
)

forecasting_trainer.train()

# %%
forecasting_trainer.save_model("model/forecasting")
tsp.save_pretrained("preprocessor")
# %%
