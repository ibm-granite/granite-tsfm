# %%[markdown]
# # Channel Independence Patch Time Series Transformer
# Fine tuning for forecasting
#
# Maybe add a picture of the PatchTST with forecasting head?

# %%
import pandas as pd

from tsfmservices.toolkit.dataset import ForecastDFDataset
from transformers import (
    PatchTSTConfig,
    PatchTSTForForecasting,
    Trainer,
    TrainingArguments,
)


# %%[markdown]
# ## Load and prepare datasets
#
# Please adjust the following parameters to suit your application:
# - timestamp_column: column name containing timestamp information, use None if there is no such column
# - id_columns: List of column names specifying the IDs of different time series. If no ID column exists, use []
# - forecast_columns: List of columns to be modeled
# - context_length: Specifies how many historical time points are used by the model
# - prediction_length: Specifies how many timepoints should be forecasted
#
# Using the parameters above load the data, divide it into train and eval portions, and create torch datasets.

# %%
timestamp_column = "date"
id_columns = []
forecast_columns = ["OT"]

data = pd.read_csv(
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
    parse_dates=[timestamp_column],
)
print(data.head())

pretrained_model_path = "model/pretrained"
pretrained_config = PatchTSTConfig.from_pretrained(pretrained_model_path)

prediction_length = 20
context_length = 32  # use pretrained_config.context_length to match pretrained model

# to do: split data
# need utility here, group sensitive splitting should be done
train_data = data.iloc[: 12 * 30 * 24,].copy()
eval_data = data.iloc[
    12 * 30 * 24 - context_length : 12 * 30 * 24 + 4 * 30 * 24,
].copy()


train_dataset = ForecastDFDataset(
    train_data,
    timestamp_column=timestamp_column,
    id_columns=id_columns,
    input_columns=forecast_columns,
    context_length=context_length,
    prediction_length=prediction_length,
)
eval_dataset = ForecastDFDataset(
    eval_data,
    timestamp_column=timestamp_column,
    id_columns=id_columns,
    input_columns=forecast_columns,
    context_length=context_length,
    prediction_length=prediction_length,
)

# %%[markdown]
# ## Configure the PatchTST model
#
# Describe only forecasting specific parameters that are configurable here.

# %%
pred_config = PatchTSTConfig.from_pretrained(
    pretrained_model_path,
    context_length=context_length,
    num_input_channels=len(forecast_columns),
    prediction_length=prediction_length,
)

# %%[markdown]
# ## Load model and freeze base model parameters

# %%
forecasting_model = PatchTSTForForecasting.from_pretrained(
    "model/pretrained",
    config=pred_config,
    ignore_mismatched_sizes=True,
)
# to unfreeze the base model parameters, comment out the cell
for param in forecasting_model.base_model.parameters():
    param.requires_grad = False


# %%[markdown]
# ## Train model
# Provide description of important training parameters.

# %%
training_args = TrainingArguments(
    output_dir="./checkpoint/forecast",
    # logging_steps = 100,
    # per_device_train_batch_size = 64, #defaults to 8
    # per_device_eval_batch_size = 64, #defaults to 8
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    # eval_steps = 100,
    save_total_limit=5,
    logging_strategy="epoch",
    load_best_model_at_end=True,
    max_steps=10,  # For a quick test
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
# ## Inference
#
# To do: use pipeline code to produce more friendly output
import torch, copy

device = forecasting_model.device


data_sample = copy.copy(eval_dataset[0])
data_sample["past_values"] = torch.unsqueeze(data_sample["past_values"], 0)
data_sample["future_values"] = torch.unsqueeze(data_sample["future_values"], 0)
forecasting_model(
    data_sample["past_values"].to(device),
    future_values=data_sample["future_values"].to(device),
)

# %%
