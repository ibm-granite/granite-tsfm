# %%[markdown]
# ## Channel Independence Patch Time Series Transformer

# %%[markdown]
# Channel Independence PatchTST model - an efficient design of Transformer-based models for multivariate time series forecasting and self-supervised representation learning - is demonstrated in the following diagram. It is based on two key components:
#
#   - segmentation of time series into subseries-level patches which are served as input tokens to Transformer;
#
#   - channel-independence where each channel contains a single univariate time series that shares the same embedding and Transformer weights across all the series.
#
# Patching design naturally has three-fold benefit: local semantic information is retained in the embedding; computation and memory usage of the attention maps are quadratically reduced given the same look-back window; and the model can attend longer history.
#
# Channel independence allows each time series to have its own embedding and attention maps while sharing the same model parameters across different channels.
#
# <div> <img src="./figures/patchTST.png" alt="Drawing" style="width: 600px;"/></div>

# %%
import pandas as pd

from tsfmservices.toolkit.dataset import PretrainDFDataset
from tsfmservices.toolkit.util import select_by_index
from tsfmservices.toolkit.time_series_preprocessor import TimeSeriesPreprocessor

from transformers import (
    PatchTSTConfig,
    PatchTSTForPretraining,
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
forecast_columns = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]

context_length = 512

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

tsp = TimeSeriesPreprocessor(
    timestamp_column=timestamp_column,
    id_columns=id_columns,
    input_columns=forecast_columns,
    output_columns=forecast_columns,
    scaling=True,
)
tsp.train(train_data)

# %%
train_dataset = PretrainDFDataset(
    tsp.preprocess(train_data),
    id_columns=id_columns,
    input_columns=forecast_columns,
    context_length=context_length,
)
eval_dataset = PretrainDFDataset(
    tsp.preprocess(eval_data),
    timestamp_column=timestamp_column,
    input_columns=forecast_columns,
    context_length=context_length,
)


# %%[markdown]
# ## Configure the PatchTST model
#
# The settings below control the different components in the PatchTST model.
#  - num_input_channels: the number of input channels (or dimensions) in the time series data. This is
#    automatically set to the number for forecast columns.
#  - context_length: As described above, the amount of historical data used as input to the model.
#  - patch_length: The length of the patches extracted from the context window (of length `context_length``).
#  - stride: The stride used when extracting patches from the context window.
#  - mask_ratio: The fraction of input patches that are completely masked for the purpose of pretraining the model.
#  - d_model: Dimension of the transformer layers.
#  - encoder_attention_heads: The number of attention heads for each attention layer in the Transformer encoder.
#  - encoder_layers: The number of encoder layers.
#  - encoder_ffn_dim: Dimension of the intermediate (often referred to as feed-forward) layer in the encoder.
#  - dropout: Dropout probability for all fully connected layers in the encoder.
#  - head_dropout: Dropout probability used in the head of the model.
#
# We recommend that you only adjust the values in the next cell.
# %%
patch_length = 12

# %%
config = PatchTSTConfig(
    num_input_channels=len(forecast_columns),
    context_length=context_length,
    patch_length=patch_length,
    stride=patch_length,
    mask_ratio=0.4,
    d_model=128,
    encoder_attention_heads=16,
    encoder_layers=3,
    encoder_ffn_dim=512,
    dropout=0.2,
    head_dropout=0.2,
    pooling=None,
    channel_attention=False,
    scaling="std",
    loss="mse",
    pre_norm=True,
    norm="batchnorm",
)
pretraining_model = PatchTSTForPretraining(config)

# %%[markdown]
# ## Train model
#
# Trains the PatchTST model based on the Mask-based pretraining strategy. We recommend that the user keep the settings
# as they are below, with the exception of:
#  - num_train_epochs: The number of training epochs. This may need to be adjusted to ensure sufficient training.
#

# %%
training_args = TrainingArguments(
    output_dir="./checkpoint/pretrain",
    # logging_steps = 100,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=5,
    logging_strategy="epoch",
    load_best_model_at_end=True,
    max_steps=100,
    label_names=["past_values"],
)
pretrainer = Trainer(
    model=pretraining_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

pretrainer.train()

# %%
pretrainer.save_model("model/pretrained")
# %%
