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
from transformers import (
    PatchTSTConfig,
    PatchTSTForMaskPretraining,
    Trainer,
    TrainingArguments,
)


# %%[markdown]
# ## Load and prepare datasets
#
# For the dataset of interest, you must supply the names of the following columns:
#
# - timestamp_column: column name containing timestamp information, use None if there is no such column
# - id_columns: List of column names specifying the IDs of different time series. If no ID column exists, use []
# - forecast_columns: List of columns to be modeled
#
# The data is first loaded into a Pandas dataframe and then converted the appropriate torch dataset needed
# for training. Finaly, the data is split into training and evaluation datasets.

# %%
timestamp_column = "date"
id_columns = []
forecast_columns = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]

context_length = 16  # 512

data = pd.read_csv(
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
    parse_dates=[timestamp_column],
)
print(data.head())

# to do: split data
# need utility here, group sensitive splitting should be done
train_data = data.iloc[: 12 * 30 * 24,].copy()
eval_data = data.iloc[
    12 * 30 * 24 - context_length : 12 * 30 * 24 + 4 * 30 * 24,
].copy()


train_dataset = PretrainDFDataset(
    train_data,
    timestamp_column=timestamp_column,
    id_columns=id_columns,
    input_columns=forecast_columns,
    context_length=context_length,
)
eval_dataset = PretrainDFDataset(
    eval_data,
    timestamp_column=timestamp_column,
    id_columns=id_columns,
    input_columns=forecast_columns,
    context_length=context_length,
)

# %%[markdown]
# ## Configure the PatchTST model
#
# Describe parameters here.

# %%
config = PatchTSTConfig(
    num_input_channels=len(forecast_columns),
    context_length=context_length,
    patch_length=12,
    stride=12,
    mask_ratio=0.4,
    # standardscale=None,  # 'bysample'
    d_model=16,  # 128,
    encoder_attention_heads=2,  # 16,
    encoder_layers=2,  # 6,
    encoder_ffn_dim=16,  # 512,
    dropout=0.2,
    head_dropout=0.2,
)
pretraining_model = PatchTSTForMaskPretraining(config)

# %%[markdown]
# ## Train model

# %%
training_args = TrainingArguments(
    output_dir="./checkpoint/pretrain",
    # logging_steps = 100,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    # eval_steps = 100,
    save_total_limit=5,
    logging_strategy="epoch",
    load_best_model_at_end=True,
    max_steps=10,
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
