#!/usr/bin/env python
# coding: utf-8

# # TTM-R3 Sample Notebook

# # 🚀 TTM-R3 GIFT Evaluation Notebook
#
# ---
#
# ### 📦 Model
# - **TTM-R3 Model Card**
#   👉 https://huggingface.co/ibm-research/ttm-r3
#
# ---
#
# ### ⚙️ Setup Instructions
# - Follow the setup guide:
#   👉 https://github.com/ibm-granite/granite-tsfm/blob/ttm-r3-release-mq2/notebooks/hfdemo/tinytimemixer/full_benchmarking/gift_leaderboard_ttm_r3_nc/README.md
# - It requires cloning of `granite-tsfm` repo and moving this notebook to `gift_leaderboard_ttm_r3_nc` folder as per the instructions mentioned in the link.
# ---
#
# ### 📝 Notes
# - This notebook runs **GIFT evaluation workflows** for TTM-R3
# - Demonstrates **pre-trained and fine-tuned workflows**
# - Ensure environment setup is completed before execution
#
# ---

# # Importing libraries

# In[1]:


# -------------------------
# Cell 1: Imports
# -------------------------
import csv
import json
import os
import sys
from copy import deepcopy
from pprint import pprint

import pandas as pd
from dotenv import load_dotenv
from gift_eval.data import Dataset
from gluonts.ev.metrics import (
    MAE,
    MAPE,
    MASE,
    MSE,
    MSIS,
    ND,
    NRMSE,
    RMSE,
    SMAPE,
    MeanWeightedSumQuantileLoss,
)
from gluonts.model import evaluate_model
from gluonts.time_feature import get_seasonality


sys.path.append("/dccstor/dnn_forecasting/arindam/FM/velajobs/ttm-r3-release/granite-tsfm")

from notebooks.hfdemo.tinytimemixer.full_benchmarking.gift_leaderboard_ttm_r3_nc.src import (
    ttm_gluonts_predictor,
)
from notebooks.hfdemo.tinytimemixer.full_benchmarking.gift_leaderboard_ttm_r3_nc.src.ttm_gluonts_predictor import (
    TTM_MAX_FORECAST_HORIZON,
    TTMGluonTSPredictor,
)
from notebooks.hfdemo.tinytimemixer.full_benchmarking.gift_leaderboard_ttm_r3_nc.src.utils import (
    delete_if_empty_csv,
    get_args,
    set_seed,
)


print("Loaded predictor from:", ttm_gluonts_predictor.__file__)


# # Checking gift-eval env

# In[2]:


load_dotenv()

print("GIFT_EVAL =", os.getenv("GIFT_EVAL"))


# # Select the workflow (pretrain or finetune)

# In[3]:


# -------------------------
# Cell 3: Config
# -------------------------
RUN_MODE = "pretrain"  # "pretrain" or "finetune"
SEED = 42
OUT_ROOT = "../results"


# # Input args

# In[4]:


def get_args_notebook_safe():
    """
    Call existing get_args() safely inside notebook by temporarily
    replacing sys.argv so Jupyter args do not interfere.
    """
    old_argv = deepcopy(sys.argv)
    try:
        sys.argv = ["notebook"]
        args = get_args()
    finally:
        sys.argv = old_argv

    return args


# -------------------------
# Cell 4: Build args from existing parser defaults + override only what you need
# -------------------------
args = get_args_notebook_safe()

args.upper_bound_fewshot_samples = 1
args.auto_sampler = 1
args.rolling_norm = 1
args.ft_zs_ensemble = 1
args.ft_zs_ensemble_mode = "backtest_mean"
args.ensemble_shrink_lambda = 0.7


# # Configuration

# In[5]:


if RUN_MODE == "pretrain":
    args.ttm_version = "TTM-R3-Pretrained-Test-DEBUG"
    args.num_epochs = 0

elif RUN_MODE == "finetune":
    args.ttm_version = "TTM-R3-Finetuned-Test-DEBUG"
    args.num_epochs = 5
    args.decoder_tune = 0
    args.bias_tune = 1
    args.automate_fewshot_fraction = 1
    args.head_tune = 0
    args.prefix_tune = 0
    args.plot_predictions = 0
    args.quantile_tune = 0
    args.few_shot_data_limit_config = "../gift_leaderboard_ttm_r3_nc/resources/fewshot_data_limit.json"

else:
    raise ValueError(f"Unsupported RUN_MODE: {RUN_MODE}")


OUT_DIR = os.path.join(OUT_ROOT, args.ttm_version)
os.makedirs(OUT_DIR, exist_ok=True)

print("RUN_MODE:", RUN_MODE)
print("ttm_version:", args.ttm_version)
print("num_epochs:", args.num_epochs)
print("OUT_DIR:", OUT_DIR)


# # Choose Datasets

# In[6]:


# short_datasets = "solar/W"
# med_long_datasets = ""

short_datasets = "m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly electricity/15T electricity/H electricity/D electricity/W solar/10T solar/H solar/D solar/W hospital covid_deaths us_births/D us_births/M us_births/W saugeenday/D saugeenday/M saugeenday/W temperature_rain_with_missing kdd_cup_2018_with_missing/H kdd_cup_2018_with_missing/D car_parts_with_missing restaurant hierarchical_sales/D hierarchical_sales/W LOOP_SEATTLE/5T LOOP_SEATTLE/H LOOP_SEATTLE/D SZ_TAXI/15T SZ_TAXI/H M_DENSE/H M_DENSE/D ett1/15T ett1/H ett1/D ett1/W ett2/15T ett2/H ett2/D ett2/W jena_weather/10T jena_weather/H jena_weather/D bitbrains_fast_storage/5T bitbrains_fast_storage/H bitbrains_rnd/5T bitbrains_rnd/H bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"
med_long_datasets = "electricity/15T electricity/H solar/10T solar/H kdd_cup_2018_with_missing/H LOOP_SEATTLE/5T LOOP_SEATTLE/H SZ_TAXI/15T M_DENSE/H ett1/15T ett1/H ett2/15T ett2/H jena_weather/10T jena_weather/H bitbrains_fast_storage/5T bitbrains_rnd/5T bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"


# In[7]:


# -------------------------
# Cell 5: Environment, FS mode, datasets
# -------------------------
set_seed(SEED)
load_dotenv()

FS_MODE = None
if getattr(args, "few_shot_data_limit_config", None):
    with open(args.few_shot_data_limit_config, "r") as f:
        FS_MODE = json.load(f)

with open("../gift_leaderboard_ttm_r3_nc/resources/dataset_properties.json", "r") as f:
    dataset_properties_map = json.load(f)

pretty_names = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}


all_datasets = sorted(set(short_datasets.split() + med_long_datasets.split()))

print("Datasets:", all_datasets)


# # Metrics

# In[8]:


# -------------------------
# Cell 6: Metrics
# -------------------------
metrics = [
    MSE(forecast_type="mean"),
    MSE(forecast_type=0.5),
    MAE(forecast_type="mean"),
    MAE(forecast_type=0.5),
    MASE(),
    MAPE(),
    SMAPE(),
    MSIS(),
    RMSE(),
    NRMSE(),
    ND(),
    MeanWeightedSumQuantileLoss(quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
]


# In[9]:


# -------------------------
# Cell 7: Results CSV setup
# -------------------------
csv_file_path = os.path.join(OUT_DIR, "all_results.csv")
delete_if_empty_csv(csv_file_path)

DONE_DATASETS = []
if os.path.exists(csv_file_path):
    try:
        df_res = pd.read_csv(csv_file_path)
        if "dataset" in df_res.columns:
            DONE_DATASETS = df_res["dataset"].astype(str).tolist()
    except Exception as e:
        print("Warning: could not read existing CSV:", e)

print("Already done datasets:", DONE_DATASETS)

file_exists = os.path.exists(csv_file_path)
csv_file = open(csv_file_path, "a", newline="")
csv_writer = csv.writer(csv_file)

if not file_exists:
    csv_writer.writerow(
        [
            "dataset",
            "model",
            "eval_metrics/MSE[mean]",
            "eval_metrics/MSE[0.5]",
            "eval_metrics/MAE[mean]",
            "eval_metrics/MAE[0.5]",
            "eval_metrics/MASE[0.5]",
            "eval_metrics/MAPE[0.5]",
            "eval_metrics/sMAPE[0.5]",
            "eval_metrics/MSIS",
            "eval_metrics/RMSE[mean]",
            "eval_metrics/NRMSE[mean]",
            "eval_metrics/ND[0.5]",
            "eval_metrics/mean_weighted_sum_quantile_loss",
            "domain",
            "num_variates",
            "horizon",
            "ttm_context_len",
            "available_context_len",
            "finetune_success",
            "finetune_train_num_samples",
            "finetune_valid_num_samples",
            "insample_error",
        ]
    )
    csv_file.flush()

print("Writing to:", csv_file_path)


# # Main Run Loop

# In[10]:


# -------------------------
# Cell 8: Main run loop
# -------------------------
for ds_name in all_datasets:
    set_seed(SEED)

    print("\n" + "=" * 80)
    print(f"Processing dataset: {ds_name}")
    print("=" * 80)

    terms = ["short", "medium", "long"]

    for term in terms:
        if term in ["medium", "long"] and ds_name not in med_long_datasets.split():
            continue

        print(f"\n--- dataset={ds_name}, term={term} ---")

        # -------------------------
        # dataset parsing
        # -------------------------
        if "/" in ds_name:
            ds_key = ds_name.split("/")[0].lower()
            ds_freq = ds_name.split("/")[1]
            ds_key = pretty_names.get(ds_key, ds_key)
        else:
            ds_key = pretty_names.get(ds_name.lower(), ds_name.lower())
            ds_freq = dataset_properties_map[ds_key]["frequency"]

        ds_config = f"{ds_key}/{ds_freq}/{term}"

        if ds_config in DONE_DATASETS:
            print(f"Skipping {ds_config} (already done)")
            continue

        dataset = Dataset(name=ds_name, term=term, to_univariate=False)
        season_length = get_seasonality(dataset.freq)

        print(f"Freq={dataset.freq}, Horizon={dataset.prediction_length}")

        # -------------------------
        # context and channel discovery
        # -------------------------
        all_lengths = []
        num_channels = None
        last_item = None

        for x in dataset.test_data:
            last_item = x
            target = x[0]["target"]

            if len(target.shape) == 1:
                all_lengths.append(len(target))
                num_channels = 1
            else:
                all_lengths.append(target.shape[1])
                num_channels = target.shape[0]

        min_context_length = min(all_lengths)
        print("Min context length:", min_context_length)

        num_prediction_channels = num_channels
        prediction_channel_indices = list(range(num_channels))

        # -------------------------
        # exogenous variables
        # -------------------------
        past_feat_dynamic_real_exist = False
        if args.use_exogs and last_item is not None and "past_feat_dynamic_real" in last_item[0]:
            num_exogs = last_item[0]["past_feat_dynamic_real"].shape[0]
            print(f"Using {num_exogs} past dynamic real exogenous channels")
            num_channels += num_exogs
            past_feat_dynamic_real_exist = True

        if dataset.prediction_length > TTM_MAX_FORECAST_HORIZON:
            prediction_channel_indices = list(range(num_channels))

        # -------------------------
        # predictor
        # -------------------------
        predictor = TTMGluonTSPredictor(
            context_length=min_context_length,
            prediction_length=min(dataset.prediction_length, TTM_MAX_FORECAST_HORIZON),
            model_path="ibm-granite/granite-timeseries-ttm-r3",
            test_data_label=dataset.test_data.label,
            random_seed=SEED,
            term=term,
            ds_name=ds_name,
            out_dir=OUT_DIR,
            scale=True,
            upper_bound_fewshot_samples=args.upper_bound_fewshot_samples,
            force_short_context=(args.force_short_context if term == "short" else False),
            min_context_mult=args.min_context_mult,
            past_feat_dynamic_real_exist=past_feat_dynamic_real_exist,
            num_prediction_channels=num_prediction_channels,
            freq=dataset.freq,
            use_valid_from_train=args.use_valid_from_train,
            insample_forecast=args.insample_forecast,
            insample_use_train=args.insample_use_train,
            head_dropout=args.head_dropout,
            decoder_mode=args.decoder_mode,
            num_input_channels=num_channels,
            huber_delta=args.huber_delta,
            quantile=args.quantile,
            loss=args.loss,
            prediction_channel_indices=prediction_channel_indices,
            use_mask=True,
            force_zeroshot=(args.num_epochs == 0),
            auto_sampler=args.auto_sampler,
            fs_mode_dict=FS_MODE,
            use_lite=args.use_lite,
            plot_predictions=args.plot_predictions,
            rolling_norm=args.rolling_norm,
            ft_zs_ensemble=args.ft_zs_ensemble,
            ft_zs_ensemble_mode=args.ft_zs_ensemble_mode,
            ensemble_shrink_lambda=args.ensemble_shrink_lambda,
            use_zs_anchor=args.use_zs_anchor,
            zs_anchor_weight=args.zs_anchor_weight,
            disable_extra_point_weightage=args.disable_extra_point_weightage,
        )

        # -------------------------
        # training
        # -------------------------
        batch_size = args.batch_size
        optimize_batch_size = batch_size is None

        finetune_success = False
        finetune_train_num_samples = 0
        finetune_valid_num_samples = 0

        try:
            if dataset.prediction_length > TTM_MAX_FORECAST_HORIZON:
                predictor.prediction_length = dataset.prediction_length

            predictor.train(
                train_dataset=dataset.training_dataset,
                valid_dataset=dataset.validation_dataset,
                batch_size=batch_size,
                optimize_batch_size=optimize_batch_size,
                learning_rate=args.learning_rate,
                num_epochs=args.num_epochs,
                fewshot_fraction=args.fewshot_fraction,
                fewshot_location=args.fewshot_location,
                automate_fewshot_fraction=args.automate_fewshot_fraction,
                automate_fewshot_fraction_threshold=args.automate_fewshot_fraction_threshold,
                fewshot_max_samples=args.fewshot_max_samples,
                valid_max_samples=args.valid_max_samples,
                bias_tune=args.bias_tune,
                norm_tune=args.norm_tune,
                patch_tune=args.patch_tune,
                backbone_tune=args.backbone_tune,
                prefix_tune=args.prefix_tune,
                decoder_tune=args.decoder_tune,
                head_tune=args.head_tune,
                quantile_tune=args.quantile_tune,
                enable_staging=args.enable_staging,
            )
            finetune_success = True
            finetune_train_num_samples = getattr(predictor, "train_num_samples", 0)
            finetune_valid_num_samples = getattr(predictor, "valid_num_samples", 0)

        except Exception as e:
            print("Training failed. Falling back to evaluation:", e)
            finetune_success = False

        # -------------------------
        # evaluation
        # -------------------------
        res = evaluate_model(
            predictor,
            test_data=dataset.test_data,
            metrics=metrics,
            batch_size=batch_size,
            axis=None,
            mask_invalid_label=True,
            allow_nan_forecast=False,
            seasonality=season_length,
        )

        insample_error = 0

        result = [
            ds_config,
            args.ttm_version,
            res["MSE[mean]"].iloc[0],
            res["MSE[0.5]"].iloc[0],
            res["MAE[mean]"].iloc[0],
            res["MAE[0.5]"].iloc[0],
            res["MASE[0.5]"].iloc[0],
            res["MAPE[0.5]"].iloc[0],
            res["sMAPE[0.5]"].iloc[0],
            res["MSIS"].iloc[0],
            res["RMSE[mean]"].iloc[0],
            res["NRMSE[mean]"].iloc[0],
            res["ND[0.5]"].iloc[0],
            res["mean_weighted_sum_quantile_loss"].iloc[0],
            dataset_properties_map[ds_key]["domain"],
            dataset_properties_map[ds_key]["num_variates"],
            dataset.prediction_length,
            predictor.ttm.config.context_length,
            min_context_length,
            finetune_success,
            finetune_train_num_samples,
            finetune_valid_num_samples,
            insample_error,
        ]

        csv_writer.writerow(result)
        csv_file.flush()

        DONE_DATASETS.append(ds_config)
        print(f"Saved result for {ds_config}")


# In[11]:


# Cell 9: Close CSV
# -------------------------
csv_file.close()
print("CSV closed:", csv_file_path)


# # Results

# In[12]:


# -------------------------
# Cell 10: Read results
# -------------------------
df = pd.read_csv(csv_file_path)
df = df.sort_values(by="dataset")

display_cols = [
    "dataset",
    "model",
    "eval_metrics/MASE[0.5]",
    "eval_metrics/NRMSE[mean]",
    "eval_metrics/mean_weighted_sum_quantile_loss",
]

pprint(df[display_cols].to_dict(orient="records"))


# In[13]:


print(df[display_cols])


# In[ ]:
