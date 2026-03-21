import csv
import json
import os
from pprint import pprint

from dotenv import load_dotenv
from gift_eval.data import Dataset
from gluonts.ev.metrics import (
    MAE,
    MAPE,
    MASE,
    MSE,
    MSIS,
    MeanWeightedSumQuantileLoss,
    ND,
    NRMSE,
    RMSE,
    SMAPE,
)
from gluonts.model import evaluate_model
from gluonts.time_feature import get_seasonality
from notebooks.hfdemo.tinytimemixer.full_benchmarking.gift_leaderboard_ttm_r3_nc.src import (
    ttm_gluonts_predictor,
)
from notebooks.hfdemo.tinytimemixer.full_benchmarking.gift_leaderboard_ttm_r3_nc.src.ttm_gluonts_predictor import (
    TTMGluonTSPredictor,
    TTM_MAX_FORECAST_HORIZON,
)
from notebooks.hfdemo.tinytimemixer.full_benchmarking.gift_leaderboard_ttm_r3_nc.src.utils import (
    delete_if_empty_csv,
    get_args,
    set_seed,
)
import pandas as pd
import torch


# ============================================================================
# Verify that the GIFT Eval TTM source path is correctly loaded
# ============================================================================
print("GIFT Eval TTM src path =", ttm_gluonts_predictor.__file__)


# ============================================================================
# Configuration Section - Modify settings below as needed
# ============================================================================
# Set output directory for results

args = get_args()

# Version identifier for result files
ttm_version = args.ttm_version
OUT_DIR = "../results"
OUT_DIR = os.path.join(OUT_DIR, ttm_version)
os.makedirs(OUT_DIR, exist_ok=True)



# ============================================================================
# Global Configuration
# ============================================================================

# Random seed for reproducibility
SEED = 42
set_seed(SEED)
# Load environment variables from .env file
load_dotenv()

# Ensure the output directory exists
os.makedirs(OUT_DIR, exist_ok=True)

# Few-shot configuration (Optional): Restricts the number of training samples based on available data
# Supports 1K, 10K, or 50K few-shot samples. Can be enabled for faster execution.
FS_MODE = None
if args.few_shot_data_limit_config:
    FS_MODE = json.load(open(args.few_shot_data_limit_config))
    print(FS_MODE)

# ============================================================================
# Dataset Configuration
# ============================================================================
# Complete list of available datasets (commented out for reference)
short_datasets = "m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly electricity/15T electricity/H electricity/D electricity/W solar/10T solar/H solar/D solar/W hospital covid_deaths us_births/D us_births/M us_births/W saugeenday/D saugeenday/M saugeenday/W temperature_rain_with_missing kdd_cup_2018_with_missing/H kdd_cup_2018_with_missing/D car_parts_with_missing restaurant hierarchical_sales/D hierarchical_sales/W LOOP_SEATTLE/5T LOOP_SEATTLE/H LOOP_SEATTLE/D SZ_TAXI/15T SZ_TAXI/H M_DENSE/H M_DENSE/D ett1/15T ett1/H ett1/D ett1/W ett2/15T ett2/H ett2/D ett2/W jena_weather/10T jena_weather/H jena_weather/D bitbrains_fast_storage/5T bitbrains_fast_storage/H bitbrains_rnd/5T bitbrains_rnd/H bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"
med_long_datasets = "electricity/15T electricity/H solar/10T solar/H kdd_cup_2018_with_missing/H LOOP_SEATTLE/5T LOOP_SEATTLE/H SZ_TAXI/15T M_DENSE/H ett1/15T ett1/H ett2/15T ett2/H jena_weather/10T jena_weather/H bitbrains_fast_storage/5T bitbrains_rnd/5T bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"

# Active datasets for current run
# short_datasets = "ett1/H"
# med_long_datasets = ""

# Combine short and medium/long datasets into a single sorted list
all_datasets = sorted(set(short_datasets.split() + med_long_datasets.split()))

# Load dataset properties configuration
# TODO: Make this path configurable and add error handling
dataset_properties_map = json.load(open("dataset_properties.json"))
print("num datsets =", len(all_datasets))


# ============================================================================
# Evaluation Metrics Configuration
# ============================================================================
# Instantiate evaluation metrics for model performance assessment
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
    MeanWeightedSumQuantileLoss(
        quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ),
]

# Dataset name mappings for cleaner output
pretty_names = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}

# ============================================================================
# Results File Configuration
# ============================================================================
# Define the path for the CSV results file
csv_file_path = os.path.join(OUT_DIR, f"all_results.csv")
delete_if_empty_csv(csv_file_path)
# Track already processed datasets to enable resumption
DONE_DATASETS = []
if os.path.exists(csv_file_path):
    df_res = pd.read_csv(csv_file_path)
    DONE_DATASETS = df_res["dataset"].values

# ============================================================================
# CSV Writer Actor for Thread-Safe File Writing
# ============================================================================
class CSVWriter:
    def __init__(self, file_path):
        self.file_path = file_path
        if not os.path.exists(file_path):
            # Initialize CSV file with header row
            with open(file_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)

                # Write CSV header with all metric columns
                writer.writerow(
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

    def write_row(self, row):
        """Append a single row to the CSV file in a thread-safe manner."""
        with open(self.file_path, mode="a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(row)


# Instantiate the CSV writer actor
writer_actor = CSVWriter(csv_file_path)


def run_dataset(ds_name, writer_actor, done_datasets):
    """
    Process a single dataset with TTM model evaluation.
    
    Args:
        ds_name: Dataset name to process
        writer_actor: Ray actor for thread-safe CSV writing
        done_datasets: List of already processed datasets
    """
    # Set PyTorch matrix multiplication precision
    torch.set_float32_matmul_precision(precision=args.torch_matmul_precision)
    
    print(f"Processing dataset: {ds_name}")
    set_seed(SEED)
    terms = ["short", "medium", "long"]
    for term in terms:
        if (
            term == "medium" or term == "long"
        ) and ds_name not in med_long_datasets.split():
            continue

        print(f"Processing dataset: {ds_name}, term: {term}")

        if "/" in ds_name:
            ds_key = ds_name.split("/")[0]
            ds_freq = ds_name.split("/")[1]
            ds_key = ds_key.lower()
            ds_key = pretty_names.get(ds_key, ds_key)
        else:
            ds_key = ds_name.lower()
            ds_key = pretty_names.get(ds_key, ds_key)
            ds_freq = dataset_properties_map[ds_key]["frequency"]
        ds_config = f"{ds_key}/{ds_freq}/{term}"

        if ds_config in done_datasets:
            print(f"Done with {ds_config}. Skipping...")
            continue

        # Load dataset with multivariate support
        dataset = Dataset(name=ds_name, term=term, to_univariate=False)
        season_length = get_seasonality(dataset.freq)

        print(
            f"Dataset: {ds_name}, Freq = {dataset.freq}, H = {dataset.prediction_length}"
        )

        # Determine appropriate context length based on available data
        all_lengths = []
        for x in dataset.test_data:
            if len(x[0]["target"].shape) == 1:
                all_lengths.append(len(x[0]["target"]))
                num_channels = 1
            else:
                all_lengths.append(x[0]["target"].shape[1])
                num_channels = x[0]["target"].shape[0]

        min_context_length = min(all_lengths)
        print(
            "Minimum context length among all time series in this dataset =",
            min_context_length,
        )

        # Configure prediction channels
        num_prediction_channels = num_channels
        prediction_channel_indices = list(range(num_channels))

        # Check for exogenous features in the dataset
        past_feat_dynamic_real_exist = False
        if args.use_exogs and "past_feat_dynamic_real" in x[0].keys():
            num_exogs = x[0]["past_feat_dynamic_real"].shape[0]
            print(f"Data has `past_feat_dynamic_real` features of size {num_exogs}.")
            num_channels += num_exogs
            past_feat_dynamic_real_exist = True

        if dataset.prediction_length > TTM_MAX_FORECAST_HORIZON:
            # predict all channels, needed for recursive forecast
            prediction_channel_indices = list(range(num_channels))

        print("prediction_channel_indices =", prediction_channel_indices)

        # For very short time series, force short context window for finetuning
        if term == "short":
            force_short_context = args.force_short_context
        else:
            force_short_context = False

        force_zeroshot = False
        if args.num_epochs == 0:
            force_zeroshot = True
        gluonts_predictor_args = {
            "context_length": min_context_length,
            "prediction_length": min(
                dataset.prediction_length, TTM_MAX_FORECAST_HORIZON
            ),
            "test_data_label": dataset.test_data.label,
            "random_seed": SEED,
            "term": term,
            "ds_name": ds_name,
            "out_dir": OUT_DIR,
            "scale": True,
            "upper_bound_fewshot_samples": args.upper_bound_fewshot_samples,
            "force_short_context": force_short_context,
            "min_context_mult": args.min_context_mult,
            "past_feat_dynamic_real_exist": past_feat_dynamic_real_exist,
            "num_prediction_channels": num_prediction_channels,
            "freq": dataset.freq,
            "use_valid_from_train": args.use_valid_from_train,
            "insample_forecast": args.insample_forecast,
            "insample_use_train": args.insample_use_train,
            # TTM model-specific arguments
            "head_dropout": args.head_dropout,
            "decoder_mode": args.decoder_mode,
            "num_input_channels": num_channels,
            "huber_delta": args.huber_delta,
            "quantile": args.quantile,
            "loss": args.loss,
            "prediction_channel_indices": prediction_channel_indices,
            "use_mask": True,
            "force_zeroshot": force_zeroshot,
            "auto_sampler": args.auto_sampler,
            "fs_mode_dict": FS_MODE,
            "use_lite": args.use_lite,
            "plot_predictions": args.plot_predictions,
            "rolling_norm": args.rolling_norm,
            "ft_zs_ensemble": args.ft_zs_ensemble,
            "ft_zs_ensemble_mode": args.ft_zs_ensemble_mode,
            "ensemble_shrink_lambda": args.ensemble_shrink_lambda,
            "use_zs_anchor": args.use_zs_anchor,
            "zs_anchor_weight": args.zs_anchor_weight,
            "disable_extra_point_weightage": args.disable_extra_point_weightage,
        }

        predictor = TTMGluonTSPredictor(**gluonts_predictor_args)

        print(f"Number of channels in the dataset {ds_name} =", num_channels)
        if args.batch_size is None:
            batch_size = None
            optimize_batch_size = True
        else:
            batch_size = args.batch_size
            optimize_batch_size = False
        print("Batch size is set to", batch_size)

        finetune_train_num_samples = 0
        finetune_valid_num_samples = 0
        try:
            # Adjust prediction length for long-horizon forecasting
            if dataset.prediction_length > TTM_MAX_FORECAST_HORIZON:
                predictor.prediction_length = dataset.prediction_length
            
            # Fine-tune the model on the training split
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
            finetune_train_num_samples = predictor.train_num_samples
            finetune_valid_num_samples = predictor.valid_num_samples
        except Exception as e:
            print("Error in finetune workflow. Error =", e)
            print("Fallback to zero-shot performance.")
            finetune_success = False

        # Evaluate model performance on test data
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
        
        # Calculate in-sample error (currently set to 0)
        insample_error = 0
        result = [
            ds_config,
            args.ttm_version,
            res["MSE[mean]"][0],
            res["MSE[0.5]"][0],
            res["MAE[mean]"][0],
            res["MAE[0.5]"][0],
            res["MASE[0.5]"][0],
            res["MAPE[0.5]"][0],
            res["sMAPE[0.5]"][0],
            res["MSIS"][0],
            res["RMSE[mean]"][0],
            res["NRMSE[mean]"][0],
            res["ND[0.5]"][0],
            res["mean_weighted_sum_quantile_loss"][0],
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

        # Write results to CSV via the writer actor
        writer_actor.write_row(result)

        print(f"Results for {ds_name} have been written to {csv_file_path}")


print("All datasets:", all_datasets)
print("Done datasets", DONE_DATASETS)

# ============================================================================
# Execute Parallel Dataset Processing
# ============================================================================
futures = [
    run_dataset(ds_name, writer_actor, DONE_DATASETS) for ds_name in all_datasets
]

# ============================================================================
# Results Summary and Export
# ============================================================================
# Load and display final results
df = pd.read_csv(csv_file_path)
df = df.sort_values(by="dataset")
pprint(
    df[
        [
            "dataset",
            "eval_metrics/MASE[0.5]",
            "eval_metrics/NRMSE[mean]",
            "eval_metrics/mean_weighted_sum_quantile_loss",
        ]
    ]
)
df.to_excel(csv_file_path.replace(".csv", ".xlsx"))
df.to_excel(csv_file_path.replace(".csv", ".xlsx"))
