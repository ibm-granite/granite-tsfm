import argparse
import json
import logging
import os
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gift_eval_windows import get_gift_ensemble_predictions_df, get_test_window_lengths


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
from gluonts.model.evaluation import evaluate_forecasts
from gluonts.model.forecast import QuantileForecast
from gluonts.time_feature import get_seasonality
from tqdm import tqdm
import random
import torch

logging.getLogger("gluonts.model.predictor").setLevel(logging.ERROR)
logging.getLogger("gluonts.model.forecast").setLevel(logging.ERROR)
import warnings

warnings.filterwarnings("ignore")

load_dotenv()

DATASET_PROPERTIES_FILE = Path(__file__).with_name("dataset_properties.json")


def load_experiment_config(path=None):
    """Load the adjacent JSON by default, independent of the working directory."""
    path = Path(path) if path is not None else Path(__file__).with_name("experiment_config.json")
    config = json.loads(path.read_text())
    configurations = config["configurations"]
    if not configurations or config["defaults"]["model_name_config"] not in configurations:
        raise ValueError("The default model_name_config must name a configured experiment.")
    for name, recipe in configurations.items():
        if not recipe.get("model_names"):
            raise ValueError(f"Experiment {name!r} must specify model_names.")
        if recipe.get("ensemble") not in ("probability_space_aggregation", "quantile_space_aggregation", "iqr_weighted"):
            raise ValueError(f"Experiment {name!r} has an unsupported ensemble method.")
    # Benchmark adapters currently emit this fixed quantile grid.
    if config["quantile_levels"] != [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        raise ValueError("GIFT-Eval adapters require quantile_levels 0.1 through 0.9 in increments of 0.1.")
    if not config["terms"] or any(term not in ("short", "medium", "long") for term in config["terms"]):
        raise ValueError("terms must contain short, medium, or long.")
    return config


EXPERIMENT_CONFIG = load_experiment_config()
RUN_DEFAULTS = EXPERIMENT_CONFIG["defaults"]
CONFIGURATIONS = EXPERIMENT_CONFIG["configurations"]
QUANTILE_LEVELS = EXPERIMENT_CONFIG["quantile_levels"]
pretty_names = EXPERIMENT_CONFIG["dataset_aliases"]
short_datasets = " ".join(EXPERIMENT_CONFIG["datasets"]["short"])
med_long_datasets = " ".join(EXPERIMENT_CONFIG["datasets"]["medium_long"])
DATASET_FAST_FIRST = EXPERIMENT_CONFIG["datasets"]["fast_first"]

# Define datasets and fallback model
all_datasets = list(set(short_datasets.split() + med_long_datasets.split()))
dataset_properties_map = json.loads(DATASET_PROPERTIES_FILE.read_text())


# Auxiliary functions
def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_quantiles_prediction(df):
    """Extract quantiles predictions and convert them into glutonts compatible format
    The input df should have fields 'quantiles_0' to 'quantiles_8'
    """
    quantiles = []
    for i in range(len(QUANTILE_LEVELS)):
        quantiles.append(df[f"quantile_{i}"])

    stacked_lists = [np.stack(li, axis=0) for li in quantiles]
    combined = np.stack(stacked_lists, axis=1)
    quantile_forecasts = [
        QuantileForecast(
            forecast_arrays=x,
            start_date=pd.Period(df["future_start"].iloc[i], freq=df["frequency"].iloc[i]),
            forecast_keys=[str(q) for q in QUANTILE_LEVELS],
        )
        for i, x in enumerate(combined)
    ]
    return quantile_forecasts


def eval_gift_dataset(dataset, ds_config, df):
    print(f"Processing {ds_config}")
    test_data = dataset.test_data
    L: Any = test_data.prediction_length
    season_length = get_seasonality(dataset.freq)

    pred_cols = pd.json_normalize(df["final_pred"])
    df = df.drop(columns=["final_pred"]).join(pred_cols)
    quantile_forecasts = extract_quantiles_prediction(df)

    metrics = [
        MSE(forecast_type="mean"),
        MSE(forecast_type=0.5),
        MAE(),
        MASE(),
        MAPE(),
        SMAPE(),
        MSIS(),
        RMSE(),
        NRMSE(),
        ND(),
        MeanWeightedSumQuantileLoss(quantile_levels=QUANTILE_LEVELS),
    ]

    results = evaluate_forecasts(
        forecasts=quantile_forecasts,
        test_data=test_data,
        metrics=metrics,
        axis=None,
        mask_invalid_label=True,
        allow_nan_forecast=False,
        seasonality=season_length,
    )
    results.insert(loc=0, column="dataset", value=ds_config)
    return results


def reformatted_metrics_for_leaderboard(row):
    reformatted = {}
    # .columns will give the keys, .iloc[0] will get the value for the first (only) row
    for key in row.columns:
        if key == "dataset":
            continue
        else:
            reformatted[f"eval_metrics/{key}"] = row.iloc[0][key]
    return reformatted


def append_leaderboard_result(output_path, dataset_config, model_name, metrics, domain, num_variates):
    row = {
        "dataset": dataset_config,
        "model": model_name,
        **metrics,
        "domain": domain,
        "num_variates": num_variates,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(
        output_path,
        mode="a",
        header=not output_path.exists(),
        index=False,
    )


def get_processed_datasets(out_name):
    """Get list of datasets already processed in the output file."""
    if not os.path.exists(out_name):
        return set()
    try:
        df = pd.read_csv(out_name)
        if "dataset" in df.columns:
            return set(df["dataset"].unique())
    except Exception as e:
        print(f"Warning: Could not read existing results file: {e}")
    return set()


def log_execution_status(error_log_file, dataset_config, success, execution_time, error_message=""):
    """Log execution status to error log file."""
    log_entry = {
        "dataset_config": dataset_config,
        "success": success,
        "execution_time_seconds": execution_time,
        "error_message": error_message,
        "timestamp": pd.Timestamp.now().isoformat(),
    }
    log_df = pd.DataFrame([log_entry])
    log_df.to_csv(error_log_file, mode="a", header=not os.path.exists(error_log_file), index=False)



def resolve_device(device=None):
    """Select an available inference device, preserving the CLI's auto order."""
    if device is None:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if device not in {"cuda", "cpu", "mps"}:
        raise ValueError("device must be cuda, cpu, mps, or None")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; select device='cpu' or use a CUDA host")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable; select device='cpu'")
    return device


def run_evaluation(
    model_name_config=RUN_DEFAULTS["model_name_config"],
    out_dir=RUN_DEFAULTS["out_dir"],
    out_name=RUN_DEFAULTS["out_name"],
    error_log_name=RUN_DEFAULTS["error_log_name"],
    skip_processed=RUN_DEFAULTS["skip_processed"],
    patchtst_use_fill_nan=RUN_DEFAULTS["patchtst_use_fill_nan"],
    save_member_results=RUN_DEFAULTS["save_member_results"],
    datasets=RUN_DEFAULTS["datasets"],
    seed=RUN_DEFAULTS["seed"],
    device=RUN_DEFAULTS["device"],
):
    """Run the reference benchmark and return its CSV path.

    Individual task failures are logged and skipped, as in the original CLI.
    Resumption skips task names already present in the output CSV.
    """
    if model_name_config not in CONFIGURATIONS:
        raise ValueError(f"Unknown model configuration: {model_name_config}")
    device = resolve_device(device)
    set_seed(seed)
    out_dir = str(Path(out_dir) / model_name_config)
    os.makedirs(out_dir, exist_ok=True)
    result_filename = out_name
    out_name = os.path.join(out_dir, out_name)
    error_log_file = os.path.join(out_dir, error_log_name)

    from ptm_forecasters import build_gift_ensemble

    CANDIDATE_MODELS = CONFIGURATIONS[model_name_config]["model_names"]
    if "ensemble" in CONFIGURATIONS[model_name_config].keys():
        ENSEMBLE = CONFIGURATIONS[model_name_config]["ensemble"]
    else:
        ENSEMBLE = "probability_space_aggregation"
    IQR_TEMPERATURE = CONFIGURATIONS[model_name_config].get("iqr_temperature", 1.0)
    IQR_MAX_WEIGHT = CONFIGURATIONS[model_name_config].get("iqr_max_weight", None)

    # Get already processed datasets if skip_processed is enabled
    processed_datasets = get_processed_datasets(out_name) if skip_processed else set()
    if processed_datasets:
        print(f"Found {len(processed_datasets)} already processed datasets. Skipping them.")

    argv = []
    # Sort datasets based on number of samples
    all_datasets = datasets or DATASET_FAST_FIRST
    # all_datasets = DATASET_FAST_FIRST[0:1] ## UNCOMMENT TO TEST ONE DATASET
    for ds_name in tqdm(all_datasets, desc="Processing datasets"):
        ds_key = ds_name.split("/")[0]
        terms = EXPERIMENT_CONFIG["terms"]
        for term in terms:
            if (term == "medium" or term == "long") and ds_name not in med_long_datasets.split():
                continue

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

            # Skip if already processed
            if ds_config in processed_datasets:
                print(f"Skipping already processed dataset: {ds_config}")
                continue

            # Track execution time
            start_time = time.time()

            try:
                """
                Initialize the dataset
                """
                to_univariate = (
                    False if Dataset(name=ds_name, term=term, to_univariate=False).target_dim == 1 else True
                )
                dataset = Dataset(name=ds_name, term=term, to_univariate=to_univariate)
                freq_str = str(dataset.freq)
                season_length = get_seasonality(freq_str.replace("H", "h")) # Using H throws pandas compatibility warning
                domain = dataset_properties_map[ds_key]["domain"]
                num_variates = dataset_properties_map[ds_key]["num_variates"]
                no_daily = "l2c" in ds_name

                # season_length = get_seasonality(str(dataset.freq))
                dataset_config = f"{ds_key}/{ds_freq}/{term}"

                """
                Initialize Ensemble Model
                """
                ttm_pred_length = None
                ttm_context_length = None
                ### TTM model selection needed parameters: min context_length and max pred_length in the dataset
                if ("ttm-r3-pt" in CANDIDATE_MODELS) | ("granite-ttm-r3" in CANDIDATE_MODELS):
                    ttm_context_length, ttm_pred_length = get_test_window_lengths(dataset)
                    print(
                        f"TTM model updated with context length {ttm_context_length} and prediction length {ttm_pred_length}"
                    )
                ############
                model_pipeline = build_gift_ensemble(
                    candidate_models=CANDIDATE_MODELS,
                    ensemble_method=ENSEMBLE,
                    freq=freq_str,
                    domain=domain,
                    term=term,
                    no_daily=no_daily,
                    ttm_context_length=ttm_context_length,
                    ttm_pred_length=ttm_pred_length,
                    ttm_scaling_data=(
                        dataset.test_data.input
                        if ttm_context_length is not None
                        else None
                    ),
                    device=device,
                    patchtst_use_fill_nan=patchtst_use_fill_nan,
                    quantile_levels=QUANTILE_LEVELS,
                    iqr_temperature=IQR_TEMPERATURE,
                    iqr_max_weight=IQR_MAX_WEIGHT,
                )

                """
                Run & Evaluate Model's prediction
                """

                prediction_frames = get_gift_ensemble_predictions_df(
                    dataset,
                    model_pipeline,
                    include_member_forecasts=save_member_results,
                )
                if save_member_results:
                    df, member_frames = prediction_frames
                else:
                    df = prediction_frames
                    member_frames = {}
                out = eval_gift_dataset(dataset, ds_config, df)
                result_metrics = reformatted_metrics_for_leaderboard(out)
                append_leaderboard_result(
                    out_name,
                    ds_config,
                    model_name_config,
                    result_metrics,
                    domain,
                    num_variates,
                )

                expected_forecasts = len(dataset.test_data)
                for member_name, member_frame in member_frames.items():
                    if len(member_frame) != expected_forecasts:
                        logging.warning(
                            "Not saving %s metrics for %s: received %d of %d forecasts",
                            member_name,
                            ds_config,
                            len(member_frame),
                            expected_forecasts,
                        )
                        continue
                    member_out = eval_gift_dataset(dataset, ds_config, member_frame)
                    member_metrics = reformatted_metrics_for_leaderboard(member_out)
                    member_output_path = Path(out_dir) / "members" / member_name / result_filename
                    append_leaderboard_result(
                        member_output_path,
                        ds_config,
                        member_name,
                        member_metrics,
                        domain,
                        num_variates,
                    )

                # Log success
                execution_time = time.time() - start_time
                log_execution_status(error_log_file, ds_config, True, execution_time)
                print(f"✓ Successfully processed {ds_config} in {execution_time:.2f}s")

            except Exception as e:
                # Log failure
                execution_time = time.time() - start_time
                error_message = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
                log_execution_status(error_log_file, ds_config, False, execution_time, error_message)
                print(f"✗ Error processing {ds_config}: {type(e).__name__}: {str(e)}")
                print(f"  Full traceback logged to {error_log_file}")
                # Continue with next dataset instead of crashing
                continue

    return Path(out_name)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluation script")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=RUN_DEFAULTS["out_dir"],
    )
    parser.add_argument("--out_name", type=str, default=RUN_DEFAULTS["out_name"])
    parser.add_argument("--error_log_name", type=str, default=RUN_DEFAULTS["error_log_name"])
    parser.add_argument(
        "--model_name_config",
        type=str,
        default=RUN_DEFAULTS["model_name_config"],
        choices=CONFIGURATIONS,
        help="Ensemble configuration to evaluate",
    )
    parser.add_argument("--skip_processed", action="store_true", default=RUN_DEFAULTS["skip_processed"], help="Skip datasets already in output file")
    parser.add_argument(
        "--patchtst-use-fill-nan",
        action="store_true",
        default=RUN_DEFAULTS["patchtst_use_fill_nan"],
        help="Fill NaN values in input series for PatchTST-FM forecasters. Default comes from experiment_config.json.",
    )
    parser.add_argument(
        "--save-member-results",
        action="store_true",
        default=RUN_DEFAULTS["save_member_results"],
        help="Save metrics for each ensemble member without rerunning inference.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=RUN_DEFAULTS["datasets"],
        help="Dataset names to run, e.g. m_dense/D LOOP_SEATTLE/5T. Defaults to the full dataset list.",
    )
    parser.add_argument("--seed", type=int, default=RUN_DEFAULTS["seed"])
    parser.add_argument("--device", choices=["cuda", "cpu", "mps"], default=RUN_DEFAULTS["device"],
                        help="Inference device; defaults to CUDA, then MPS, then CPU")
    return parser.parse_args(argv)



if __name__ == "__main__":
    run_evaluation(**vars(parse_args()))
