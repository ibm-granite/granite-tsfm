import argparse
import logging
import os
import random
import tempfile
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import Trainer, TrainingArguments
from tsfm_public import TimeSeriesPreprocessor, get_datasets
from tsfm_public.models.tinytimemixer import (
    TinyTimeMixerForDecomposedPrediction,
    TinyTimeMixerForPrediction,
)

logger = logging.getLogger("ttm_infer_only")
logging.basicConfig(level=logging.INFO)


# ---------------------------
# Demo dataset configuration
# ---------------------------
TARGET_DATASET = "etth1"
DATASET_URL = (
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
)
TIMESTAMP_COLUMN = "date"
ID_COLUMNS = []  # ETTh1 is a single multivariate series

# ETTh1 standard 7 targets
TARGET_COLUMNS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]

# Standard split used in many examples
SPLIT_CONFIG = {
    "train": [0, 8640],
    "valid": [8640, 11520],
    "test": [11520, 14400],
}

import inspect

import numpy as np
import pandas as pd
import torch
from transformers import default_data_collator


def make_ttm_collator(model):
    """
    Build a collator that:
      1) Drops pandas Timestamps (and other non-tensorizable metadata)
      2) Keeps only keys accepted by model.forward (unless it has **kwargs)
    """
    sig = inspect.signature(model.forward)
    accepts_kwargs = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
    allowed = set(sig.parameters.keys())  # includes self? (no), just args
    # Common keys you never want to pass to forward
    drop_always = {"id", "timestamp", "start", "freq", "item_id"}

    def _is_bad_value(v):
        if isinstance(v, pd.Timestamp):
            return True
        # list/array of timestamps
        if (
            isinstance(v, (list, tuple))
            and len(v) > 0
            and isinstance(v[0], pd.Timestamp)
        ):
            return True
        return False

    def collate(features):
        if len(features) == 0:
            return {}

        cleaned = []
        for f in features:
            g = {}
            for k, v in f.items():
                if k in drop_always:
                    continue
                if _is_bad_value(v):
                    continue
                # If forward does NOT accept **kwargs, keep only allowed keys
                if (not accepts_kwargs) and (k not in allowed):
                    continue
                g[k] = v
            cleaned.append(g)

        return default_data_collator(cleaned)

    return collate


def print_model_summary(model, verbose: bool = False) -> None:
    print("\n" + "=" * 80)
    print("MODEL STRUCTURE")
    print("=" * 80)
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    print("\n" + "=" * 80)
    print("PARAMETER COUNT")
    print("=" * 80)
    print(f"Total params      : {total_params:,}")
    print(f"Trainable params  : {trainable_params:,}")
    print(f"Non-trainable     : {non_trainable_params:,}")

    if verbose:
        print("\n" + "=" * 80)
        print("NAMED PARAMETERS (VERBOSE)")
        print("=" * 80)
        for name, param in model.named_parameters():
            print(
                f"{name:80s} "
                f"shape={tuple(param.shape)} "
                f"numel={param.numel():,} "
                f"trainable={param.requires_grad}"
            )


def build_test_dataset_from_model_config(model):
    """
    Build ETTh1 test dataset where context_length & prediction_length are taken
    from model.config.
    """
    context_length = int(getattr(model.config, "context_length"))
    prediction_length = int(getattr(model.config, "prediction_length"))

    logger.info(
        f"Using model.config context_length={context_length}, prediction_length={prediction_length}"
    )

    data = pd.read_csv(DATASET_URL, parse_dates=[TIMESTAMP_COLUMN])

    column_specifiers = {
        "timestamp_column": TIMESTAMP_COLUMN,
        "id_columns": ID_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "control_columns": [],
    }

    tsp = TimeSeriesPreprocessor(
        **column_specifiers,
        context_length=context_length,
        prediction_length=prediction_length,
        scaling=True,
        encode_categorical=False,
        scaler_type="standard",
    )

    _, _, dset_test = get_datasets(tsp, data, SPLIT_CONFIG)
    return dset_test


@torch.no_grad()
def run_inference_only(model_path: str) -> None:
    # Load model
    model = TinyTimeMixerForPrediction.from_pretrained(
        model_path, light_mode=True
    )
    model.eval()

    # if torch.cuda.is_available():
    #     model = model.to("cuda")

    #     # compile for faster inference (PyTorch 2.x)
    #     try:
    #         model = torch.compile(model, mode="reduce-overhead")
    #         print("torch.compile enabled")
    #     except Exception as e:
    #         print(f"torch.compile not enabled: {e}")
    # Build dataset driven by model.config context/pred lengths

    dset_test = build_test_dataset_from_model_config(model)

    # Minimal Trainer just for eval/predict
    temp_dir = tempfile.mkdtemp(prefix="ttm_infer_")
    args = TrainingArguments(
        output_dir=temp_dir,
        per_device_eval_batch_size=512,  # fixed, no CLI param
        report_to="none",
        dataloader_num_workers=2,
        # bf16=torch.cuda.is_available(),  # safe auto-enable if GPU
        # remove_unused_columns=False,  # <-- critical for torch.compile
        bf16=True,
    )
    trainer = Trainer(
        model=model,
        args=args,
        # data_collator=make_ttm_collator(model),
    )  # <-- add this)

    # Evaluate
    # print("+" * 20, "Test metrics (trainer.evaluate)", "+" * 20)
    # metrics = trainer.evaluate(dset_test)
    # print(metrics)

    # Predict
    # pred_out = trainer.predict(dset_test)

    # TTM predict() returns a tuple-like predictions array pack.
    # We keep the same robust indexing you had, but without args/use_internal_tsfm.
    # Predict
    print("\nStarting prediction...")
    start_time = time.perf_counter()

    pred_out = trainer.predict(dset_test)
    preds = pred_out.predictions
    end_time = time.perf_counter()
    elapsed_sec = end_time - start_time

    print(f"\nPrediction completed in {elapsed_sec:.4f} seconds")
    print(f"Total samples: {len(dset_test)}")
    print(f"Throughput: {len(dset_test) / elapsed_sec:.2f} windows/sec")

    if not model.config.light_mode:
        # Convention used in your script:
        # - predictions_output: preds[0]          -> [N, F, C] (inverse-scaled forecast)
        # - input_data:         preds[-3]         -> [N, L, C] (raw scale input)
        # - ground truth:       preds[-2]         -> [N, F, C] (raw scale)
        predictions_output = preds[0]
        input_data = preds[-3]
        forecast_groundtruth = preds[-2]
        has_quantiles = bool(getattr(model.config, "multi_quantile_head", False))
        comb_q = preds[-1] if has_quantiles else None

        mse = float(np.mean((predictions_output - forecast_groundtruth) ** 2))
        print("\nMSE (forecast vs GT) =", mse)

        # Save a few plots near the model_path
        save_folder = os.path.join(model_path, "inference_plots")
        os.makedirs(save_folder, exist_ok=True)

        num_samples = predictions_output.shape[0]
        num_plots = min(20, num_samples)
        ch = 0

        for i in range(num_plots):
            idx = random.randint(0, num_samples - 1)

            forecast_main = predictions_output[idx, :, ch]
            forecast_ori = forecast_groundtruth[idx, :, ch]
            input_main = input_data[idx, :, ch]

            plt.figure(figsize=(12, 8))

            plt.subplot(2, 1, 1)
            plt.plot(input_main, label="Input", linewidth=1.5)
            plt.title(f"Inputs (Sample {idx}, Channel {ch})")
            plt.legend()

            plt.subplot(2, 1, 2)
            plt.plot(forecast_main, label="Forecast", linewidth=1.5)
            plt.plot(forecast_ori, label="Ground Truth", linewidth=1.0)
            plt.title(f"Forecasts (Sample {idx}, Channel {ch})")
            plt.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(save_folder, f"plot_{i}.png"))
            plt.close()

            if has_quantiles and comb_q is not None:
                # Expected shape often: [N, Q, F, C] where quantiles are ordered.
                # We try q10/q50/q90 by index; if your quantile set differs, adapt here.
                try:
                    q10 = comb_q[idx, 0, :, ch]
                    q50 = comb_q[idx, comb_q.shape[1] // 2, :, ch]
                    q90 = comb_q[idx, -1, :, ch]

                    plt.figure(figsize=(12, 6))
                    plt.plot(q50, label="Combined q50", linewidth=1.8)
                    plt.plot(q10, label="Combined q10", linewidth=1.0)
                    plt.plot(q90, label="Combined q90", linewidth=1.0)
                    plt.plot(forecast_ori, label="Ground Truth", linewidth=1.0)
                    plt.fill_between(
                        np.arange(len(q50)), q10, q90, alpha=0.15, label="Band"
                    )
                    plt.title(f"Combined Quantiles (Sample {idx}, Channel {ch})")
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_folder, f"quant_combined_{i}.png"))
                    plt.close()
                except Exception as e:
                    logger.warning(f"Quantile plotting skipped for sample {idx}: {e}")

        print(f"\nSaved {num_plots} plots to: {save_folder}")

    # print_model_summary(model, verbose=False)


def main():
    parser = argparse.ArgumentParser(
        description="TTM inference-only (model_path only)."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to a saved TinyTimeMixer checkpoint (from_pretrained compatible).",
    )
    args = parser.parse_args()
    run_inference_only(args.model_path)


if __name__ == "__main__":
    main()
