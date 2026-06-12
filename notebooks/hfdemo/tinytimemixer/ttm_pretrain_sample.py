#!/usr/bin/env python
# coding: utf-8

"""
Clean TTM pre-training script.

Model architecture is controlled by MODEL_CONFIG below.
Only runtime/data/training controls are exposed as CLI arguments.
"""

import argparse
import logging
import math
import os
import random
import tempfile
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed

from tsfm_public import TimeSeriesPreprocessor, get_datasets
from tsfm_public.models.tinytimemixer import (
    TinyTimeMixerConfig,
    TinyTimeMixerForPrediction,
)


logger = logging.getLogger(__file__)


# ---------------------------------------------------------------------
# Model config: single source of truth for architecture/model settings.
# ---------------------------------------------------------------------
MODEL_CONFIG = {
    "context_length": 1536,
    "patch_length": 64,
    "num_input_channels": 1,
    "prediction_length": 96,
    "patch_stride": 64,
    "d_model": 256,
    "expansion_factor": 3,
    "num_layers": 8,
    "dropout": 0.2,
    "loss": "mae",
    "adaptive_patching_levels": 3,
    "decoder_num_layers": 2,
    "decoder_d_model": 128,
    "decoder_adaptive_patching_levels": 0,
    "head_dropout": 0.2,
    "resolution_prefix_tuning": False,
    "multi_scale": True,
    "register_tokens": 5,
    "fft_length": 16,
    "use_fft_embedding": True,
    "multi_quantile_head": True,
    "point_extra_weight": 2,
    "gate_mode": "glu",
    "mq_use_decoder_pool": True,
    "mq_use_positional": False,
    "mq_q50_type": "mean",
    "mq_cond_mode": "concat",
    "mq_cond_path": "flatten",
    "mq_decoder_d_model": 8,
}

# Extra model defaults that were previously hardcoded in the script.
# Keep them here, not in argparse, so all model choices remain code-driven.
MODEL_DEFAULTS = {
    "mode": "common_channel",
    "scaling": "std",
    "gated_attn": True,
    "decoder_mode": "common_channel",
    "decoder_raw_residual": False,
    "use_decoder": True,
}


# ---------------------------------------------------------------------
# Runtime args only. No architecture/model config args here.
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="TinyTimeMixer pre-training with hardcoded model config.")

    # Data
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
        help="CSV path/URL for the dataset.",
    )
    parser.add_argument("--timestamp_column", type=str, default="date")
    parser.add_argument(
        "--target_columns",
        type=str,
        default="HUFL,HULL,MUFL,MULL,LUFL,LULL,OT",
        help="Comma-separated target columns.",
    )
    parser.add_argument("--train_start", type=int, default=0)
    parser.add_argument("--train_end", type=int, default=8640)
    parser.add_argument("--valid_start", type=int, default=8640)
    parser.add_argument("--valid_end", type=int, default=11520)
    parser.add_argument("--test_start", type=int, default=11520)
    parser.add_argument("--test_end", type=int, default=14400)

    # Training/runtime
    parser.add_argument("--save_dir", type=str, default="./ttm_runs")
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--early_stopping_patience", type=int, default=10)
    parser.add_argument("--use_lr_finder", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--skip_inference", action="store_true")
    parser.add_argument("--num_plots", type=int, default=40)

    return parser.parse_args()


def make_training_arguments(**kwargs) -> TrainingArguments:
    """
    transformers renamed evaluation_strategy -> eval_strategy in newer versions.
    Try the newer name first and fall back for older environments.
    """
    try:
        return TrainingArguments(eval_strategy="epoch", **kwargs)
    except TypeError:
        return TrainingArguments(evaluation_strategy="epoch", **kwargs)


# ---------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------
def get_base_model():
    config_kwargs = {**MODEL_DEFAULTS, **MODEL_CONFIG}

    # If you later add "quantile_levels" to MODEL_CONFIG, map it to the
    # TinyTimeMixerConfig field used by the model.
    quantile_levels = config_kwargs.pop("quantile_levels", None)
    if quantile_levels is not None:
        config_kwargs["quantile_list"] = list(quantile_levels)

    config = TinyTimeMixerConfig(**config_kwargs)
    model = TinyTimeMixerForPrediction(config)
    return model


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------
def get_data(args):
    target_columns = [c.strip() for c in args.target_columns.split(",") if c.strip()]
    split_config = {
        "train": [args.train_start, args.train_end],
        "valid": [args.valid_start, args.valid_end],
        "test": [args.test_start, args.test_end],
    }

    data = pd.read_csv(args.dataset_path, parse_dates=[args.timestamp_column])

    tsp = TimeSeriesPreprocessor(
        timestamp_column=args.timestamp_column,
        id_columns=[],
        target_columns=target_columns,
        control_columns=[],
        context_length=MODEL_CONFIG["context_length"],
        prediction_length=MODEL_CONFIG["prediction_length"],
        scaling=True,
        encode_categorical=False,
        scaler_type="standard",
    )

    return get_datasets(tsp, data, split_config)


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------
def maybe_find_lr(args, model, dset_train):
    if not args.use_lr_finder:
        return args.learning_rate, model

    from tsfm_public.toolkit.lr_finder import optimal_lr_finder

    learning_rate, model = optimal_lr_finder(
        model,
        dset_train,
        batch_size=args.batch_size,
    )
    print("OPTIMAL SUGGESTED LEARNING RATE =", learning_rate)
    return learning_rate, model


def steps_per_epoch(dset_train, batch_size: int) -> int:
    try:
        return max(1, math.ceil(len(dset_train) / batch_size))
    except TypeError:
        # Safe fallback for iterable datasets.
        return 1


def pretrain(args, model, dset_train, dset_val):
    learning_rate, model = maybe_find_lr(args, model, dset_train)

    trainer_args_kwargs = {
        "output_dir": os.path.join(args.save_dir, "checkpoint"),
        "overwrite_output_dir": True,
        "learning_rate": learning_rate,
        "num_train_epochs": args.num_epochs,
        "seed": args.random_seed,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "dataloader_num_workers": args.num_workers,
        "ddp_find_unused_parameters": False,
        "report_to": "tensorboard",
        "save_strategy": "epoch",
        "logging_strategy": "epoch",
        "save_total_limit": 1,
        "logging_dir": os.path.join(args.save_dir, "logs"),
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "bf16": args.bf16,
        "bf16_full_eval": args.bf16,
    }

    trainer_args = make_training_arguments(**trainer_args_kwargs)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = OneCycleLR(
        optimizer,
        learning_rate,
        epochs=args.num_epochs,
        steps_per_epoch=steps_per_epoch(dset_train, args.batch_size),
    )

    callbacks = []
    if args.early_stopping:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=0.0,
            )
        )

    trainer = Trainer(
        model=model,
        args=trainer_args,
        train_dataset=dset_train,
        eval_dataset=dset_val,
        optimizers=(optimizer, scheduler),
        callbacks=callbacks if callbacks else None,
    )

    trainer.train()

    model_save_path = os.path.join(args.save_dir, "ttm_pretrained")
    trainer.save_model(model_save_path)
    return model_save_path


# ---------------------------------------------------------------------
# Quantile helpers: no hardcoded q10/q50/q90 indices.
# ---------------------------------------------------------------------
def _fmt_q(q: float) -> str:
    return f"q{int(round(float(q) * 100)):02d}"


def _get_quantile_levels(model, qarr: Optional[np.ndarray] = None) -> Optional[list[float]]:
    """
    Prefer the model config. If unavailable, return None and use index-based labels.
    This avoids assuming a fixed number/order of quantiles.
    """
    levels = getattr(model.config, "quantile_list", None)
    if levels is None:
        levels = getattr(model.config, "quantile_levels", None)

    if levels is None:
        return None

    levels = [float(q) for q in levels]
    if qarr is not None and len(levels) != qarr.shape[1]:
        print(
            f"[WARN] quantile level count ({len(levels)}) does not match prediction Q "
            f"({qarr.shape[1]}). Falling back to index-based quantile labels."
        )
        return None
    return levels


def _pick_low_mid_high_indices(qarr: np.ndarray, quantile_levels: Optional[Sequence[float]]):
    Q = qarr.shape[1]
    low_i = 0
    high_i = Q - 1

    if quantile_levels is None:
        mid_i = Q // 2
    else:
        taus = np.asarray(quantile_levels, dtype=np.float32)
        mid_i = int(np.argmin(np.abs(taus - 0.5)))

    return low_i, mid_i, high_i


def _quantile_label(index: int, quantile_levels: Optional[Sequence[float]]) -> str:
    if quantile_levels is None:
        return f"q_index_{index}"
    return _fmt_q(float(quantile_levels[index]))


def compute_crps_from_quantiles(qhat: np.ndarray, y: np.ndarray, quantile_levels: Optional[Sequence[float]]):
    """
    qhat: [N, Q, F, C]
    y   : [N, F, C]

    If quantile levels are unavailable, CRPS is skipped because the integral
    over quantile levels cannot be computed correctly.
    """
    if quantile_levels is None:
        print("CRPS skipped: model.config.quantile_list is unavailable.")
        return None

    taus = np.asarray(quantile_levels, dtype=np.float32)
    if len(taus) != qhat.shape[1]:
        print(f"CRPS skipped: len(quantile_levels)={len(taus)} but qhat.shape[1]={qhat.shape[1]}.")
        return None

    order = np.argsort(taus)
    taus_s = taus[order]
    qhat_s = qhat[:, order, :, :]

    y_exp = np.expand_dims(y, axis=1)
    err = y_exp - qhat_s
    taus_exp = taus_s.reshape(1, len(taus_s), 1, 1)
    pinball = np.maximum(taus_exp * err, (taus_exp - 1.0) * err)

    if len(taus_s) == 1:
        return 2.0 * np.mean(pinball)

    w = np.zeros(len(taus_s), dtype=np.float32)
    w[0] = 0.5 * (taus_s[1] - taus_s[0])
    w[-1] = 0.5 * (taus_s[-1] - taus_s[-2])
    if len(taus_s) > 2:
        w[1:-1] = 0.5 * (taus_s[2:] - taus_s[:-2])

    return 2.0 * np.mean(pinball * w.reshape(1, len(w), 1, 1))


def plot_quantiles_low_mid_high(
    qarr: np.ndarray,
    forecast_groundtruth: np.ndarray,
    idx: int,
    ch: int,
    save_path: str,
    title: str,
    quantile_levels: Optional[Sequence[float]],
):
    low_i, mid_i, high_i = _pick_low_mid_high_indices(qarr, quantile_levels)

    low = qarr[idx, low_i, :, ch]
    mid = qarr[idx, mid_i, :, ch]
    high = qarr[idx, high_i, :, ch]
    gt = forecast_groundtruth[idx, :, ch]

    low_label = _quantile_label(low_i, quantile_levels)
    mid_label = _quantile_label(mid_i, quantile_levels)
    high_label = _quantile_label(high_i, quantile_levels)

    plt.figure(figsize=(12, 6))
    plt.plot(mid, label=f"Combined {mid_label}", linewidth=1.8)
    plt.plot(low, label=f"Combined {low_label}", linewidth=1.0)
    plt.plot(high, label=f"Combined {high_label}", linewidth=1.0)
    plt.plot(gt, label="Ground Truth", linewidth=1.0)

    coverage_label = f"{low_label}-{high_label} band"
    plt.fill_between(np.arange(len(mid)), low, high, alpha=0.15, label=coverage_label)

    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ---------------------------------------------------------------------
# Reporting + inference
# ---------------------------------------------------------------------
def print_model_summary(model, verbose=False):
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

    print("\n" + "=" * 80)
    print("TOP-LEVEL MODULE PARAMS")
    print("=" * 80)
    for name, module in model.named_children():
        mod_total = sum(p.numel() for p in module.parameters())
        mod_train = sum(p.numel() for p in module.parameters() if p.requires_grad)
        if mod_total > 0:
            print(f"{name:30s} total={mod_total:12,}  trainable={mod_train:12,}")

    if hasattr(model, "num_parameters"):
        try:
            print("\n" + "=" * 80)
            print("HF MODEL NUM PARAMETERS")
            print("=" * 80)
            print(
                "num_parameters(trainable_only=False) =",
                model.num_parameters(trainable_only=False),
            )
            print(
                "num_parameters(trainable_only=True ) =",
                model.num_parameters(trainable_only=True),
            )
        except Exception:
            pass

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


def _extract_prediction_tensors(predictions):
    """Extract tensors for the public tsfm_public TinyTimeMixer prediction output."""
    predictions_output = predictions[0]  # [N, F, C]
    input_data = predictions[-3]  # [N, L, C]
    forecast_groundtruth = predictions[-2]  # [N, F, C]
    comb_q = predictions[-1] if len(predictions) >= 1 else None

    return predictions_output, input_data, forecast_groundtruth, comb_q


def inference(args, model_path, dset_test):
    model = TinyTimeMixerForPrediction.from_pretrained(model_path)

    temp_dir = tempfile.mkdtemp()
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=temp_dir,
            per_device_eval_batch_size=args.batch_size,
            seed=args.random_seed,
            report_to="none",
            bf16=args.bf16,
            bf16_full_eval=args.bf16,
        ),
    )

    print("+" * 20, "Test output:", "+" * 20)
    output = trainer.evaluate(dset_test)
    print(output)

    predictions_dict = trainer.predict(dset_test)
    predictions_output, input_data, forecast_groundtruth, comb_q = _extract_prediction_tensors(
        predictions_dict.predictions
    )

    has_quantiles = bool(getattr(model.config, "multi_quantile_head", False)) and comb_q is not None

    mse = np.mean((predictions_output - forecast_groundtruth) ** 2)
    mae = np.mean(np.abs(predictions_output - forecast_groundtruth))
    print("MSE =", mse)
    print("MAE =", mae)

    quantile_levels = None
    if has_quantiles:
        quantile_levels = _get_quantile_levels(model, comb_q)
        crps = compute_crps_from_quantiles(comb_q, forecast_groundtruth, quantile_levels)
        if crps is not None:
            print("CRPS =", crps)

    save_folder = os.path.join(args.save_dir, "plots")
    os.makedirs(save_folder, exist_ok=True)

    num_samples = predictions_output.shape[0]
    num_plots = min(args.num_plots, num_samples)

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
        plt.plot(forecast_ori, label="Ground Truth", linewidth=1)
        plt.title(f"Forecasts (Sample {idx}, Channel {ch})")
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, f"plot_{i}.png"))
        plt.close()

        if has_quantiles:
            plot_quantiles_low_mid_high(
                qarr=comb_q,
                forecast_groundtruth=forecast_groundtruth,
                idx=idx,
                ch=ch,
                quantile_levels=quantile_levels,
                title=f"Combined Quantiles (Sample {idx}, Channel {ch})",
                save_path=os.path.join(save_folder, f"quant_combined_{i}.png"),
            )

    print(f"Saved {num_plots} plots to: {save_folder}")
    print_model_summary(model, verbose=True)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    print("public TSFM")

    set_seed(args.random_seed)

    logger.info(
        "%s Pre-training TTM | context=%s prediction=%s %s",
        "*" * 20,
        MODEL_CONFIG["context_length"],
        MODEL_CONFIG["prediction_length"],
        "*" * 20,
    )

    dset_train, dset_valid, dset_test = get_data(args)

    model = get_base_model()

    model_save_path = pretrain(args, model, dset_train, dset_valid)
    print("=" * 20, "Pretraining Completed!", "=" * 20)
    print("Model saved in location:", model_save_path)

    if not args.skip_inference:
        inference(
            args=args,
            model_path=model_save_path,
            dset_test=dset_test,
        )
        print("inference completed..")
