from __future__ import annotations

import argparse
import logging
import os
import random
import tempfile
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import Trainer, TrainingArguments, set_seed

from tsfm_public import load_dataset
from tsfm_public.models.tinytimemixer import (
    TinyTimeMixerConfig,
    TinyTimeMixerForDecomposedPrediction,
)


logger = logging.getLogger(__name__)


MODEL_CONFIG = {
    "context_length": 512,
    "patch_length": 32,
    "num_input_channels": 1,
    "prediction_length": 48,
    "patch_stride": 32,
    "d_model": 192,
    "expansion_factor": 3,
    "num_layers": 8,
    "dropout": 0.2,
    "loss": "mae",
    "adaptive_patching_levels": 3,
    "decoder_num_layers": 2,
    "decoder_d_model": 64,
    "decoder_adaptive_patching_levels": 0,
    "head_dropout": 0.2,
    "resolution_prefix_tuning": False,
    "multi_scale": True,
    "register_tokens": 2,
    "fft_length": 0,
    "use_fft_embedding": True,
    "multi_quantile_head": True,
    "point_extra_weight": 2,
    "residual_context_length": 512,
    "trend_patch_length": 64,
    "trend_patch_stride": 64,
    "trend_d_model": 96,
    "trend_decoder_d_model": 96,
    "trend_num_layers": 5,
    "trend_decoder_num_layers": 2,
    "mq_use_decoder_pool": True,
    "mq_use_positional": False,
    "mq_q50_type": "mean",
    "decompose": True,
    "mq_cond_mode": "concat",
    "mq_cond_path": "flatten",
    "mq_decoder_d_model": 8,
    "combine_quantiles_via_variance": True,
    # Single source of truth for all model quantile outputs, CRPS, and plots.
    "quantile_levels": [0.1, 0.2, 0.5, 0.8, 0.9],
}

# Fixed model defaults that were previously mixed into argparse/get_base_model.
MODEL_DEFAULTS = {
    "mode": "common_channel",
    "scaling": "revin",
    "gated_attn": True,
    "decoder_mode": "common_channel",
    "decoder_raw_residual": False,
    "use_decoder": True,
}


@dataclass
class RuntimeConfig:
    dataset_name: str = "etth1"
    dataset_root_path: str = "/dccstor/tsfm23/datasets"
    save_dir: str = "./ttm_runs"
    batch_size: int = 32
    num_workers: int = 4
    num_epochs: int = 30
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    random_seed: int = 42
    epochs_phase1: Optional[int] = None
    epochs_phase2: Optional[int] = None
    epochs_phase3: Optional[int] = None
    report_to: str = "tensorboard"
    bf16: bool = True
    run_inference: bool = True
    num_random_plots: int = 10
    debug_grad_hooks: bool = False


@dataclass
class PhaseEpochs:
    trend: int
    residual: int
    joint: int


def parse_args() -> RuntimeConfig:
    parser = argparse.ArgumentParser(description="TinyTimeMixer decomposed pretraining")

    # Data/runtime knobs only. Model config lives in MODEL_CONFIG above.
    parser.add_argument("--dataset_name", type=str, default=RuntimeConfig.dataset_name)
    parser.add_argument("--dataset_root_path", type=str, default=RuntimeConfig.dataset_root_path)
    parser.add_argument("--save_dir", type=str, default=RuntimeConfig.save_dir)
    parser.add_argument("--batch_size", type=int, default=RuntimeConfig.batch_size)
    parser.add_argument("--num_workers", type=int, default=RuntimeConfig.num_workers)
    parser.add_argument("--num_epochs", type=int, default=RuntimeConfig.num_epochs)
    parser.add_argument("--learning_rate", type=float, default=RuntimeConfig.learning_rate)
    parser.add_argument("--weight_decay", type=float, default=RuntimeConfig.weight_decay)
    parser.add_argument("--random_seed", type=int, default=RuntimeConfig.random_seed)
    parser.add_argument("--epochs_phase1", type=int, default=None, help="Trend-only epochs")
    parser.add_argument("--epochs_phase2", type=int, default=None, help="Residual-only epochs")
    parser.add_argument("--epochs_phase3", type=int, default=None, help="Joint epochs")
    parser.add_argument("--report_to", type=str, default=RuntimeConfig.report_to)
    parser.add_argument("--num_random_plots", type=int, default=RuntimeConfig.num_random_plots)
    parser.add_argument("--no_bf16", action="store_true", help="Disable bf16 training/eval")
    parser.add_argument("--no_inference", action="store_true", help="Skip final inference/plots")
    parser.add_argument("--debug_grad_hooks", action="store_true", help="Print missing gradients")

    args = parser.parse_args()
    return RuntimeConfig(
        dataset_name=args.dataset_name,
        dataset_root_path=args.dataset_root_path,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        random_seed=args.random_seed,
        epochs_phase1=args.epochs_phase1,
        epochs_phase2=args.epochs_phase2,
        epochs_phase3=args.epochs_phase3,
        report_to=args.report_to,
        bf16=not args.no_bf16,
        run_inference=not args.no_inference,
        num_random_plots=args.num_random_plots,
        debug_grad_hooks=args.debug_grad_hooks,
    )


def model_cfg(key: str):
    return MODEL_CONFIG[key]


def infer_phase_epochs(args: RuntimeConfig) -> PhaseEpochs:
    provided = [args.epochs_phase1, args.epochs_phase2, args.epochs_phase3]
    if all(x is not None for x in provided):
        return PhaseEpochs(*map(int, provided))

    if any(x is not None for x in provided):
        raise ValueError(
            "Either provide all three phase epoch args "
            "(--epochs_phase1/2/3), or provide none and let the script split num_epochs."
        )

    base = max(1, args.num_epochs // 3)
    joint = max(1, args.num_epochs - 2 * base)
    return PhaseEpochs(trend=base, residual=base, joint=joint)


def print_learnable_blocks(model: nn.Module) -> None:
    print("=== Learnable Blocks in Model ===")
    total_params = 0
    trainable_params = 0

    for name, module in model.named_modules():
        params = list(module.parameters(recurse=False))
        total_params += sum(p.numel() for p in params)
        if any(p.requires_grad for p in params):
            n_params = sum(p.numel() for p in params if p.requires_grad)
            trainable_params += n_params
            print(f"[{name}] -> {n_params:,} parameters")

    print("\n=== Summary ===")
    print(f"Total trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    if total_params:
        print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")


def set_requires_grad(module: nn.Module, flag: bool) -> None:
    for param in module.parameters():
        param.requires_grad = flag


def trainable_params(model: nn.Module):
    return [param for param in model.parameters() if param.requires_grad]


def make_training_arguments(args: RuntimeConfig, save_suffix: str, num_epochs: int) -> TrainingArguments:
    kwargs = {
        "output_dir": os.path.join(args.save_dir, f"checkpoint_{save_suffix}"),
        "overwrite_output_dir": True,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "num_train_epochs": num_epochs,
        "seed": args.random_seed,
        "eval_strategy": "epoch",
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "dataloader_num_workers": args.num_workers,
        "ddp_find_unused_parameters": False,
        "report_to": args.report_to,
        "save_strategy": "epoch",
        "logging_strategy": "epoch",
        "save_total_limit": 1,
        "logging_dir": os.path.join(args.save_dir, f"logs_{save_suffix}"),
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "bf16": args.bf16,
        "bf16_full_eval": args.bf16,
    }

    try:
        return TrainingArguments(**kwargs)
    except TypeError:
        # Compatibility with older transformers versions.
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
        return TrainingArguments(**kwargs)


def make_trainer(
    model: nn.Module,
    dset_train,
    dset_val,
    args: RuntimeConfig,
    num_epochs: int,
    save_suffix: str,
) -> Trainer:
    trainer_args = make_training_arguments(args, save_suffix=save_suffix, num_epochs=num_epochs)
    optimizer = AdamW(trainable_params(model), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, int(num_epochs)))

    return Trainer(
        model=model,
        args=trainer_args,
        train_dataset=dset_train,
        eval_dataset=dset_val,
        optimizers=(optimizer, scheduler),
    )


def register_missing_grad_hooks(model: nn.Module) -> None:
    for name, param in model.named_parameters():

        def hook(grad, n=name):
            if grad is None:
                print(f"[WARN] No grad for {n}")

        param.register_hook(hook)


def get_quantile_levels() -> list[float]:
    """Return the quantile levels configured for the model.

    Keep this as the single source of truth in the script. TinyTimeMixerConfig
    currently expects the field name `quantile_list`, so build_model maps
    MODEL_CONFIG["quantile_levels"] -> quantile_list before constructing
    the HF config.
    """
    levels = MODEL_CONFIG.get("quantile_levels")
    if not levels:
        raise ValueError('MODEL_CONFIG must define non-empty "quantile_levels".')

    levels = [float(q) for q in levels]
    if any(q <= 0.0 or q >= 1.0 for q in levels):
        raise ValueError(f"All quantile_levels must be between 0 and 1. Got: {levels}")
    if len(set(levels)) != len(levels):
        raise ValueError(f"quantile_levels must be unique. Got: {levels}")
    return levels


def get_config_quantile_levels(config=None, q_count: Optional[int] = None) -> list[float]:
    """Resolve quantile levels from a saved model config or MODEL_CONFIG."""
    levels = None
    if config is not None:
        levels = getattr(config, "quantile_levels", None)
        if levels is None:
            levels = getattr(config, "quantile_list", None)

    if levels is None:
        levels = get_quantile_levels()
    else:
        levels = [float(q) for q in levels]

    if q_count is not None and len(levels) != q_count:
        raise ValueError(
            f"Quantile level count mismatch: len(quantile_levels)={len(levels)} but prediction Q={q_count}. "
            f"Configured levels: {levels}"
        )
    return levels


def build_ttm_config_dict() -> dict:
    config_dict = {**MODEL_DEFAULTS, **MODEL_CONFIG}

    quantile_levels = get_quantile_levels()

    # TinyTimeMixerConfig uses `quantile_list`. Keep the user-facing script key
    # as `quantile_levels`, but pass the expected config key to the model.
    config_dict.pop("quantile_levels", None)
    config_dict["quantile_list"] = quantile_levels

    return config_dict


def build_model(debug_grad_hooks: bool = False) -> TinyTimeMixerForDecomposedPrediction:
    config = TinyTimeMixerConfig(**build_ttm_config_dict())
    model = TinyTimeMixerForDecomposedPrediction(config)

    if debug_grad_hooks:
        register_missing_grad_hooks(model)

    return model


def set_stage_weights(model, stage: str, trend: float, residual: float, joint: float) -> None:
    if hasattr(model, "set_stage"):
        model.set_stage(stage, trend, residual, joint)
        return

    model.trend_loss_weight = trend
    model.residual_loss_weight = residual
    model.joint_loss_weight = joint


def pretrain(args: RuntimeConfig, model, dset_train, dset_val) -> str:
    phase_epochs = infer_phase_epochs(args)

    # ---------------- Phase 1: trend-only ----------------
    print("\n=== Phase 1: Trend-only ===")
    model.config.forecast_loss_type = "trend"
    set_stage_weights(model, "trend", trend=1.0, residual=0.0, joint=0.0)
    set_requires_grad(model, False)
    set_requires_grad(model.trend_forecaster, True)
    if getattr(model, "multi_quantile_head_block", None) is not None:
        set_requires_grad(model.multi_quantile_head_block, True)
    print_learnable_blocks(model)

    trainer = make_trainer(
        model=model,
        dset_train=dset_train,
        dset_val=dset_val,
        args=args,
        num_epochs=phase_epochs.trend,
        save_suffix="phase1_trend",
    )
    trainer.train()
    trainer.save_model(os.path.join(args.save_dir, "ttm_phase1"))

    # ---------------- Phase 2: residual-only ----------------
    print("\n=== Phase 2: Residual-only ===")
    model.config.forecast_loss_type = "residual"
    set_stage_weights(model, "residual", trend=0.0, residual=1.0, joint=0.0)
    set_requires_grad(model, False)
    set_requires_grad(model.residual_forecaster, True)
    if getattr(model, "multi_quantile_head_block", None) is not None:
        set_requires_grad(model.multi_quantile_head_block, True)
    print_learnable_blocks(model)

    trainer = make_trainer(
        model=model,
        dset_train=dset_train,
        dset_val=dset_val,
        args=args,
        num_epochs=phase_epochs.residual,
        save_suffix="phase2_residual",
    )
    trainer.train()
    trainer.save_model(os.path.join(args.save_dir, "ttm_phase2"))

    # ---------------- Phase 3: joint ----------------
    print("\n=== Phase 3: Joint ===")
    model.config.forecast_loss_type = "joint"
    set_stage_weights(model, "joint", trend=0.1, residual=0.1, joint=1.0)
    set_requires_grad(model, True)
    print_learnable_blocks(model)

    trainer = make_trainer(
        model=model,
        dset_train=dset_train,
        dset_val=dset_val,
        args=args,
        num_epochs=phase_epochs.joint,
        save_suffix="phase3_joint",
    )
    trainer.train()

    model_save_path = os.path.join(args.save_dir, "ttm_pretrained")
    trainer.save_model(model_save_path)
    return model_save_path


def fmt_q(q: float) -> str:
    return f"q{int(round(q * 100)):02d}"


def pick_first_mid_last(qarr, quantile_values, idx: int, ch: int):
    quantile_values = (
        get_config_quantile_levels(q_count=qarr.shape[1])
        if quantile_values is None
        else [float(q) for q in quantile_values]
    )

    if len(quantile_values) != qarr.shape[1]:
        raise ValueError(f"len(quantile_values)={len(quantile_values)} must match Q={qarr.shape[1]}.")

    # Low/high use the first/last configured levels. The middle curve is the
    # configured quantile closest to 0.5, not a hardcoded positional assumption.
    low_i = 0
    mid_i = int(np.argmin(np.abs(np.asarray(quantile_values, dtype=np.float32) - 0.5)))
    high_i = qarr.shape[1] - 1

    return (
        float(quantile_values[low_i]),
        float(quantile_values[mid_i]),
        float(quantile_values[high_i]),
        qarr[idx, low_i, :, ch],
        qarr[idx, mid_i, :, ch],
        qarr[idx, high_i, :, ch],
    )


def plot_quantiles_first_mid_last(
    qarr,
    quantile_values,
    idx: int,
    ch: int,
    save_path: str,
    title: str,
    gt=None,
    shade_between: bool = True,
    shade_label: Optional[str] = None,
    figsize=(12, 6),
) -> None:
    q_low, q_mid, q_high, low, mid, high = pick_first_mid_last(qarr, quantile_values, idx, ch)

    plt.figure(figsize=figsize)
    plt.plot(mid, label=f"{fmt_q(q_mid)} (mid)", linewidth=1.8)
    plt.plot(low, label=f"{fmt_q(q_low)} (low)", linewidth=1.0)
    plt.plot(high, label=f"{fmt_q(q_high)} (high)", linewidth=1.0)

    if gt is not None:
        plt.plot(gt, label="Ground Truth", linewidth=1.0)

    if shade_between:
        x = np.arange(len(mid))
        if shade_label is None:
            coverage = int(round((q_high - q_low) * 100))
            shade_label = f"P{coverage} band"
        plt.fill_between(x, low, high, alpha=0.15, label=shade_label)

    plt.title(title)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def pick_q_indices(quantile_values, q_count: int, target_low: float = 0.5, target_high: float = 0.9):
    quantile_values = (
        get_config_quantile_levels(q_count=q_count) if quantile_values is None else [float(q) for q in quantile_values]
    )
    if len(quantile_values) != q_count:
        raise ValueError(f"len(quantile_values)={len(quantile_values)} must match Q={q_count}.")

    taus = np.asarray(quantile_values, dtype=np.float32)
    i_low = int(np.argmin(np.abs(taus - target_low)))
    i_high = int(np.argmin(np.abs(taus - target_high)))
    return i_low, i_high, float(taus[i_low]), float(taus[i_high])


def plot_error_quantiles_with_forecast(
    pred_err_q,
    y_pred,
    y_true,
    idx: int,
    ch: int,
    save_path: str,
    quantile_values=None,
) -> None:
    forecast = y_pred[idx, :, ch]
    gt = y_true[idx, :, ch]
    actual_err = np.abs(forecast - gt)

    i50, i90, q50_level, q90_level = pick_q_indices(quantile_values, pred_err_q.shape[1])
    err_q50 = pred_err_q[idx, i50, :, ch]
    err_q90 = pred_err_q[idx, i90, :, ch]

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(forecast, label="Forecast", linewidth=1.8)
    plt.plot(gt, label="Ground Truth", linewidth=1.8)
    plt.title("Forecast vs Ground Truth")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(actual_err, label="Actual |error|", linewidth=2)
    plt.plot(err_q50, label=f"Pred Error {fmt_q(q50_level)}", linewidth=1.5)
    plt.plot(err_q90, label=f"Pred Error {fmt_q(q90_level)}", linewidth=1.5)
    plt.title("Error Prediction (q50 & q90)")
    plt.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def compute_crps(qhat, y, quantile_values) -> float:
    quantile_values = (
        get_config_quantile_levels(q_count=qhat.shape[1])
        if quantile_values is None
        else [float(q) for q in quantile_values]
    )

    taus = np.array(quantile_values, dtype=np.float32)
    if len(taus) != qhat.shape[1]:
        raise ValueError(
            f"Mismatch between quantile count and predictions: len(quantile_values)={len(taus)}, Q={qhat.shape[1]}"
        )

    y_exp = np.expand_dims(y, axis=1)
    errors = y_exp - qhat
    taus_exp = taus.reshape(1, len(taus), 1, 1)
    pinball = np.maximum(taus_exp * errors, (taus_exp - 1.0) * errors)

    order = np.argsort(taus)
    taus_s = taus[order]
    pinball_s = pinball[:, order, :, :]

    if len(taus) == 1:
        return float(2.0 * np.mean(pinball_s))

    weights = np.zeros(len(taus), dtype=np.float32)
    weights[0] = 0.5 * (taus_s[1] - taus_s[0])
    weights[-1] = 0.5 * (taus_s[-1] - taus_s[-2])
    if len(taus) > 2:
        weights[1:-1] = 0.5 * (taus_s[2:] - taus_s[:-2])

    return float(2.0 * np.mean(pinball_s * weights.reshape(1, len(taus), 1, 1)))


def inference(args: RuntimeConfig, model_path: str, dset_test, label: str = "iid") -> None:
    model = TinyTimeMixerForDecomposedPrediction.from_pretrained(model_path)
    temp_dir = tempfile.mkdtemp()

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=temp_dir,
            per_device_eval_batch_size=args.batch_size,
            seed=args.random_seed,
            report_to="none",
        ),
    )

    print("+" * 20, "Test output", "+" * 20)
    print(trainer.evaluate(dset_test))

    predictions_dict = trainer.predict(dset_test)
    predictions = predictions_dict.predictions

    has_error_est = bool(getattr(model.config, "estimate_errors", False))
    prediction_errors = predictions[-3] if has_error_est else None
    if prediction_errors is not None:
        print("prediction_errors shape:", prediction_errors.shape)

    predictions_output = predictions[0]  # [N, F, C]
    forecast_groundtruth = predictions[-1]  # [N, F, C]
    input_data = predictions[-2]  # [N, L, C]
    trend_prediction_outputs = predictions[2]
    residual_prediction_outputs = predictions[3]
    trend_input = predictions[4]
    residual_input_t = predictions[5]

    residual_input = np.full_like(input_data, np.nan)
    residual_input[:, -residual_input_t.shape[1] :] = residual_input_t

    has_quantiles = bool(getattr(model.config, "multi_quantile_head", False))
    if has_quantiles:
        comb_q = predictions[1]
        trend_q = predictions[6]
        resid_q = predictions[7]

    print("PRED--->", predictions_output[0, 0:10, 0])
    print("GRD---->", forecast_groundtruth[0, 0:10, 0])
    print("MSE =", float(np.mean((predictions_output - forecast_groundtruth) ** 2)))

    if has_quantiles:
        quantile_values = get_config_quantile_levels(model.config, q_count=comb_q.shape[1])
        print("Quantile levels =", quantile_values)
        print("CRPS =", compute_crps(comb_q, forecast_groundtruth, quantile_values))

    save_folder = os.path.join(args.save_dir, "random_plots", label)
    os.makedirs(save_folder, exist_ok=True)

    num_samples = predictions_output.shape[0]
    num_plots = min(args.num_random_plots, num_samples)
    ch = 0

    for plot_id in range(num_plots):
        idx = random.randint(0, num_samples - 1)

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        plt.plot(input_data[idx, :, ch], label="Input", linewidth=1.5)
        plt.plot(trend_input[idx, :, ch], label="Trend Input", linewidth=1)
        plt.plot(residual_input[idx, :, ch], label="Residual Input", linewidth=1)
        plt.title(f"Inputs (Sample {idx}, Channel {ch})")
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(predictions_output[idx, :, ch], label="Forecast", linewidth=1.5)
        plt.plot(trend_prediction_outputs[idx, :, ch], label="Trend Forecast", linewidth=1)
        plt.plot(residual_prediction_outputs[idx, :, ch], label="Residual Forecast", linewidth=1)
        plt.plot(forecast_groundtruth[idx, :, ch], label="Ground Truth", linewidth=1)
        plt.title(f"Forecasts (Sample {idx}, Channel {ch})")
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, f"plot_{plot_id}.png"))
        plt.close()

        if prediction_errors is not None:
            plot_error_quantiles_with_forecast(
                pred_err_q=prediction_errors,
                y_pred=predictions_output,
                y_true=forecast_groundtruth,
                idx=idx,
                ch=ch,
                quantile_values=getattr(model.config, "err_quantiles", None)
                or (
                    get_config_quantile_levels(model.config)
                    if prediction_errors.shape[1] == len(get_config_quantile_levels(model.config))
                    else None
                ),
                save_path=os.path.join(save_folder, f"forecast_error_combined_{plot_id}.png"),
            )

        if has_quantiles:
            quantile_values = get_config_quantile_levels(model.config, q_count=comb_q.shape[1])
            plot_quantiles_first_mid_last(
                comb_q,
                quantile_values=quantile_values,
                idx=idx,
                ch=ch,
                gt=forecast_groundtruth[idx, :, ch],
                title=f"Combined Quantiles (Sample {idx}, Channel {ch})",
                save_path=os.path.join(save_folder, f"quant_combined_{plot_id}.png"),
            )
            plot_quantiles_first_mid_last(
                trend_q,
                quantile_values=quantile_values,
                idx=idx,
                ch=ch,
                gt=None,
                title=f"Trend Quantiles (Sample {idx}, Channel {ch})",
                save_path=os.path.join(save_folder, f"quant_trend_{plot_id}.png"),
            )
            plot_quantiles_first_mid_last(
                resid_q,
                quantile_values=quantile_values,
                idx=idx,
                ch=ch,
                gt=None,
                title=f"Residual Quantiles (Sample {idx}, Channel {ch})",
                save_path=os.path.join(save_folder, f"quant_residual_{plot_id}.png"),
            )

    print(f"Saved {num_plots} plots to: {save_folder}")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    set_seed(args.random_seed)

    context_length = model_cfg("context_length")
    prediction_length = model_cfg("prediction_length")

    logger.info(
        "%s Pre-training TTM | context=%s prediction=%s %s",
        "*" * 20,
        context_length,
        prediction_length,
        "*" * 20,
    )

    dset_train, dset_valid, dset_test = load_dataset(
        dataset_name=args.dataset_name,
        context_length=context_length,
        forecast_length=prediction_length,
        dataset_root_path=args.dataset_root_path,
    )

    model = build_model(debug_grad_hooks=args.debug_grad_hooks)
    model_save_path = pretrain(args, model, dset_train, dset_valid)

    print("=" * 20, "Pretraining Completed!", "=" * 20)
    print("Model saved in location:", model_save_path)

    if args.run_inference:
        inference(args=args, model_path=model_save_path, dset_test=dset_test)
        print("inference completed.")


if __name__ == "__main__":
    main()
