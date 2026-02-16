#!/usr/bin/env python
# coding: utf-8

import logging
import math
import os
import random
import tempfile

# Standard
from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.data import ConcatDataset

# First Party
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed
from tsfm_public import TimeSeriesPreprocessor, get_datasets, load_dataset

# from tsfm_public.models.tinytimemixer.utils import get_ttm_args
from tsfm_public.toolkit.get_model import get_model
from tsfm_public.toolkit.lr_finder import optimal_lr_finder
from tsfm_public.toolkit.visualization import plot_predictions

from tsfm.models.tinytimemixer import (
    TinyTimeMixerConfig,
    TinyTimeMixerForDecomposedPrediction,
)

# Third Party


logger = logging.getLogger(__file__)


import argparse

# Standard
from dataclasses import dataclass

# First Party
from transformers import TrainerCallback


def print_learnable_blocks(model):
    print("=== Learnable Blocks in Model ===")
    total_params = 0
    trainable_params = 0

    for name, module in model.named_modules():
        params = list(module.parameters(recurse=False))
        if any(p.requires_grad for p in params):
            n_params = sum(p.numel() for p in params if p.requires_grad)
            trainable_params += n_params
            print(f"[{name}] → {n_params:,} parameters")
        total_params += sum(p.numel() for p in params)

    print("\n=== Summary ===")
    print(f"Total trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")


@dataclass
class TrainingStages:
    trend_epochs: int
    residual_epochs: int
    joint_epochs: int
    # sensible defaults (use stronger raw alignment in joint)
    trend_stage_weights: tuple = (1.0, 0.0, 0.0)
    residual_stage_weights: tuple = (0.0, 1.0, 0.0)
    joint_stage_weights: tuple = (0.1, 0.1, 1)


class DecomposedTrainingStagesCallback(TrainerCallback):
    """
    Stages: trend -> residual -> joint
    Freeze logic (simple):
      - trend:    enable {trend_forecaster, multi_quantile_head?}, freeze all others
      - residual: enable {residual_forecaster, multi_quantile_head?}, freeze all others
      - joint:    enable only decoder+head of both branches (+ quantile head if present)
    """

    def __init__(self, stages: TrainingStages, verbose: bool = True):
        self.s = stages
        self.verbose = verbose
        self.b1 = stages.trend_epochs
        self.b2 = stages.trend_epochs + stages.residual_epochs
        self.last_stage = None
        self.trainer = None

    # HF wires trainer here
    def set_trainer(self, trainer):
        self.trainer = trainer
        print("[StagesCB] set_trainer called — callback is registered.")

    @staticmethod
    def _set_all(model, flag: bool):
        for p in model.parameters():
            p.requires_grad = flag

    @staticmethod
    def _set_mod(module, flag: bool):
        if module is None:
            return
        for p in module.parameters():
            p.requires_grad = flag

    @staticmethod
    def _get_path(root, dotted: str):
        obj = root
        for part in dotted.split("."):
            if not hasattr(obj, part):
                return None
            obj = getattr(obj, part)
        return obj

    def _rebuild_optim(self, stage_len_epochs: int):
        """Recreate optimizer/scheduler with ONLY current trainable params."""
        model = self.trainer.model
        args = self.trainer.args

        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            print("[StagesCB] WARNING: no trainable params when rebuilding optimizer.")

        # Respect HF args where possible
        lr = getattr(args, "learning_rate", 1e-4)
        wd = getattr(args, "weight_decay", 0.0)

        opt = AdamW(trainable, lr=lr, weight_decay=wd)
        sch = CosineAnnealingLR(opt, T_max=max(1, int(stage_len_epochs)))

        # Prevent HF Trainer from recreating/overwriting these
        self.trainer.create_optimizer = lambda *a, **k: None
        self.trainer.create_scheduler = lambda *a, **k: None
        self.trainer.optimizer = opt
        self.trainer.lr_scheduler = sch

        # Log a quick summary
        n_tr = sum(p.numel() for p in trainable)
        print(
            f"[StagesCB] Rebuilt optimizer: {len(trainable)} tensors, {n_tr:,} params | "
            f"lr={lr} wd={wd} | Cosine(T_max={int(stage_len_epochs)})"
        )

    def _enable_only_paths(self, model, dotted_paths):
        # Freeze all first
        self._set_all(model, False)
        enabled, missing = [], []
        for path in dotted_paths:
            mod = self._get_path(model, path)
            if mod is None:
                missing.append(path)
            else:
                enabled.append(path)
                self._set_mod(mod, True)

        # Logging: how many params enabled
        n_all = sum(p.numel() for p in model.parameters())
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  • Enabled: {enabled}")
        if missing:
            print(f"  • Missing (ignored): {missing}")
        print(f"  • Trainable params: {n_train:,} / {n_all:,}")
        print_learnable_blocks(model)

    def _switch(self, stage: str, *, force_log: bool = False):
        # Guard
        if self.trainer is None:
            print("[StagesCB] Warning: trainer not set yet.")
            return

        # Only log/do work if stage changes OR forced
        if stage == self.last_stage and not force_log:
            return
        self.last_stage = stage

        model = self.trainer.model
        cfg = model.config

        if stage == "trend":
            w = self.s.trend_stage_weights
            model.set_stage("trend", *w)
            cfg.forecast_loss_type = "trend"
            print(f"\n>>> ENTERING TREND STAGE (epochs 1–{self.b1})")
            self._enable_only_paths(
                model,
                [
                    "trend_forecaster",
                    "multi_quantile_head_block",  # optional
                ],
            )
            self._rebuild_optim(stage_len_epochs=self.b1)

        elif stage == "residual":
            w = self.s.residual_stage_weights
            model.set_stage("residual", *w)
            cfg.forecast_loss_type = "residual"
            print(f"\n>>> ENTERING RESIDUAL STAGE (epochs {self.b1+1}–{self.b2})")
            self._enable_only_paths(
                model,
                [
                    "residual_forecaster",
                    "multi_quantile_head_block",
                ],
            )
            self._rebuild_optim(stage_len_epochs=(self.b2 - self.b1))

        else:  # joint
            w = self.s.joint_stage_weights
            model.set_stage("joint", *w)
            cfg.forecast_loss_type = "joint"
            print(f"\n>>> ENTERING JOINT STAGE (epochs {self.b2+1}–end)")

            # self._set_all(model,True)
            self._enable_only_paths(
                model,
                [
                    "trend_forecaster.decoder",
                    "trend_forecaster.head",
                    "residual_forecaster.decoder",
                    "residual_forecaster.head",
                    "multi_quantile_head_block",  # optional
                ],
            )
            total_epochs = int(self.trainer.args.num_train_epochs)
            self._rebuild_optim(stage_len_epochs=(total_epochs - self.b2))

            print("Trainable modules in joint stage:")
            for name, p in model.named_parameters():
                if p.requires_grad:
                    print("  ", name, p.shape)

        if self.verbose:
            print(f"  → Stage tag      : {stage.upper()}")
            print(f"  → Loss weights   : trend={w[0]}  residual={w[1]}  joint={w[2]}")
            # mirror into trainer logs (TB/W&B)
            if hasattr(self.trainer, "log"):
                n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
                self.trainer.log(
                    {
                        "stage": stage,
                        "trend_loss_weight": w[0],
                        "residual_loss_weight": w[1],
                        "joint_loss_weight": w[2],
                        "trainable_params": n_tr,
                    }
                )

    # ---- HF hooks ----
    def on_init_end(self, args, state, control, **kw):
        # Fires right after Trainer is fully initialized
        print("[StagesCB] on_init_end — trainer is ready.")

    def on_train_begin(self, args, state, control, **kw):
        print("\n=== TRAINING BEGINS ===")
        self._switch("trend", force_log=True)

    def on_epoch_begin(self, args, state, control, **kw):
        # robust epoch index (state.epoch can be float or None)
        e = state.epoch
        e = 0 if e is None else int(e)
        print(f"\n[StagesCB] Epoch {e+1} begin.")
        if e < self.b1:
            self._switch("trend")
        elif e < self.b2:
            self._switch("residual")
        else:
            self._switch("joint")

    def on_log(self, args, state, control, logs=None, **kw):
        # Echo current stage+weights periodically
        if self.trainer is None:
            return
        m = self.trainer.model
        print(
            f"[StagesCB] step={state.global_step} stage={getattr(m,'current_stage','?')} "
            f"wt(tr,res,joint)=({getattr(m,'trend_loss_weight',-1)},"
            f"{getattr(m,'residual_loss_weight',-1)},"
            f"{getattr(m,'joint_loss_weight',-1)})"
        )


def get_ttm_args():
    parser = argparse.ArgumentParser(description="TinyTimeMixer Decomposed Pretraining")

    # ---------------- Data Setup ----------------
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--forecast_length", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)

    # ---------------- Optim Setup ----------------
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--random_seed", type=int, default=42)

    # ---------------- Multi-stage (NEW) ----------------
    parser.add_argument(
        "--epochs_phase1", type=int, default=None, help="Trend-only epochs"
    )
    parser.add_argument(
        "--epochs_phase2", type=int, default=None, help="Residual-only epochs"
    )
    parser.add_argument("--epochs_phase3", type=int, default=None, help="Joint epochs")

    # ---------------- Base TTM Backbone ----------------
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--head_dropout", type=float, default=0.0)
    parser.add_argument(
        "--scaling",
        type=str,
        default="revin",
        choices=[
            "mean",
            "std",
            "none",
        ],
    )

    parser.add_argument("--head_d_model", type=int, default=None)
    parser.add_argument("--trend_head_d_model", type=int, default=None)

    # Patching / token mixing
    parser.add_argument("--patch_length", type=int, default=16)
    parser.add_argument("--patch_stride", type=int, default=8)
    parser.add_argument("--adaptive_patching_levels", type=int, default=0)
    parser.add_argument("--multi_scale", action="store_true")

    parser.add_argument(
        "--register_tokens",
        type=int,
        required=False,
        default=0,
        help="Number of  register tokens",
    )

    parser.add_argument(
        "--trend_register_tokens",
        type=int,
        required=False,
        default=0,
        help="Number of  register tokens",
    )

    parser.add_argument(
        "--fft_length",
        type=int,
        required=False,
        default=0,
        help="FFT Length",
    )

    # ---------------- Decoder Settings ----------------
    parser.add_argument("--decoder_mode", type=str, default="common_channel")
    parser.add_argument("--decoder_d_model", type=int, default=32)
    parser.add_argument("--decoder_num_layers", type=int, default=2)
    parser.add_argument("--decoder_raw_residual", action="store_true")

    # ---------------- Trend Model Overrides ----------------
    parser.add_argument("--residual_context_length", type=int, default=None)
    parser.add_argument("--trend_patch_length", type=int, default=None)
    parser.add_argument("--trend_patch_stride", type=int, default=None)
    parser.add_argument("--trend_d_model", type=int, default=None)
    parser.add_argument("--trend_decoder_d_model", type=int, default=None)
    parser.add_argument("--trend_num_layers", type=int, default=None)
    parser.add_argument("--trend_decoder_num_layers", type=int, default=None)

    # ---------------- Quantile Head (Phase 4) ----------------
    parser.add_argument("--multi_quantile_head", action="store_true")

    # ---------------- Logging / Save ----------------
    parser.add_argument("--save_dir", type=str, default="./ttm_runs")
    parser.add_argument("--early_stopping", action="store_true")

    args = parser.parse_args()
    return args


# -----------------------------
# Helpers: freezing & trainer
# -----------------------------
def set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


def params_to_opt(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]


def make_trainer(
    model, dset_train, dset_val, args, lr, num_epochs, save_suffix, callback=None
):

    trainer_args = TrainingArguments(
        output_dir=os.path.join(args.save_dir, f"checkpoint_{save_suffix}"),
        overwrite_output_dir=True,
        learning_rate=lr,
        num_train_epochs=num_epochs,
        seed=args.random_seed,
        eval_strategy="epoch",  # <--- FIXED (was eval_strategy)
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        dataloader_num_workers=args.num_workers,
        ddp_find_unused_parameters=False,
        report_to="tensorboard",
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=1,
        logging_dir=os.path.join(args.save_dir, f"logs_{save_suffix}"),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,  # use bf16 mixed precision for training
        bf16_full_eval=True,  # (optional) also use bf16 for eval
    )

    optimizer = AdamW(params_to_opt(model), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    callbacks = [callback] if callback is not None else None

    return Trainer(
        model=model,
        args=trainer_args,
        train_dataset=dset_train,
        eval_dataset=dset_val,
        optimizers=(optimizer, scheduler),
        callbacks=callbacks,  # passes None when absent
    )


# -----------------------------
# Model builder
# -----------------------------
def get_base_model(args):
    # Base config (you can pass your usual args)
    config = TinyTimeMixerConfig(
        context_length=args.context_length,
        prediction_length=args.forecast_length,
        patch_length=args.patch_length,
        patch_stride=args.patch_stride,
        num_input_channels=1,
        d_model=args.d_model,
        num_layers=args.num_layers,
        mode="common_channel",
        expansion_factor=2,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        scaling=args.scaling,
        loss="mse",
        gated_attn=True,
        adaptive_patching_levels=args.adaptive_patching_levels,
        # decoder
        decoder_num_layers=args.decoder_num_layers,
        decoder_adaptive_patching_levels=0,
        decoder_mode=args.decoder_mode,
        decoder_raw_residual=False,
        use_decoder=True,
        decoder_d_model=args.decoder_d_model,
        multi_scale=args.multi_scale,
        register_tokens=args.register_tokens,
        trend_register_tokens=args.trend_register_tokens,
        fft_length=args.fft_length,
        # trend head specific overrides (if you pass them)
        residual_context_length=args.residual_context_length,
        trend_patch_length=args.trend_patch_length,
        trend_patch_stride=args.trend_patch_stride,
        trend_d_model=args.trend_d_model,
        trend_decoder_d_model=args.trend_decoder_d_model,
        trend_num_layers=args.trend_num_layers,
        trend_decoder_num_layers=args.trend_decoder_num_layers,
        # quantiles off for this 3-phase (can add Phase-4 later)
        multi_quantile_head=args.multi_quantile_head,
        head_d_model=args.head_d_model,
        trend_head_d_model=args.trend_head_d_model,
    )

    # Residual tail knob (safe default = quarter context)

    model = TinyTimeMixerForDecomposedPrediction(config)

    # (Optional) quick grad diagnostics
    for name, p in model.named_parameters():

        def hook(grad, n=name):
            if grad is None:
                print(f"[WARN] No grad for {n}")

        p.register_hook(hook)

    return model


def pretrain_simple(args, model, dset_train, dset_val):
    # You can run LR finder, but we’ll use args.learning_rate finally
    # _lr_suggest, _ = optimal_lr_finder(
    #     model,
    #     dset_train,
    #     batch_size=args.batch_size,
    # )
    # print("OPTIMAL SUGGESTED LEARNING RATE =", _lr_suggest)

    lr = args.learning_rate

    # ---------- PHASE 3: Joint (point) ----------
    print("\n=== Phase 3: Joint (point) ===")

    # # Reload from Phase 2 and turn on multi-quantile head via kwargs
    # model = TinyTimeMixerForDecomposedPrediction.from_pretrained(
    #     phase2_path,
    #     multi_quantile_head=True,          # <--- enable quantiles in config & modules
    # )

    model.config.forecast_loss_type = "joint"

    trainer = make_trainer(
        model=model,
        dset_train=dset_train,
        dset_val=dset_val,
        args=args,
        lr=lr,
        num_epochs=args.num_epochs,
        save_suffix="phase3_joint",
    )
    trainer.train()

    # Final save
    model_save_path = os.path.join(args.save_dir, "ttm_pretrained")
    trainer.save_model(model_save_path)
    return model_save_path


def pretrain_with_callback(args, model, dset_train, dset_val):
    # You can run LR finder, but we’ll use args.learning_rate finally
    # _lr_suggest, _ = optimal_lr_finder(
    #     model,
    #     dset_train,
    #     batch_size=args.batch_size,
    # )
    # print("OPTIMAL SUGGESTED LEARNING RATE =", _lr_suggest)

    lr = args.learning_rate

    stages = TrainingStages(
        trend_epochs=args.epochs_phase1,
        residual_epochs=args.epochs_phase2,
        joint_epochs=args.epochs_phase3,
        # if you want even stronger raw alignment in joint:
        # joint_stage_weights=(1.0, 1.0, 3.0)
    )
    total_epochs = args.epochs_phase1 + args.epochs_phase2 + args.epochs_phase3

    callback = DecomposedTrainingStagesCallback(stages, verbose=True)

    # ---------- PHASE 3: Joint (point) ----------

    # # Reload from Phase 2 and turn on multi-quantile head via kwargs
    # model = TinyTimeMixerForDecomposedPrediction.from_pretrained(
    #     phase2_path,
    #     multi_quantile_head=True,          # <--- enable quantiles in config & modules
    # )

    # model.config.forecast_loss_type = "joint"

    trainer = make_trainer(
        model=model,
        dset_train=dset_train,
        dset_val=dset_val,
        args=args,
        lr=lr,
        num_epochs=total_epochs,
        save_suffix="phase3_joint",
        callback=callback,
    )
    if hasattr(callback, "set_trainer"):
        callback.set_trainer(trainer)

    trainer.train()

    # Final save
    model_save_path = os.path.join(args.save_dir, "ttm_pretrained")
    trainer.save_model(model_save_path)
    return model_save_path


# -----------------------------
# Multistage training
# -----------------------------
def pretrain(args, model, dset_train, dset_val):
    # You can run LR finder, but we’ll use args.learning_rate finally
    # _lr_suggest, _ = optimal_lr_finder(
    #     model,
    #     dset_train,
    #     batch_size=args.batch_size,
    # )
    # print("OPTIMAL SUGGESTED LEARNING RATE =", _lr_suggest)

    lr = args.learning_rate

    # Allow optional per-phase epochs via args; otherwise split evenly
    e1 = getattr(args, "epochs_phase1", None)
    e2 = getattr(args, "epochs_phase2", None)
    e3 = getattr(args, "epochs_phase3", None)
    if any(x is None for x in [e1, e2, e3]):
        # simple even split (at least 1 each)
        base = max(1, args.num_epochs // 3)
        rem = max(0, args.num_epochs - 2 * base)
        e1, e2, e3 = base, base, max(1, rem)

    # ---------- PHASE 1: Trend-only ----------
    print("\n=== Phase 1: Trend-only ===")
    model.config.forecast_loss_type = "trend"

    set_requires_grad(model.trend_forecaster, True)
    set_requires_grad(model.residual_forecaster, False)
    model.trend_loss_weight = 1
    model.residual_loss_weight = 0
    model.joint_loss_weight = 0

    print_learnable_blocks(model)

    trainer = make_trainer(
        model=model,
        dset_train=dset_train,
        dset_val=dset_val,
        args=args,
        lr=lr,
        num_epochs=e1,
        save_suffix="phase1_trend",
    )
    trainer.train()
    trainer.save_model(os.path.join(args.save_dir, "ttm_phase1"))

    # ---------- PHASE 2: Residual-only ----------
    print("\n=== Phase 2: Residual-only ===")
    model.config.forecast_loss_type = "residual"

    model.trend_loss_weight = 0
    model.residual_loss_weight = 1
    model.joint_loss_weight = 0
    set_requires_grad(model.trend_forecaster, False)
    set_requires_grad(model.residual_forecaster, True)

    trainer = make_trainer(
        model=model,
        dset_train=dset_train,
        dset_val=dset_val,
        args=args,
        lr=lr,
        num_epochs=e2,
        save_suffix="phase2_residual",
    )
    trainer.train()
    phase2_path = os.path.join(args.save_dir, "ttm_phase2")
    trainer.save_model(phase2_path)

    print_learnable_blocks(model)

    # ---------- PHASE 3: Joint (point) ----------
    print("\n=== Phase 3: Joint (point) ===")

    # # Reload from Phase 2 and turn on multi-quantile head via kwargs
    # model = TinyTimeMixerForDecomposedPrediction.from_pretrained(
    #     phase2_path,
    #     multi_quantile_head=True,          # <--- enable quantiles in config & modules
    # )

    model.config.forecast_loss_type = "joint"
    model.trend_loss_weight = 0.1
    model.residual_loss_weight = 0.1
    model.joint_loss_weight = 1
    # set_requires_grad(model, True)

    # set_requires_grad(model.residual_forecaster, False)

    set_requires_grad(model, True)
    # set_requires_grad(model, False)

    # set_requires_grad(model.trend_forecaster.decoder, True)
    # set_requires_grad(model.trend_forecaster.head, True)

    # set_requires_grad(model.residual_forecaster.decoder, True)
    # set_requires_grad(model.residual_forecaster.head, True)

    # # Ensure quantile head learns
    # if getattr(model, "multi_quantile_head_block", None) is not None:
    #     set_requires_grad(model.multi_quantile_head_block, True)

    print_learnable_blocks(model)

    print("Trainable modules in joint stage:")
    for name, p in model.named_parameters():
        if p.requires_grad:
            print("  ", name, p.shape)

    # set_requires_grad(model.trend_forecaster, False)
    # set_requires_grad(model.residual_forecaster, False)

    # set_requires_grad(model.trend_forecaster.decoder, True)
    # set_requires_grad(model.trend_forecaster.head, True)

    # set_requires_grad(model.residual_forecaster.decoder, True)
    # set_requires_grad(model.residual_forecaster.head, True)

    # # Ensure quantile head learns
    # if getattr(model, "multi_quantile_head_block", None) is not None:
    #     set_requires_grad(model.multi_quantile_head_block, True)

    trainer = make_trainer(
        model=model,
        dset_train=dset_train,
        dset_val=dset_val,
        args=args,
        lr=lr,
        num_epochs=e3,
        save_suffix="phase3_joint",
    )
    trainer.train()

    # Final save
    model_save_path = os.path.join(args.save_dir, "ttm_pretrained")
    trainer.save_model(model_save_path)
    return model_save_path


# -----------------------------
# Inference + Visualization
# -----------------------------
def inference(args, model_path, dset_test, label="iid"):
    model = TinyTimeMixerForDecomposedPrediction.from_pretrained(model_path)
    print(
        model.trend_loss_weight,
        model.residual_loss_weight,
        model.joint_loss_weight,
        model.forecast_loss_type,
    )
    # print_learnable_blocks(model)
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

    print("+" * 20, "Test MSE output:", "+" * 20)
    output = trainer.evaluate(dset_test)
    print(output)

    # Predict
    predictions_dict = trainer.predict(dset_test)

    # Unpack (matches your TinyTimeMixerForDecomposedPredictionOutput ordering)
    predictions_output = predictions_dict.predictions[
        0
    ]  # [N, F, C] combined (inverse-scaled)
    trend_prediction_outputs = predictions_dict.predictions[
        1
    ]  # [N, F, C] (inverse-scaled)
    residual_prediction_outputs = predictions_dict.predictions[
        2
    ]  # [N, F, C] (inverse-scaled)
    input_data = predictions_dict.predictions[3]  # [N, L, C] (raw scale)
    trend_input = predictions_dict.predictions[4]  # [N, L, C] trend (raw scale)
    residual_input_t = predictions_dict.predictions[5]  # [N, L, C] residual (raw scale)
    forecast_groundtruth = predictions_dict.predictions[6]  # [N, F, C] (raw scale)
    L_res = residual_input_t.shape[1]
    residual_input = np.full_like(
        input_data, np.nan
    )  # fill with NaN so it does not draw

    residual_input[:, -L_res:] = residual_input_t  # right-aligned residual
    has_quantiles = model.config.multi_quantile_head

    if has_quantiles:
        trend_q = predictions_dict.predictions[7]
        resid_q = predictions_dict.predictions[8]
        comb_q = predictions_dict.predictions[9]

    print("PRED--->", predictions_output[0, 0:10, 0])
    print("GRD---->", forecast_groundtruth[0, 0:10, 0])

    mse = np.mean((predictions_output - forecast_groundtruth) ** 2)
    print("MSE =", mse)
    # Save random plots
    save_folder = os.path.join(args.save_dir, "random_plots", label)
    os.makedirs(save_folder, exist_ok=True)

    num_samples = predictions_output.shape[0]
    num_plots = min(10, num_samples)

    ch = 0  # plot first channel by default
    for i in range(num_plots):
        idx = random.randint(0, num_samples - 1)

        forecast_main = predictions_output[idx, :, ch]
        forecast_trend = trend_prediction_outputs[idx, :, ch]
        forecast_residual = residual_prediction_outputs[idx, :, ch]
        forecast_ori = forecast_groundtruth[idx, :, ch]

        input_main = input_data[idx, :, ch]
        input_trend = trend_input[idx, :, ch]

        input_residual = residual_input[idx, :, ch]

        plt.figure(figsize=(12, 8))

        # Inputs
        plt.subplot(2, 1, 1)
        plt.plot(input_main, label="Input", linewidth=1.5)
        plt.plot(input_trend, label="Trend Input", linewidth=1)
        plt.plot(input_residual, label="Residual Input", linewidth=1)
        plt.title(f"Inputs (Sample {idx}, Channel {ch})")
        plt.legend()

        # Forecasts
        plt.subplot(2, 1, 2)
        plt.plot(forecast_main, label="Forecast", linewidth=1.5)
        plt.plot(forecast_trend, label="Trend Forecast", linewidth=1)
        plt.plot(forecast_residual, label="Residual Forecast", linewidth=1)
        plt.plot(forecast_ori, label="Ground Truth", linewidth=1)
        plt.title(f"Forecasts (Sample {idx}, Channel {ch})")
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, f"plot_{i}.png"))
        plt.close()

        # ---------------- Extra quantile plots ----------------
        if has_quantiles:
            # Helper to extract q10, q50, q90 lines
            def q1090(qarr, idx, ch):
                # qarr: [N, 9, F, C]; quantile order = [0.1,...,0.9], median at index 4
                q10 = qarr[idx, 0, :, ch]
                q50 = qarr[idx, 4, :, ch]
                q90 = qarr[idx, 8, :, ch]
                return q10, q50, q90

            # --- Combined quantiles ---
            q10, q50, q90 = q1090(comb_q, idx, ch)
            plt.figure(figsize=(12, 6))
            plt.plot(q50, label="Combined q50", linewidth=1.8)
            plt.plot(q10, label="Combined q10", linewidth=1.0)
            plt.plot(q90, label="Combined q90", linewidth=1.0)
            # ground truth for reference
            plt.plot(forecast_ori, label="Ground Truth", linewidth=1.0)
            # (optional) band shading
            plt.fill_between(
                np.arange(len(q50)), q10, q90, alpha=0.15, label="P80 band"
            )
            plt.title(f"Combined Quantiles (Sample {idx}, Channel {ch})")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_folder, f"quant_combined_{i}.png"))
            plt.close()

            # --- Trend quantiles ---
            tq10, tq50, tq90 = q1090(trend_q, idx, ch)
            plt.figure(figsize=(12, 6))
            plt.plot(tq50, label="Trend q50", linewidth=1.8)
            plt.plot(tq10, label="Trend q10", linewidth=1.0)
            plt.plot(tq90, label="Trend q90", linewidth=1.0)
            # optional teacher target overlay if you keep it around
            # plt.plot(tau_tgt[idx,:,ch], label='Trend target', linewidth=1.0)
            plt.fill_between(
                np.arange(len(tq50)), tq10, tq90, alpha=0.15, label="P80 band"
            )
            plt.title(f"Trend Quantiles (Sample {idx}, Channel {ch})")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_folder, f"quant_trend_{i}.png"))
            plt.close()

            # --- Residual quantiles ---
            rq10, rq50, rq90 = q1090(resid_q, idx, ch)
            plt.figure(figsize=(12, 6))
            plt.plot(rq50, label="Residual q50", linewidth=1.8)
            plt.plot(rq10, label="Residual q10", linewidth=1.0)
            plt.plot(rq90, label="Residual q90", linewidth=1.0)
            plt.fill_between(
                np.arange(len(rq50)), rq10, rq90, alpha=0.15, label="P80 band"
            )
            plt.title(f"Residual Quantiles (Sample {idx}, Channel {ch})")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_folder, f"quant_residual_{i}.png"))
            plt.close()

    print(f"Saved {num_plots} plots to: {save_folder}")


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # Args
    args = get_ttm_args()
    set_seed(args.random_seed)

    logger.info(
        f"{'*' * 20} Pre-training a TTM for context len = {args.context_length}, forecast len = {args.forecast_length} {'*' * 20}"
    )

    # Data
    dset_train, dset_valid, dset_test = load_dataset(
        dataset_name="etth1",
        context_length=args.context_length,
        forecast_length=args.forecast_length,
        dataset_root_path="/dccstor/tsfm23/datasets",
    )

    # dset_train, dset_valid, dset_test = add_trend_to_splits(
    #     dset_train, dset_valid, dset_test,
    #     mode="linear", scale=2, per_channel=True,
    #     same_across_items=False, seed=42, augment_prob=1.0,
    #     return_original=False,   # keep original copies if you want to compare
    #     )

    # dset_train, dset_valid, dset_test = get_multi_seasonal_datasets(
    #     num_samples=12000,
    #     seq_len=args.context_length,
    #     forecast_len=args.forecast_length,
    #     enable_left_masking=args.enable_left_masking,
    #     num_classes=100,
    # )

    # dset_train, dset_valid, dset_test, dset_test_ood = get_multi_seasonal_datasets_iid_ood(num_samples_train=10000,
    #                                     num_samples_valid=1000,
    #                                     num_samples_test_iid=1000,
    #                                     num_samples_test_ood=1000,
    #                                     seq_len=args.context_length,
    #                                     forecast_len=args.forecast_length,
    #                                     enable_left_masking = False,
    #                                     num_classes=100,)

    # Model
    model = get_base_model(args)

    # # model_save_path = pretrain_with_callback(args, model, dset_train, dset_valid)
    model_save_path = pretrain(args, model, dset_train, dset_valid)
    # # model_save_path = pretrain_simple(args, model, dset_train, dset_valid)

    print("=" * 20, "Pretraining Completed!", "=" * 20)
    print("Model saved in location:", model_save_path)

    # model_save_path = "/dccstor/tsfm-irl/vijaye12/hacking/ttm_r3_models/nov11_decompose_2/ttm_quantile_1536_96_point_v2/final_model_teacher"
    # model_save_path = "/dccstor/tsfm-irl/vijaye12/hacking/ttm_r3_models/nov11_decompose_2/ttm_quantile_1536_96_point/final_model_teacher"
    # Inference + viz
    inference(args=args, model_path=model_save_path, dset_test=dset_test)
    print("inference completed..")

    # inference(args=args, model_path=model_save_path, dset_test=dset_test_ood, label = "ood")
    # print("inference ood completed..")
