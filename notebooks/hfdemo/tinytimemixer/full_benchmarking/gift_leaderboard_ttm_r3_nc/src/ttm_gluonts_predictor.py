# Copyright contributors to the TSFM project
#
"""Tools for building TTM Predictor that works with GluonTS datasets"""

import copy
import math
import os
import tempfile
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from gluonts.dataset.split import InputDataset, LabelDataset, TrainingDataset
from gluonts.ev.metrics import MASE
from gluonts.evaluation.metrics import calculate_seasonal_error
from gluonts.itertools import batcher
from gluonts.model.forecast import QuantileForecast
from scipy.stats import norm
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import ConcatDataset, DataLoader, Subset
from tqdm.auto import tqdm
from transformers import (  # [STAGED_PRETRAINING]
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from transformers.integrations import INTEGRATION_TO_CALLBACK
from transformers.utils import logging
from tsfm_public import TrackingCallback, count_parameters
from tsfm_public.toolkit.get_model import (
    TTM_LOW_RESOLUTION_MODELS_MAX_CONTEXT,
    get_model,
)
from tsfm_public.toolkit.lr_finder import optimal_lr_finder

from .gluonts_data_wrapper import (
    RESOLUTION_MAP,
    TTM_MAX_FORECAST_HORIZON,
    ForecastDataset,
    StandardScalingGluonTSDataset,
    TorchDatasetFromGluonTSTestDataset,
    TorchDatasetFromGluonTSTrainingDataset,
    get_freq_mapping,
    impute_series,
)

# from .search import build_topk_univariate_neighbor_subset
from .utils import CustomMASETrainer, plot_forecast

logger = logging.get_logger(__name__)


# TTM Constants:
# Fewshot max allowed number of samples
# This is only used when `upper_bound_fewshot_samples=True`.
# For example, if 5% few-shot for a dataset exceeds this number,
# this `FEWSHOT_MAX_NUM_SAMPLES` upper bound will be used.
FEWSHOT_MAX_NUM_SAMPLES = 500_000
VALID_MAX_NUM_SAMPLES = 50_000


def enable_norm_grads(root_module, norm_types=(nn.LayerNorm,)):
    for m in root_module.modules():  # walks all descendants
        if isinstance(m, norm_types):
            for p in m.parameters(recurse=False):
                p.requires_grad_(True)


def print_learnable_blocks_old(model):
    print("=== Parameter Breakdown by Module ===")
    total_params = 0
    trainable_params = 0
    nontrainable_params = 0

    for name, module in model.named_modules():
        params = list(module.parameters(recurse=False))
        if len(params) == 0:
            continue  # skip modules with no parameters

        # Count per block
        t_params = sum(p.numel() for p in params if p.requires_grad)
        nt_params = sum(p.numel() for p in params if not p.requires_grad)

        # Accumulate totals
        trainable_params += t_params
        nontrainable_params += nt_params
        total_params += t_params + nt_params

        # Print only if something exists in this block
        if t_params > 0 or nt_params > 0:
            print(f"[{name}] → trainable: {t_params:,} | non-trainable: {nt_params:,}")

    print("\n=== Summary ===")
    print(f"Total parameters:        {total_params:,}")
    print(f"  Trainable:             {trainable_params:,}")
    print(f"  Non-trainable:         {nontrainable_params:,}")

    pct = 100 * trainable_params / total_params if total_params > 0 else 0
    print(f"Percentage trainable:    {pct:.2f}%")


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


def enable_ft_ttm_grads(
    model,
    patch_tune=False,
    bias_tune=False,
    norm_tune=False,
    backbone_tune=False,
    prefix_tune=False,
    decoder_tune=True,
    head_tune=True,
    quantile_tune=True,
):
    """
    Enable gradient updates for selected submodules inside TTM.
    model: your TTM object (self.ttm)
    """

    freeze_all_params(model)

    if backbone_tune and hasattr(model, "backbone"):
        for p in model.backbone.parameters():
            p.requires_grad = True

    # 1. add_tokens
    if prefix_tune and hasattr(model.backbone.encoder, "add_tokens"):
        for p in model.backbone.encoder.add_tokens.parameters():
            p.requires_grad = True

    # 2. patcher if patch_tune enabled
    if patch_tune and hasattr(model.backbone.encoder, "patcher"):
        for p in model.backbone.encoder.patcher.parameters():
            p.requires_grad = True

    # 3. multi quantile head
    if quantile_tune and hasattr(model, "multi_quantile_head_block"):
        for p in model.multi_quantile_head_block.parameters():
            p.requires_grad = True

    if decoder_tune and hasattr(model, "decoder"):
        for p in model.decoder.parameters():
            p.requires_grad = True

    # 4. scaler
    if hasattr(model.backbone, "scaler"):
        for p in model.backbone.scaler.parameters():
            p.requires_grad = True

    # 5. affine quantile adapter
    if (
        hasattr(model, "affine_quantile_adapter_obj")
        and model.affine_quantile_adapter_obj is not None
    ):
        for p in model.affine_quantile_adapter_obj.parameters():
            p.requires_grad = True

    # 6. past-aware block
    if hasattr(model, "pastaware_cc_block") and model.pastaware_cc_block is not None:
        for p in model.pastaware_cc_block.parameters():
            p.requires_grad = True

    # 7. main head
    if head_tune and hasattr(model, "head"):
        for p in model.head.parameters():
            p.requires_grad = True

    # 8. bias tuning
    if bias_tune:
        for name, p in model.named_parameters():
            if ".bias" in name:
                p.requires_grad = True

    # 9. norm tuning
    if norm_tune:
        enable_norm_grads(model.backbone)

    # if hasattr(model, "tiny_adapter_head") and model.tiny_adapter_head is not None:
    #     for p in model.tiny_adapter_head.parameters():
    #         p.requires_grad = True

    if hasattr(model, "basis_calibrators") and model.basis_calibrators is not None:
        for p in model.basis_calibrators.parameters():
            p.requires_grad = True


# # [STAGED_PRETRAINING] helper to toggle grads on a module
# def set_requires_grad(module: nn.Module, requires_grad: bool):
#     for p in module.parameters():
#         p.requires_grad = requires_grad


def freeze_all_params(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False


# revision_map = {
#     "2048-96-r3": ["2560-96-r3", "3072-96-r3"],
#     "2048-720-r3": ["2560-720-r3", "3072-720-r3"],
#     "2048-96-lite-r3": ["2560-96-lite-r3", "3072-96-lite-r3"],
#     "2048-720-lite-r3": ["2560-720-lite-r3", "3072-720-lite-r3"],
# }


revision_map = {
    "156-16-dec-52-r3": ["52-16-dec-52-r3"],
    "512-30-dec-90-r3": ["180-60-dec-180-r3"],
    "768-48-dec-512-r3": ["512-48-dec-512-r3"],
    "2048-96-r3": ["1536-96-r3", "2560-96-r3", "3072-96-r3"],
    "2048-720-r3": ["1536-720-r3", "2560-720-r3", "3072-720-r3"],
    "2048-96-lite-r3": ["2560-96-lite-r3", "3072-96-lite-r3"],
    "2048-720-lite-r3": ["2560-720-lite-r3", "3072-720-lite-r3"],
}


class ZeroShotMedianAnchoredTrainer(Trainer):
    """
    Adds anchor penalty only on the 0.5 quantile forecast.

    Assumes quantile output shape:
        [B, Q, H, C]
    """

    def __init__(
        self,
        *args,
        zero_shot_model=None,
        anchor_weight=0.01,
        prediction_key="quantile_outputs",
        quantile_keys=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if zero_shot_model is None:
            raise ValueError("zero_shot_model must be provided")
        if quantile_keys is None:
            raise ValueError("quantile_keys must be provided")
        if "0.5" not in quantile_keys:
            raise ValueError("'0.5' must be present in quantile_keys")

        self.zero_shot_model = zero_shot_model
        self.anchor_weight = anchor_weight
        self.prediction_key = prediction_key
        self.quantile_keys = list(quantile_keys)
        self.median_idx = self.quantile_keys.index("0.5")

        self.zero_shot_model.eval()
        for p in self.zero_shot_model.parameters():
            p.requires_grad = False

    def _get_pred_tensor(self, outputs):
        if hasattr(outputs, self.prediction_key):
            return getattr(outputs, self.prediction_key)
        if isinstance(outputs, dict) and self.prediction_key in outputs:
            return outputs[self.prediction_key]
        raise ValueError(f"Could not find '{self.prediction_key}' in model outputs")

    def _select_q50(self, pred):
        # pred shape: [B, Q, H, C]
        return pred[:, self.median_idx, :, :]

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        with torch.no_grad():
            zs_outputs = self.zero_shot_model(**inputs)
            zs_pred = self._get_pred_tensor(zs_outputs).detach()
            zs_q50 = self._select_q50(zs_pred)

        outputs = model(**inputs)

        if hasattr(outputs, "loss"):
            base_loss = outputs.loss
        elif isinstance(outputs, dict) and "loss" in outputs:
            base_loss = outputs["loss"]
        else:
            raise ValueError("Model outputs do not contain 'loss'")

        ft_pred = self._get_pred_tensor(outputs)
        ft_q50 = self._select_q50(ft_pred)

        anchor_loss = torch.mean(torch.abs(ft_q50 - zs_q50))
        loss = base_loss + self.anchor_weight * anchor_loss

        # print(
        #     f"base_loss={base_loss.item():.6f} " f"anchor_loss={anchor_loss.item():.6f}"
        # )
        if return_outputs:
            return loss, outputs
        return loss


class TTMGluonTSPredictor:
    """Wrapper to TTM that can be directly trained, validated, and tested with GluonTS datasets."""

    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        model_path: str = "ibm-research/ttm-r3",
        test_data_label: LabelDataset = None,  # provide this for plotting
        scale: bool = False,
        random_seed: int = 42,
        term: str = None,
        ds_name: str = None,
        out_dir: str = None,
        upper_bound_fewshot_samples: bool = False,
        force_short_context: bool = False,
        min_context_mult: int = 4,
        past_feat_dynamic_real_exist: bool = False,
        num_prediction_channels: int = None,
        use_mask: bool = False,
        freq: str = None,
        use_valid_from_train: bool = True,
        insample_forecast: bool = True,
        insample_use_train: bool = False,
        plot_predictions: bool = False,
        force_zeroshot: bool = False,
        model_key: str = None,
        auto_sampler: bool = False,
        fs_mode_dict=None,
        rolling_norm: bool = False,
        ft_zs_ensemble: bool = False,
        ft_zs_ensemble_mode: str = "median",
        ensemble_shrink_lambda=0.7,  # 0 = only global, 1 = only local; tune if needed
        # ---- NEW: multi-model inputs (aligned lists) ----
        use_zs_anchor: bool = False,
        zs_anchor_weight: float = 0.01,
        zs_anchor_prediction_key: str = "quantile_outputs",
        disable_extra_point_weightage: bool = False,
        use_lite: bool = False,
        **kwargs,
    ):
        """Initialize a TTMGluonTSPredictor object."""

        self.prediction_length = prediction_length
        self.test_data_label = test_data_label
        self.scale = scale
        self.scaler = None
        self.random_seed = random_seed
        self.term = term
        self.ds_name = ds_name
        self.out_dir = out_dir
        self.upper_bound_fewshot_samples = upper_bound_fewshot_samples
        self.force_short_context = force_short_context
        self.min_context_mult = min_context_mult
        self.past_feat_dynamic_real_exist = past_feat_dynamic_real_exist
        self.num_prediction_channels = num_prediction_channels
        self.freq = freq
        self.use_mask = use_mask
        self.use_valid_from_train = use_valid_from_train
        self.insample_forecast = insample_forecast
        self.insample_use_train = insample_use_train
        self.plot_predictions = plot_predictions
        self.auto_sampler = auto_sampler
        self.fs_mode_dict = fs_mode_dict
        self.ft_zs_ensemble = ft_zs_ensemble
        self.ft_zs_ensemble_mode = ft_zs_ensemble_mode
        self.rolling_norm = rolling_norm
        self.ensemble_shrink_lambda = ensemble_shrink_lambda
        self.use_zs_anchor = use_zs_anchor
        self.zs_anchor_weight = zs_anchor_weight
        self.zs_anchor_prediction_key = zs_anchor_prediction_key
        self.ttm_zs_anchor = None
        self.disable_extra_point_weightage = disable_extra_point_weightage
        self.use_lite = use_lite

        self.quantile_keys = [
            "0.1",
            "0.2",
            "0.3",
            "0.4",
            "0.5",
            "0.6",
            "0.7",
            "0.8",
            "0.9",
            "mean",
        ]

        self.insample_errors = None
        self.force_zeroshot = force_zeroshot

        if force_short_context:
            logger.info(
                f"Forcing short context: H = {prediction_length}, CL={prediction_length * min_context_mult}"
            )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Clean kwargs
        if "dropout" in kwargs and kwargs["dropout"] is None:
            del kwargs["dropout"]
        if "head_dropout" in kwargs and kwargs["head_dropout"] is None:
            del kwargs["head_dropout"]
        if "loss" in kwargs and kwargs["loss"] is not None:
            self.loss = kwargs["loss"]
            if kwargs["loss"] not in ["mse", "mae", "huber", "pinball", "mase"]:
                del kwargs["loss"]
        if "prediction_channel_indices" in kwargs:
            self.prediction_channel_indices = kwargs["prediction_channel_indices"]

        model_keys = []
        base_model_key = self._get_gift_model(
            model_path,
            context_length,
            prediction_length,
            freq,
            return_model_key=True,
            **kwargs,
        )
        model_keys.append(base_model_key)

        if ft_zs_ensemble:
            if base_model_key in revision_map:
                model_keys.extend(revision_map[base_model_key])

        self.model_keys = model_keys
        # ---- Load ALL models into self.ttm_list ----
        self.ttm_list = []
        for mk in model_keys:
            self._get_gift_model(
                model_path,
                context_length,
                prediction_length,
                freq,
                model_revision=mk,
                return_model_key=False,
                **kwargs,
            )
            self.ttm_list.append(self.ttm.to(self.device))

        # Primary handle for legacy code paths
        self.ttm = self.ttm_list[0]
        self.model_key = self.model_keys[0]

        # ---- Keep ZS snapshots for ALL models (if enabled) ----
        self.ttm_zeroshot_list = None
        self.ttm_zeroshot = None
        if ft_zs_ensemble and not self.force_zeroshot:
            self.ttm_zeroshot_list = [copy.deepcopy(m) for m in self.ttm_list]
            self.ttm_zeroshot = self.ttm_zeroshot_list[0]

    def weighted_mean_from_stacked(self, stacked, q_low="0.1", q_high="0.9", eps=1e-6):
        """
        stacked: np.ndarray shaped [2, N, Q+1, H, C?]
                stacked[0] = forecast_ft
                stacked[1] = forecast_zs

        quantile_keys: list of strings (["0.1", ..., "0.9", "mean"])

        Returns:
            forecast_samples (np.ndarray): weighted average forecast
        """

        # Unpack
        quantile_keys = self.quantile_keys
        forecast_ft = stacked[0]
        forecast_zs = stacked[1]

        # Basic validation
        if (
            quantile_keys is None
            or q_low not in quantile_keys
            or q_high not in quantile_keys
        ):
            print("w_mean fallback → normal mean due to missing quantiles")
            return np.mean(stacked, axis=0)

        # Indices
        idx_low = quantile_keys.index(q_low)
        idx_high = quantile_keys.index(q_high)

        # Quantile widths (uncertainty signal)
        width_ft = np.abs(forecast_ft[:, idx_high] - forecast_ft[:, idx_low])  # [N,H,C]
        width_zs = np.abs(forecast_zs[:, idx_high] - forecast_zs[:, idx_low])  # [N,H,C]

        # Collapse to per-series scalar uncertainty
        # shape [N]
        w_ft_scalar = width_ft.mean(axis=tuple(range(1, width_ft.ndim)))
        w_zs_scalar = width_zs.mean(axis=tuple(range(1, width_zs.ndim)))

        # Confidence = inverse width
        conf_ft = 1.0 / (w_ft_scalar + eps)
        conf_zs = 1.0 / (w_zs_scalar + eps)

        # Normalize weights
        tot = conf_ft + conf_zs + eps
        w_ft = conf_ft / tot  # [N]
        w_zs = conf_zs / tot  # [N]

        # Broadcast to [N,1,1,...]
        shape = [forecast_ft.shape[0]] + [1] * (forecast_ft.ndim - 1)
        w_ft_b = w_ft.reshape(shape)
        w_zs_b = w_zs.reshape(shape)

        # Weighted mixture
        return w_ft_b * forecast_ft + w_zs_b * forecast_zs

    # [STAGEWISE_ES]
    def _build_trainer(
        self,
        dset_train,
        dset_valid,
        temp_dir,
        batch_size,
        learning_rate,
        num_epochs,
        num_workers,
        phase_name: str = "main",
        freq_for_mase=None,
    ):
        """
        Build a HF Trainer (or CustomMASETrainer) for a given phase.
        Shared by both staged and non-staged training.
        """
        phase_out_dir = os.path.join(temp_dir, f"output_{phase_name}")
        phase_log_dir = os.path.join(temp_dir, f"logs_{phase_name}")

        finetune_args = TrainingArguments(
            output_dir=phase_out_dir,
            overwrite_output_dir=True,
            learning_rate=learning_rate,
            num_train_epochs=num_epochs,
            do_eval=True,
            eval_strategy="epoch",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            dataloader_num_workers=num_workers,
            report_to="none",
            save_strategy="epoch",
            logging_strategy="epoch",
            save_total_limit=1,
            logging_dir=phase_log_dir,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            seed=self.random_seed,
            data_seed=self.random_seed,
            eval_accumulation_steps=10,
            disable_tqdm=True,  # hides the progress bars
            logging_steps=0,
        )

        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=5,
            early_stopping_threshold=1e-5,
        )
        tracking_callback = TrackingCallback()

        optimizer = AdamW(self.ttm.parameters(), lr=learning_rate)
        scheduler = OneCycleLR(
            optimizer,
            learning_rate,
            epochs=num_epochs,
            steps_per_epoch=max(1, math.ceil(len(dset_train) / (batch_size))),
        )

        if self.loss == "mase" and freq_for_mase is not None:
            trainer = CustomMASETrainer(
                model=self.ttm,
                args=finetune_args,
                train_dataset=dset_train,
                eval_dataset=dset_valid,
                callbacks=[early_stopping_callback, tracking_callback],
                optimizers=(optimizer, scheduler),
                freq=freq_for_mase,
            )
        else:
            if self.use_zs_anchor:
                trainer = ZeroShotMedianAnchoredTrainer(
                    model=self.ttm,
                    zero_shot_model=self.ttm_zs_anchor,
                    anchor_weight=self.zs_anchor_weight,
                    prediction_key=self.zs_anchor_prediction_key,
                    quantile_keys=self.quantile_keys,
                    args=finetune_args,
                    train_dataset=dset_train,
                    eval_dataset=dset_valid,
                    callbacks=[early_stopping_callback, tracking_callback],
                    optimizers=(optimizer, scheduler),
                )
            else:
                trainer = Trainer(
                    model=self.ttm,
                    args=finetune_args,
                    train_dataset=dset_train,
                    eval_dataset=dset_valid,
                    callbacks=[early_stopping_callback, tracking_callback],
                    optimizers=(optimizer, scheduler),
                )

        trainer.remove_callback(INTEGRATION_TO_CALLBACK["codecarbon"])
        return trainer

    # [STAGEWISE_ES]
    def _staged_train_decomposed(
        self,
        dset_train,
        dset_valid,
        temp_dir,
        batch_size,
        learning_rate,
        num_epochs,
        num_workers,
        patch_tune,
        bias_tune,
        norm_tune,
        backbone_tune,
        prefix_tune,
        decoder_tune,
        head_tune,
        quantile_tune,
        freq_for_mase=None,
        enable_staging=False,
    ):
        """
        Stage-wise training for TinyTimeMixerForDecomposedPrediction:
          Phase 1: trend-only
          Phase 2: residual-only
          Phase 3: joint

        Each phase has its own Trainer + EarlyStoppingCallback.
        """
        # --- split epochs across 3 phases ---

        if enable_staging:
            phase1_epochs = max(1, num_epochs // 3)
            phase2_epochs = max(1, num_epochs // 3)
            phase3_epochs = max(1, num_epochs - phase1_epochs - phase2_epochs)
            print(
                f"[STAGEWISE] Epoch split: trend={phase1_epochs}, residual={phase2_epochs}, joint={phase3_epochs}"
            )
        else:
            phase3_epochs = num_epochs

        last_trainer = None

        def run_phase(
            phase_name: str,
            num_phase_epochs: int,
            loss_type: str,
            trend_w: float,
            residual_w: float,
            joint_w: float,
            train_trend: bool,
            train_residual: bool,
            patch_tune,
            bias_tune,
            norm_tune,
            backbone_tune,
            prefix_tune,
            decoder_tune,
            head_tune,
            quantile_tune,
        ):
            nonlocal last_trainer

            if num_phase_epochs <= 0:
                return

            print(
                f"\n================= {phase_name} (max {num_phase_epochs} epochs) ================="
            )

            # 1) configure loss / weights
            self.ttm.config.forecast_loss_type = loss_type
            self.ttm.trend_loss_weight = trend_w
            self.ttm.residual_loss_weight = residual_w
            self.ttm.joint_loss_weight = joint_w

            # 2) freeze everything, then enable only what we want
            freeze_all_params(self.ttm)
            if train_trend:
                enable_ft_ttm_grads(
                    self.ttm.trend_forecaster,
                    patch_tune=patch_tune,
                    bias_tune=bias_tune,
                    norm_tune=norm_tune,
                    backbone_tune=backbone_tune,
                    prefix_tune=prefix_tune,
                    decoder_tune=decoder_tune,
                    head_tune=head_tune,
                    quantile_tune=quantile_tune,
                )

            if train_residual:
                enable_ft_ttm_grads(
                    self.ttm.residual_forecaster,
                    patch_tune=patch_tune,
                    bias_tune=bias_tune,
                    norm_tune=norm_tune,
                    backbone_tune=backbone_tune,
                    prefix_tune=prefix_tune,
                    decoder_tune=decoder_tune,
                    head_tune=head_tune,
                    quantile_tune=quantile_tune,
                )

            # print_learnable_blocks(self.ttm)
            # 3) build trainer for this phase
            trainer = self._build_trainer(
                dset_train=dset_train,
                dset_valid=dset_valid,
                temp_dir=temp_dir,
                batch_size=batch_size,
                learning_rate=learning_rate,
                num_epochs=num_phase_epochs,
                num_workers=num_workers,
                phase_name=phase_name,
                freq_for_mase=freq_for_mase,
            )

            # 4) run training for this phase
            trainer.train()
            last_trainer = trainer

        # --- run the 3 phases ---

        if enable_staging:
            # Phase 1: Trend-only
            run_phase(
                phase_name="phase1_trend",
                num_phase_epochs=phase1_epochs,
                loss_type="trend",
                trend_w=1.0,
                residual_w=0.0,
                joint_w=0.0,
                train_trend=True,
                train_residual=False,
                patch_tune=patch_tune,
                bias_tune=bias_tune,
                norm_tune=norm_tune,
                backbone_tune=backbone_tune,
                prefix_tune=prefix_tune,
                decoder_tune=decoder_tune,
                head_tune=head_tune,
                quantile_tune=quantile_tune,
            )

            # Phase 2: Residual-only
            run_phase(
                phase_name="phase2_residual",
                num_phase_epochs=phase2_epochs,
                loss_type="residual",
                trend_w=0.0,
                residual_w=1.0,
                joint_w=0.0,
                train_trend=False,
                train_residual=True,
                patch_tune=patch_tune,
                bias_tune=bias_tune,
                norm_tune=norm_tune,
                backbone_tune=backbone_tune,
                prefix_tune=prefix_tune,
                decoder_tune=decoder_tune,
                head_tune=head_tune,
                quantile_tune=quantile_tune,
            )

        # Phase 3: Joint
        run_phase(
            phase_name="phase3_joint",
            num_phase_epochs=phase3_epochs,
            loss_type="joint",
            trend_w=0.1,
            residual_w=0.1,
            joint_w=1.0,
            train_trend=True,
            train_residual=True,
            patch_tune=patch_tune,
            bias_tune=bias_tune,
            norm_tune=norm_tune,
            backbone_tune=backbone_tune,
            prefix_tune=prefix_tune,
            decoder_tune=decoder_tune,
            head_tune=head_tune,
            quantile_tune=quantile_tune,
        )

        return last_trainer

    def parse_model_path_revision(self, model_path: str):
        """
        Extract HF revision from model_path if provided as:
        'repo_id/revision'.

        Example:
            ibm-research/ttm-r3/2560-96-r3
            -> repo_id: ibm-research/ttm-r3
            -> kwargs: {'revision': '2560-96-r3'}

        Otherwise returns path unchanged.
        """
        prefix = "ibm-research/ttm-r3"

        if model_path.startswith(prefix + "/"):
            parts = model_path.split("/", 2)
            repo_id = "/".join(parts[:2])
            revision = parts[2]
            return repo_id, {"revision": revision}

        return model_path, {}

    def _get_gift_model(
        self,
        model_path: str,
        context_length: int,
        prediction_length: int,
        freq: str,
        return_model_key,
        **kwargs,
    ):
        """Get suitable TTM model based on context and forecast lengths.

        Args:
            model_path (str): Model card link.
            context_length (int): Context length.
        """
        self.ttm = None

        prefer_l1_loss = False
        prefer_longer_context = True
        freq_prefix_tuning = False
        force_return = "zeropad"
        if self.term == "short" and (
            str(self.freq).startswith("W")
            or str(self.freq).startswith("M")
            or str(self.freq).startswith("Q")
            or str(self.freq).startswith("A")
        ):
            prefer_l1_loss = True
            prefer_longer_context = False
            freq_prefix_tuning = True

        if self.term == "short" and str(self.freq).startswith("D"):
            prefer_l1_loss = True
            freq_prefix_tuning = True
            if context_length < 2 * TTM_LOW_RESOLUTION_MODELS_MAX_CONTEXT:
                prefer_longer_context = False
            else:
                prefer_longer_context = True

        if self.term == "short" and str(self.freq).startswith("A"):
            self.insample_use_train = False
            self.use_valid_from_train = False
            force_return = "random_init_small"

        if prediction_length > TTM_MAX_FORECAST_HORIZON:
            force_return = "rolling"

        self.ttm = get_model(
            model_path=model_path,
            context_length=context_length,
            prediction_length=prediction_length,
            freq_prefix_tuning=freq_prefix_tuning,
            freq=RESOLUTION_MAP.get(freq, "oov"),
            prefer_l1_loss=prefer_l1_loss,
            prefer_longer_context=prefer_longer_context,
            force_return=force_return,
            return_model_key=return_model_key,
            use_lite=self.use_lite,
            **kwargs,
        )

        if return_model_key:
            return self.ttm

        self.context_length = self.ttm.config.context_length

        self.enable_prefix_tuning = False
        if hasattr(self.ttm.config, "resolution_prefix_tuning"):
            self.enable_prefix_tuning = self.ttm.config.resolution_prefix_tuning
        logger.info(f"The TTM has Prefix Tuning = {self.enable_prefix_tuning}")

        return self.ttm
        # print(self.ttm)

    def _process_time_series(
        self, dataset: TrainingDataset, truncate: bool = True
    ) -> List:
        """
        Processes a time series by truncating initial NaNs and forward filling intermittent NaNs.
        Returns a new truncated dataset, and does not modify the original one.

        Args:
            dataset (TrainingDataset): Every series of of shape [channels, length].
            truncate (bool, optional): Truncate the dataset if True. Defaults to True.

        Returns:
            List: Processed time series, each of shape [channels, truncated_length].
        """
        truncated_dataset = list(copy.deepcopy(dataset))
        for i, item in enumerate(truncated_dataset):
            if "target" not in item:
                continue
            data = item["target"]

            if data.ndim == 1:
                data = data.reshape(1, -1)  # [channels, length]

            if self.past_feat_dynamic_real_exist:
                if item["past_feat_dynamic_real"].ndim == 1:
                    item["past_feat_dynamic_real"] = item[
                        "past_feat_dynamic_real"
                    ].reshape(1, -1)
                data = np.vstack((data, item["past_feat_dynamic_real"]))

            truncated_dataset[i]["target"] = data
            if not truncate:
                continue

            # Step 1: Determine the longest stretch of initial NaNs across all channels
            valid_mask = ~np.isnan(data)  # Mask of valid (non-NaN) values
            if valid_mask.all():
                continue  # Continue if no NaN

            first_valid = np.argmax(
                valid_mask.any(axis=0)
            )  # First col with any valid value across channels
            data = data[:, first_valid:]  # Truncate cols before the first valid col
            # Step 2: Perform forward fill for NaNs
            df = pd.DataFrame(data.T, columns=range(data.shape[0]))
            df = df.ffill(axis=0)

            data = df.values.T
            if data.shape[0] == 1:  # [1, truncated_length]
                data = data.reshape(-1)  # [truncated_length]
            truncated_dataset[i]["target"] = data
        return truncated_dataset

    def compute_quantile_forecasts(self, loader, quantiles):
        all_quantile_forecasts = []

        for batch in tqdm(loader, desc="Processing Batches"):
            forecast_samples, insample_errors, point_forecasts = batch

            insample_errors[insample_errors == 0] = 1e-5  # To prevent division by zero

            # Expand scales for quantiles
            batch_size, seq_len, no_channels = forecast_samples.shape
            num_quantiles = len(quantiles)

            scales = np.expand_dims(
                insample_errors, axis=1
            )  # Shape: (batch_size, 1, H, C)
            scales = np.tile(
                scales, (1, num_quantiles, 1, 1)
            )  # Shape: (batch_size, num_quantiles, H, C)

            # Expand quantiles
            quantiles_expanded = np.reshape(
                quantiles, (1, num_quantiles, 1, 1)
            )  # Shape: (1, num_quantiles, 1, 1)
            quantiles_expanded = np.tile(
                quantiles_expanded, (batch_size, 1, seq_len, no_channels)
            )  # Shape: (batch_size, num_quantiles, H, C)

            # Expand forecasts
            forecasts_expanded = np.expand_dims(
                forecast_samples, axis=1
            )  # Shape: (batch_size, 1, H, C)
            forecasts_expanded = np.tile(
                forecasts_expanded, (1, num_quantiles, 1, 1)
            )  # Shape: (batch_size, num_quantiles, H, C)

            # Compute quantile forecasts
            quantile_forecasts = norm.ppf(
                quantiles_expanded, loc=forecasts_expanded, scale=scales
            )  # Shape: (batch_size, num_quantiles, H, C)

            # Append point forecasts
            final_forecasts = np.concatenate(
                (quantile_forecasts, point_forecasts), axis=1
            )  # Shape: (batch_size, num_quantiles+1, H, C)

            # Collect results for the batch
            all_quantile_forecasts.extend(final_forecasts)

        return all_quantile_forecasts

    def train(
        self,
        train_dataset: TrainingDataset,
        valid_dataset: TrainingDataset,
        batch_size: int = 64,
        optimize_batch_size: bool = True,
        learning_rate: float = None,
        num_epochs: int = 30,
        num_workers: int = 8,
        fewshot_fraction: int = 1.0,
        automate_fewshot_fraction: bool = True,
        automate_fewshot_fraction_threshold: int = 200,
        fewshot_location: str = "rand",  # rand/start/end
        save_model: bool = False,
        test_dataset: TrainingDataset = None,
        search_filter=False,
        plot: bool = False,
        ds_config=None,
        fewshot_max_samples=None,
        valid_max_samples=None,
        bias_tune: bool = False,
        norm_tune: bool = False,
        patch_tune: bool = False,
        backbone_tune: bool = False,
        prefix_tune: bool = False,
        decoder_tune: bool = True,
        head_tune: bool = True,
        quantile_tune: bool = True,
        enable_staging: bool = False,
    ):
        """
        Finetune ALL TTMs in self.ttm_list one-by-one using the exact same dataset/logic.
        Adds finetune logging per model (success/failure, samples, error).
        """
        # ---- NEW: init per-run logs ----

        self.num_epochs = num_epochs
        if hasattr(self, "_init_run_logs"):
            self._init_run_logs()

        # -----------------------------
        # Auto sampler settings
        # -----------------------------
        if self.auto_sampler and self.fs_mode_dict is not None:
            if self.ds_name in self.fs_mode_dict:
                fs_mode = self.fs_mode_dict[self.ds_name]
                if fs_mode == "1K":
                    valid_max_samples = 1000
                    fewshot_max_samples = 1000
                elif fs_mode == "10K":
                    valid_max_samples = 1000
                    fewshot_max_samples = 10000
                else:
                    valid_max_samples = None
                    fewshot_max_samples = None
                print(
                    "----------> Resetting val, few samples",
                    valid_max_samples,
                    fewshot_max_samples,
                )

        # -----------------------------
        # Preprocess / scale (ONCE)
        # -----------------------------
        train_dataset_scaled = self._process_time_series(train_dataset)
        valid_dataset_scaled = self._process_time_series(valid_dataset)
        logger.info(
            f"Number of series: Train = {len(train_dataset_scaled)}, Valid = {len(valid_dataset_scaled)}"
        )

        if self.scale:
            self.scaler = StandardScalingGluonTSDataset()
            self.scaler.fit(train_dataset_scaled)
            train_dataset_scaled = self.scaler.transform(train_dataset_scaled)
            valid_dataset_scaled = self.scaler.transform(valid_dataset_scaled)
            logger.info("Global scaling done successfully.")

        temp_dir = tempfile.mkdtemp()

        # Create "actual" train dataset (used for fewshot indexing)
        dset_train_actual = TorchDatasetFromGluonTSTrainingDataset(
            train_dataset_scaled,
            self.context_length,
            self.prediction_length,
            force_short_context=self.force_short_context,
            min_context_mult=self.min_context_mult,
            send_freq=self.enable_prefix_tuning,
            freq=self.freq,
        )

        if automate_fewshot_fraction:
            fewshot_data_size = int(fewshot_fraction * len(dset_train_actual))
            if (
                fewshot_data_size < automate_fewshot_fraction_threshold
                and fewshot_data_size > 10
            ):
                fewshot_fraction = 0.9
                num_epochs = 50
                logger.info(
                    f"Increasing fewshot fraction to {fewshot_fraction} due to small dataset size."
                )

        dset_valid_from_train = None

        if fewshot_max_samples is None:
            fewshot_max_samples = FEWSHOT_MAX_NUM_SAMPLES

        # -----------------------------
        # Build dset_train / dset_valid (ONCE)
        # -----------------------------
        if fewshot_fraction < 1.0:
            if fewshot_location == "rand":
                rng = np.random.default_rng(seed=self.random_seed)
                if self.upper_bound_fewshot_samples:
                    list_size = min(
                        int(fewshot_fraction * len(dset_train_actual)),
                        fewshot_max_samples,
                    )
                else:
                    list_size = int(fewshot_fraction * len(dset_train_actual))

                lst_fewshot_indx = rng.integers(
                    low=0,
                    high=len(dset_train_actual),
                    size=list_size,
                )

                dset_train = Subset(dset_train_actual, lst_fewshot_indx)
                logger.info(f"Length of orginal train set = {len(dset_train_actual)}")
                logger.info(
                    f"Length of {fewshot_fraction * 100} % train set = {len(dset_train)}"
                )

                if self.use_valid_from_train:
                    all_indx = list(range(0, len(dset_train_actual)))
                    valid_indx = list(set(all_indx) - set(lst_fewshot_indx))

                    if valid_max_samples is None:
                        valid_max_samples = VALID_MAX_NUM_SAMPLES

                    valid_size = min(
                        valid_max_samples, len(dset_train_actual) - len(dset_train)
                    )
                    valid_indx = np.random.choice(valid_indx, valid_size, replace=False)
                    dset_valid_from_train = Subset(dset_train_actual, valid_indx)

            elif fewshot_location in ("end", "start"):
                dset_train = TorchDatasetFromGluonTSTrainingDataset(
                    train_dataset_scaled,
                    self.context_length,
                    self.prediction_length,
                    fewshot_fraction=fewshot_fraction,
                    fewshot_location=fewshot_location,
                    force_short_context=self.force_short_context,
                    min_context_mult=self.min_context_mult,
                    send_freq=self.enable_prefix_tuning,
                    freq=self.freq,
                )
            else:
                raise ValueError("Wrong fewshot_location.")
        else:
            logger.info("Using 100% train data to finetune the model.")
            dset_train = TorchDatasetFromGluonTSTrainingDataset(
                train_dataset_scaled,
                self.context_length,
                self.prediction_length,
                force_short_context=self.force_short_context,
                min_context_mult=self.min_context_mult,
                send_freq=self.enable_prefix_tuning,
                freq=self.freq,
            )

        dset_valid: TorchDatasetFromGluonTSTrainingDataset = (
            TorchDatasetFromGluonTSTrainingDataset(
                valid_dataset_scaled,
                self.context_length,
                self.prediction_length,
                last_window_only=True,
                gen_more_samples_for_short_series=False,
                force_short_context=self.force_short_context,
                min_context_mult=self.min_context_mult,
                send_freq=self.enable_prefix_tuning,
                freq=self.freq,
            )
        )
        # dset_valid_actual = copy.deepcopy(dset_valid)

        if dset_valid_from_train is not None:
            dset_train = ConcatDataset((dset_train, dset_valid))
            dset_valid = dset_valid_from_train

        self.train_num_samples = len(dset_train)
        self.valid_num_samples = len(dset_valid)
        logger.info(
            f"Number of train samples = {self.train_num_samples}, valid samples = {self.valid_num_samples}"
        )

        if self.force_zeroshot or self.train_num_samples <= 10:
            print("train_num_samples", self.train_num_samples)
            raise Exception(
                "Forcing zeroshot since number of finetune samples is very low or force_zerohot is set to true"
            )

        # MASE freq lookup (ONCE)
        freq_ = None
        if getattr(self, "loss", None) == "mase":
            underlying_dataset = train_dataset.dataset.iterable
            first_entry = next(iter(underlying_dataset))
            freq_ = first_entry["freq"]

        # -----------------------------
        # Loop over ALL models
        # -----------------------------
        trained_models = []
        last_insample_errors = None

        for mi, model in enumerate(self.ttm_list):
            # Switch active model
            self.ttm = model

            if self.use_zs_anchor:
                self.ttm_zs_anchor = copy.deepcopy(self.ttm)
                self.ttm_zs_anchor.eval()
                for p in self.ttm_zs_anchor.parameters():
                    p.requires_grad = False

            self.model_key = (
                self.model_keys[mi] if hasattr(self, "model_keys") else self.model_key
            )
            mk = self.model_key if self.model_key is not None else f"model_{mi}"
            model_id = f"FT::{mk}"

            logger.info(
                f"\n================= FT MODEL {mi + 1}/{len(self.ttm_list)}: {mk} ================="
            )

            finetuned_ok = False
            err_msg = None

            try:
                # ---- enable grads (same logic as before) ----
                if (
                    self.ttm.__class__.__name__
                    == "TinyTimeMixerForDecomposedPrediction"
                ):
                    enable_ft_ttm_grads(
                        model=self.ttm.trend_forecaster,
                        patch_tune=patch_tune,
                        bias_tune=bias_tune,
                        norm_tune=norm_tune,
                        backbone_tune=backbone_tune,
                        prefix_tune=prefix_tune,
                        decoder_tune=decoder_tune,
                        head_tune=head_tune,
                        quantile_tune=quantile_tune,
                    )
                    enable_ft_ttm_grads(
                        model=self.ttm.residual_forecaster,
                        patch_tune=patch_tune,
                        bias_tune=bias_tune,
                        norm_tune=norm_tune,
                        backbone_tune=backbone_tune,
                        prefix_tune=prefix_tune,
                        decoder_tune=decoder_tune,
                        head_tune=head_tune,
                        quantile_tune=quantile_tune,
                    )
                else:
                    enable_ft_ttm_grads(
                        model=self.ttm,
                        patch_tune=patch_tune,
                        bias_tune=bias_tune,
                        norm_tune=norm_tune,
                        backbone_tune=backbone_tune,
                        prefix_tune=prefix_tune,
                        decoder_tune=decoder_tune,
                        head_tune=head_tune,
                        quantile_tune=quantile_tune,
                    )

                logger.info(
                    f"Number of params after freezing the model = {count_parameters(self.ttm)}",
                )

                # ---- optimize batch size ----
                batch_size_i = batch_size
                if optimize_batch_size:
                    if self.ttm.config.num_input_channels < 10:
                        batch_size_i = 64
                    else:
                        batch_size_i = 16

                    if len(dset_train) <= 1_000:
                        batch_size_i = 8
                    elif len(dset_train) > 100_000:
                        batch_size_i = 512

                    logger.info(
                        f"Using a batch size of {batch_size_i}, based on number of training samples = {len(dset_train)} and number of channels = {self.ttm.config.num_input_channels}."
                    )

                # ---- learning rate per model ----
                lr_i = learning_rate
                if lr_i is None:
                    lr_i, self.ttm = optimal_lr_finder(
                        self.ttm,
                        dset_train,
                        batch_size=batch_size_i,
                        enable_prefix_tuning=self.enable_prefix_tuning,
                    )
                    logger.info(f"OPTIMAL SUGGESTED LEARNING RATE = {lr_i}")
                logger.info(f"Using learning rate = {lr_i}")

                # print_learnable_blocks(self.ttm)

                # ---- build trainer (staged for decomposed) ----
                if (
                    self.ttm.__class__.__name__
                    == "TinyTimeMixerForDecomposedPrediction"
                ):
                    hf_trainer = self._staged_train_decomposed(
                        dset_train=dset_train,
                        dset_valid=dset_valid,
                        temp_dir=temp_dir,
                        batch_size=batch_size_i,
                        learning_rate=lr_i,
                        num_epochs=num_epochs,
                        num_workers=num_workers,
                        freq_for_mase=freq_,
                        patch_tune=patch_tune,
                        bias_tune=bias_tune,
                        norm_tune=norm_tune,
                        backbone_tune=backbone_tune,
                        prefix_tune=prefix_tune,
                        decoder_tune=decoder_tune,
                        head_tune=head_tune,
                        quantile_tune=quantile_tune,
                        enable_staging=enable_staging,
                    )
                else:
                    hf_trainer = self._build_trainer(
                        dset_train=dset_train,
                        dset_valid=dset_valid,
                        temp_dir=temp_dir,
                        batch_size=batch_size_i,
                        learning_rate=lr_i,
                        num_epochs=num_epochs,
                        num_workers=num_workers,
                        phase_name="main",
                        freq_for_mase=freq_,
                    )

                # For non-decomposed models we call train() here; decomposed staged trainer already runs phases.
                if (
                    self.ttm.__class__.__name__
                    != "TinyTimeMixerForDecomposedPrediction"
                ):
                    hf_trainer.train()

                # Save model (optional)
                if save_model:
                    if self.out_dir is not None:
                        save_path = os.path.join(self.out_dir, str(mk), "ttm_model")
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        hf_trainer.save_model(save_path)
                    else:
                        raise ValueError(
                            "`out_dir` should not be `None` when `save_model=True`."
                        )

                # In-sample stats (per model)
                if self.insample_forecast:
                    self.insample_errors = self.get_insample_stats(
                        hf_trainer=hf_trainer,
                        dset_valid=dset_valid,
                        dset_train=dset_train,
                        global_scaler=self.scaler,
                        use_train=self.insample_use_train,
                        search_filter=search_filter,
                        test_dataset=test_dataset,
                        plot=plot,
                        ds_config=ds_config,
                    )
                    last_insample_errors = self.insample_errors

                finetuned_ok = True

            except Exception as e:
                finetuned_ok = False
                err_msg = str(e)
                logger.exception(
                    f"[Train] Finetune failed for model={mk}. Falling back to keeping it as-is."
                )

            # ---- NEW: log finetune outcome ----
            if hasattr(self, "_append_finetune_log"):
                self._append_finetune_log(
                    model_id=model_id,
                    model_key=mk,
                    index=mi,
                    success=finetuned_ok,
                    train_num_samples=int(getattr(self, "train_num_samples", -1)),
                    valid_num_samples=int(getattr(self, "valid_num_samples", -1)),
                    error=err_msg,
                )

            trained_models.append(self.ttm)

        # -----------------------------
        # restore lists + primary handle
        # -----------------------------
        self.ttm_list = trained_models
        self.ttm = self.ttm_list[0]
        self.model_key = (
            self.model_keys[0] if hasattr(self, "model_keys") else self.model_key
        )

        if last_insample_errors is not None:
            self.insample_errors = last_insample_errors

        # Optional: print summary
        if hasattr(self, "print_finetune_log"):
            self.print_finetune_log()

    def get_channels(self, dataset):
        return dataset[0]["past_values"].shape[1]

    def get_insample_stats(
        self,
        hf_trainer,
        dset_valid,
        dset_train,
        global_scaler,
        batch_size=4096,
        use_train=False,
        search_filter=False,
        test_dataset=None,
        plot=False,
        ds_config=None,
    ):
        if len(dset_valid) == 1:
            dataset = ConcatDataset((dset_train, dset_valid))
        elif use_train:
            max_subset_size = 50_000
            if len(dset_train) > max_subset_size:
                rng = np.random.default_rng(seed=self.random_seed)
                lst_fewshot_indx = rng.integers(
                    low=0,
                    high=len(dset_train),
                    size=max_subset_size,
                )
                dset_train_subset = Subset(dset_train, lst_fewshot_indx)
                dataset = ConcatDataset((dset_train_subset, dset_valid))
            else:
                dataset = ConcatDataset((dset_train, dset_valid))
        else:
            dataset = dset_valid

        # print(
        #     "Train Samples before search filtering --->",
        #     len(dataset) * self.get_channels(dataset),
        # )

        while True:
            try:
                if batch_size == 0:
                    raise Exception(
                        "Model and data too big, do not fit into GPU even with batch size 1."
                    )

                logger.info(f"[InSampleStat] Trying batch size {batch_size}")
                # Get ground truth stacked
                dl_valid = DataLoader(dataset, batch_size=batch_size, shuffle=False)

                y_true = []
                series_ids = []
                for batch in dl_valid:
                    y_true.append(batch["future_values"].detach().cpu().numpy())
                    series_ids.extend(batch["item_id"])
                y_true = np.concatenate(y_true)

                if self.scale:
                    y_true_unscaled = global_scaler.inverse_transform(
                        y_true, series_ids
                    )
                else:
                    y_true_unscaled = y_true

                # Change batch size dynamically
                hf_trainer.args.per_device_eval_batch_size = batch_size
                # Force dataloader to rebuild with new batch size
                hf_trainer._eval_dataloader = None

                # Get validation predictions
                valid_preds_out = hf_trainer.predict(dataset)
                y_pred = valid_preds_out.predictions[0]

                if self.scale:
                    y_pred_unscaled = global_scaler.inverse_transform(
                        y_pred, series_ids
                    )
                else:
                    y_pred_unscaled = y_pred

                # Create a pands dataframe
                # Flatten (H, C) into 2D arrays
                # L = y_pred_unscaled.shape[0]
                flattened_predictions = list(y_pred_unscaled)
                flattened_ground_truth = list(y_true_unscaled)

                df = pd.DataFrame(
                    {
                        "item_id": series_ids,
                        "y_true": flattened_ground_truth,
                        "y_pred": flattened_predictions,
                    }
                )

                df["errors"] = (
                    df["y_true"] - df["y_pred"]
                ).abs()  # absolute error pointwise
                errors = df.groupby(by="item_id")[
                    "errors"
                ].mean()  # mean over all samples from a particular series

                logger.info("Successfully, calculated the in-sample statistics.")
                break

            except torch.cuda.OutOfMemoryError:
                logger.warning(
                    f"[InSampleStat] OutOfMemoryError at batch_size {batch_size}, reducing to {batch_size // 2}"
                )
                batch_size //= 2

        return errors

    def _init_run_logs(self):
        """
        Initializes per-run logs. Call at the start of train() and predict().
        """
        self.finetune_log = {
            "models": [],  # list[dict]
        }
        self.ensemble_log = {
            "models": [],  # list[dict] in stacking order
            "mode": None,  # str
            "weights": None,  # dict/list with summaries
            "extra": None,  # optional debug info
        }

    def _append_finetune_log(
        self,
        model_id: str,
        model_key: str,
        index: int,
        success: bool,
        train_num_samples: int,
        valid_num_samples: int,
        error: str = None,
    ):
        """
        Record finetune status for one model.
        """
        if not hasattr(self, "finetune_log") or self.finetune_log is None:
            self._init_run_logs()

        self.finetune_log["models"].append(
            {
                "model_id": model_id,  # e.g. "FT::decv3-52-16"
                "model_key": model_key,
                "index": int(index),
                "success": bool(success),
                "train_num_samples": int(train_num_samples),
                "valid_num_samples": int(valid_num_samples),
                "error": error,
            }
        )

    def _set_ensemble_log(self, model_ids, mode: str, weights=None, extra=None):
        """
        model_ids: list[str] in the same order as stacked_all[0..M-1]
        mode: str (ft_zs_ensemble_mode)
        weights:
        - mean/median -> list[float] length M (equal weights)
        - w_mean -> np.ndarray [M, N] or summary dict
        - best-of -> None (use extra to store counts)
        """
        if not hasattr(self, "ensemble_log") or self.ensemble_log is None:
            self._init_run_logs()

        self.ensemble_log["mode"] = mode
        self.ensemble_log["models"] = [{"id": mid} for mid in model_ids]

        # Make weights readable + compact
        if weights is None:
            self.ensemble_log["weights"] = None
        else:
            if isinstance(weights, list):
                self.ensemble_log["weights"] = weights
            else:
                w = np.asarray(weights)
                if w.ndim == 2:
                    # per-series weights: [M, N]
                    self.ensemble_log["weights"] = {
                        "type": "per_series",
                        "shape": list(w.shape),
                        "mean_per_model": w.mean(axis=1).tolist(),
                        "min_per_model": w.min(axis=1).tolist(),
                        "max_per_model": w.max(axis=1).tolist(),
                    }
                else:
                    self.ensemble_log["weights"] = {
                        "type": "array",
                        "shape": list(w.shape),
                    }

        self.ensemble_log["extra"] = extra

        # Print compact log for visibility
        # print("\n[EnsembleLog]")
        # print("  mode:", mode)
        # print("  models:", model_ids)
        # if self.ensemble_log["weights"] is not None:
        #     print("  weights_summary:", self.ensemble_log["weights"])
        # if extra is not None:
        #     print("  extra:", extra)

    def print_finetune_log(self):
        """
        Optional helper to print finetune results.
        """
        if not hasattr(self, "finetune_log") or self.finetune_log is None:
            print("[FinetuneLog] <empty>")
            return
        print("\n[FinetuneLog]")
        for m in self.finetune_log.get("models", []):
            print(" ", m)

    def get_test_dataset(
        self,
        test_dataset,
    ):
        test_dataset_input = test_dataset.input
        test_dataset_label = test_dataset.label

        test_dataset_scaled = self._process_time_series(test_dataset_input)
        if self.scale:
            if self.scaler is None:
                self.scaler = StandardScalingGluonTSDataset()
                self.scaler.fit(test_dataset_scaled)

            test_dataset_scaled = self.scaler.transform(test_dataset_scaled)
        else:
            test_dataset_scaled = test_dataset_input

        dset_test = TorchDatasetFromGluonTSTestDataset(
            test_dataset_scaled,
            test_dataset_label,
            self.context_length,
            self.prediction_length,
            force_short_context=self.force_short_context,
            min_context_mult=self.min_context_mult,
        )

        return dset_test

    def rolling_backtest_weights(
        self,
        test_data_input_scaled,
        model_ft,
        model_zs,
        context_length: int,
        prediction_length: int,
        device,
        freq=None,
        use_mask: bool = False,
        max_windows: int = 3,
        eps: float = 1e-6,
    ):
        """
        Compute per-series weights for FT and ZS via rolling backtests
        over the full context.

        For each series:
        - Ensure length >= context_length + k by zero-prepending if needed.
        - Split history into up to `max_windows` [context, target] windows.
        - Context length <= `context_length`
        - Target length = k (<= prediction_length and per-model prediction_length)
        - Run both models on each window, compute MAE vs target.
        - Aggregate errors across windows (median) → per-series err_ft, err_zs.
        - Convert to weights w_ft, w_zs ∈ [0,1] with local+global shrinkage.

        Returns:
            w_ft, w_zs as 1D arrays of shape [N].
            (Broadcasting is done in predict().)
        """

        # Safety: use same k for both models, never exceeding their single-shot horizon
        single_h_ft = getattr(model_ft.config, "prediction_length", prediction_length)
        single_h_zs = getattr(model_zs.config, "prediction_length", prediction_length)
        k = int(min(prediction_length, single_h_ft, single_h_zs))
        if k <= 0:
            raise ValueError(f"Invalid backtest forecast length k={k}")

        N = len(test_data_input_scaled)
        err_ft = np.zeros(N, dtype=np.float64)
        err_zs = np.zeros(N, dtype=np.float64)

        # For detailed logs
        all_series_err_ft = []
        all_series_err_zs = []
        all_windows_used = []

        # Optional freq mapping (only if needed)
        freq_map = None
        if freq is not None and (
            getattr(model_ft.config, "resolution_prefix_tuning", False)
            or getattr(model_zs.config, "resolution_prefix_tuning", False)
        ):
            freq_map = get_freq_mapping()

        for i, entry in enumerate(test_data_input_scaled):
            item_id = entry.get("item_id", i)

            data = entry["target"]
            if data.ndim == 1:
                data = data.reshape(1, -1)  # [C, T]
            data = impute_series(data)
            C, T = data.shape

            # --- ensure minimum length via zero-prepending ---
            if T < context_length + k:
                required = (context_length + k) - T
                logger.debug(
                    f"[RBTest] item={item_id} (idx={i}): "
                    f"T={T} too short for context_length={context_length}+k={k}. "
                    f"Prepending {required} zeros."
                )
                pad = np.zeros((C, required), dtype=data.dtype)
                data = np.concatenate([pad, data], axis=1)
                T = data.shape[1]

            series_err_ft = []
            series_err_zs = []
            windows_used = 0

            # Now T >= context_length + k, so at least one window is possible
            max_possible = (T - context_length) // k
            num_windows = int(min(max_windows, max_possible)) or 1

            for w in range(num_windows):
                # w = 0 → last k points (closest to test forecast)
                # w = 1 → k points before that, etc.
                forecast_end = T - w * k
                forecast_start = forecast_end - k
                context_end = forecast_start
                context_start = max(0, context_end - context_length)

                ctx_len = context_end - context_start
                if ctx_len < max(8, k // 2):  # avoid too tiny contexts
                    logger.debug(
                        f"[RBTest] item={item_id} (idx={i}), window={w}: ctx_len={ctx_len} too small, skipping window."
                    )
                    continue

                ctx = data[:, context_start:context_end]  # [C, ctx_len]
                tgt = data[:, forecast_start:forecast_end]  # [C, k]

                batch_ttm = {
                    "past_values": torch.tensor(ctx, dtype=torch.float32)
                    .T.unsqueeze(0)
                    .to(device),  # [1, ctx_len, C]
                    "return_loss": False,
                }

                if use_mask:
                    mask = torch.ones_like(batch_ttm["past_values"], dtype=torch.bool)
                    batch_ttm["past_observed_mask"] = mask

                if freq_map is not None:
                    batch_ttm["freq_token"] = (
                        torch.ones((1), dtype=torch.long, device=device)
                        * freq_map[freq]
                    )

                with torch.no_grad():
                    # ----- FT -----
                    out_ft = model_ft(**batch_ttm)
                    if getattr(model_ft.config, "multi_quantile_head", False):
                        y_ft = out_ft["quantile_outputs"][:, 4, :k, :]  # [1, k, C]
                    else:
                        y_ft = out_ft["prediction_outputs"][:, :k, :]  # [1, k, C]
                    y_ft_np = y_ft.squeeze(0).cpu().numpy().T  # [C, k]
                    e_ft = np.mean(np.abs(y_ft_np - tgt))
                    series_err_ft.append(e_ft)

                    # ----- ZS -----
                    out_zs = model_zs(**batch_ttm)
                    if getattr(model_zs.config, "multi_quantile_head", False):
                        y_zs = out_zs["quantile_outputs"][:, 4, :k, :]
                    else:
                        y_zs = out_zs["prediction_outputs"][:, :k, :]
                    y_zs_np = y_zs.squeeze(0).cpu().numpy().T
                    e_zs = np.mean(np.abs(y_zs_np - tgt))
                    series_err_zs.append(e_zs)

                    windows_used += 1

            if len(series_err_ft) == 0:
                # Extremely rare now (e.g. pathological tiny series)
                logger.warning(
                    f"[RBTest] item={item_id} (idx={i}): "
                    "No valid backtest windows even after padding. "
                    "Using dummy errors 1e3 for both FT and ZS."
                )
                series_err_ft = [1e3]
                series_err_zs = [1e3]
                windows_used = 0

            # Aggregate per-series errors
            err_ft[i] = np.median(series_err_ft)
            err_zs[i] = np.median(series_err_zs)

            all_series_err_ft.append(series_err_ft)
            all_series_err_zs.append(series_err_zs)
            all_windows_used.append(windows_used)

        # ---------- LOCAL weights from per-series errors ----------
        conf_ft = 1.0 / (err_ft + eps)
        conf_zs = 1.0 / (err_zs + eps)

        w_ft_local = conf_ft / (conf_ft + conf_zs + eps)
        w_zs_local = conf_zs / (conf_ft + conf_zs + eps)

        # ---------- GLOBAL weight from median errors across series ----------
        global_err_ft = np.median(err_ft)
        global_err_zs = np.median(err_zs)

        global_conf_ft = 1.0 / (global_err_ft + eps)
        global_conf_zs = 1.0 / (global_err_zs + eps)

        w_ft_global = global_conf_ft / (global_conf_ft + global_conf_zs + eps)
        w_zs_global = global_conf_zs / (global_conf_ft + global_conf_zs + eps)

        logger.debug(
            f"[RBTest] GLOBAL: median_err_ft={global_err_ft:.4f}, "
            f"median_err_zs={global_err_zs:.4f}, "
            f"w_ft_global={w_ft_global:.4f}, w_zs_global={w_zs_global:.4f}"
        )

        # ---------- shrink LOCAL weights toward GLOBAL ----------
        lambda_shrink = self.ensemble_shrink_lambda

        w_ft = lambda_shrink * w_ft_local + (1.0 - lambda_shrink) * w_ft_global
        w_zs = lambda_shrink * w_zs_local + (1.0 - lambda_shrink) * w_zs_global

        # normalize just in case of tiny numeric drift
        denom = w_ft + w_zs + eps
        w_ft /= denom
        w_zs /= denom

        # ---- Logging summary per series ----
        for i, entry in enumerate(test_data_input_scaled):
            item_id = entry.get("item_id", i)
            ft_list = all_series_err_ft[i]
            zs_list = all_series_err_zs[i]
            n_win = all_windows_used[i]
            logger.info(
                f"[RBTest] item={item_id} (idx={i}): "
                f"windows_used={n_win}, "
                f"ft_err_list={ft_list}, "
                f"zs_err_list={zs_list}, "
                f"final_err_ft={err_ft[i]:.4f}, "
                f"final_err_zs={err_zs[i]:.4f}, "
                f"w_ft={w_ft[i]:.4f}, w_zs={w_zs[i]:.4f}"
            )

        # Return 1D weights; broadcasting happens in predict()
        return w_ft, w_zs

    def validate(
        self,
        valid_dataset: TrainingDataset,
        batch_size: int = 64,
    ):
        """(Optionally) Validate.

        Args:
            valid_dataset (TrainingDataset): Validation dataset.
            batch_size (int, optional): Batch size. Defaults to 64.

        Returns:
            flat: Validation loss.
        """
        valid_dataset_scaled = self._process_time_series(valid_dataset)
        if self.scale:
            if self.scaler is None:
                self.scaler = StandardScalingGluonTSDataset()
                self.scaler.fit(valid_dataset_scaled)

            valid_dataset_scaled = self.scaler.transform(valid_dataset_scaled)
        else:
            valid_dataset_scaled = valid_dataset

        temp_dir = tempfile.mkdtemp()
        dset_valid = TorchDatasetFromGluonTSTrainingDataset(
            valid_dataset_scaled,
            self.context_length,
            self.prediction_length,
            last_window_only=True,
            force_short_context=self.force_short_context,
            min_context_mult=self.min_context_mult,
            send_freq=self.enable_prefix_tuning,
            freq=self.freq,
        )

        # hf_trainer
        hf_trainer = Trainer(
            model=self.ttm,
            args=TrainingArguments(
                output_dir=temp_dir,
                per_device_eval_batch_size=batch_size,
                report_to="none",
                eval_accumulation_steps=10,
                seed=self.random_seed,
                data_seed=self.random_seed,
            ),
        )

        # evaluate = zero-shot performance
        print("+" * 20, "Zero-shot Test Loss", "+" * 20)
        zeroshot_output = hf_trainer.predict(dset_valid)
        print(zeroshot_output)
        return zeroshot_output["eval_loss"]

    def rolling_backtest_weights_all_models(
        self,
        test_data_input_scaled,
        models,
        context_length: int,
        prediction_length: int,
        device,
        freq=None,
        use_mask: bool = False,
        max_windows: int = 3,
        eps: float = 1e-6,
    ):
        """
        Rolling backtest across ALL models together.

        For each series i:
        - Build up to `max_windows` windows: [context, target] of length k
        - Run EACH model on each window, compute MAE vs target
        - Aggregate per-series per-model error (median over windows)
        - Convert errors to per-series weights via inverse-error confidence
        - Apply global shrinkage: lambda * local + (1-lambda) * global

        Args:
            test_data_input_scaled: list of dicts with "target" and optional "item_id"
            models: list of torch models
            context_length: int
            prediction_length: int
            device: torch.device / str
            freq: str (optional)
            use_mask: bool
            max_windows: int
            eps: float

        Returns:
            W: np.ndarray shape [M, N] (per-model per-series weights, normalized per series)
            info: dict with debug summaries
        """

        mase_metric = MASE()

        M = len(models)
        N = len(test_data_input_scaled)
        if M == 0:
            raise ValueError("models list is empty")

        # Safety: common k across all models
        single_h = [
            getattr(m.config, "prediction_length", prediction_length) for m in models
        ]
        k = int(min(prediction_length, min(single_h)))
        if k <= 0:
            raise ValueError(f"Invalid backtest forecast length k={k}")

        ERR = np.zeros((M, N), dtype=np.float64)
        WIN_USED = np.zeros((M, N), dtype=np.int32)

        # Optional freq mapping
        freq_map = None
        if freq is not None:
            freq_map = get_freq_mapping()

        for i, entry in enumerate(test_data_input_scaled):
            item_id = entry.get("item_id", i)

            data = entry["target"]
            if data.ndim == 1:
                data = data.reshape(1, -1)  # [C, T]
            data = impute_series(data)
            C, T = data.shape

            # Ensure min length via zero-prepending
            data_i = []
            T_i = []
            cl_i = []
            for mi, model in enumerate(models):
                cl_i.append(getattr(model.config, "context_length", context_length))
                if T < cl_i[-1] + k:
                    required = (cl_i[-1] + k) - T
                    logger.debug(
                        f"[RBTest-ALL] item={item_id} (idx={i}): "
                        f"T={T} too short for context_length={cl_i[-1]}+k={cl_i[-1] + k}. "
                        f"Prepending {required} zeros."
                    )
                    pad = np.zeros((C, required), dtype=data.dtype)
                    data_i.append(np.concatenate([pad, data], axis=1))
                    T_i.append(data_i[-1].shape[1])
                else:
                    data_i.append(data)
                    T_i.append(data.shape[1])

            # Now T >= context_length + k
            max_possible_per_model = [
                max(0, (T_i[mi] - cl_i[mi]) // k) for mi in range(M)
            ]
            max_possible = min(max_possible_per_model)
            num_windows = int(min(max_windows, max_possible)) or 1

            # We'll accumulate errors per model
            per_model_err_lists = [[] for _ in range(M)]
            windows_used = 0

            for w in range(num_windows):
                with torch.no_grad():
                    for mi, model in enumerate(models):
                        forecast_end = T_i[mi] - w * k
                        forecast_start = forecast_end - k
                        context_end = forecast_start
                        context_start = max(0, context_end - cl_i[mi])

                        ctx_len = context_end - context_start
                        if ctx_len < max(8, k // 2):
                            continue

                        ctx = data_i[mi][:, context_start:context_end]  # [C, ctx_len]
                        tgt = data_i[mi][:, forecast_start:forecast_end]  # [C, k]

                        # maybe can remove
                        ctx_len = ctx.shape[1]
                        if ctx_len < cl_i[mi]:
                            need = cl_i[mi] - ctx_len
                            pad = np.zeros((ctx.shape[0], need), dtype=ctx.dtype)
                            ctx = np.concatenate([pad, ctx], axis=1)
                        elif ctx_len > cl_i[mi]:
                            ctx = ctx[:, -cl_i[mi] :]

                        batch_ttm = {
                            "past_values": torch.tensor(ctx, dtype=torch.float32)
                            .T.unsqueeze(0)
                            .to(device),  # [1, ctx_len, C]
                            "return_loss": False,
                        }

                        if use_mask:
                            batch_ttm["past_observed_mask"] = torch.ones_like(
                                batch_ttm["past_values"], dtype=torch.bool
                            )

                        if freq_map is not None:
                            # Only if model expects it
                            # (we check per-model below)
                            pass

                        batch = batch_ttm

                        if freq_map is not None and getattr(
                            model.config, "resolution_prefix_tuning", False
                        ):
                            # clone minimal dict to avoid overwriting
                            batch = dict(batch_ttm)
                            batch["freq_token"] = (
                                torch.ones((1), dtype=torch.long, device=device)
                                * freq_map[freq]
                            )

                        out = model(**batch)
                        if getattr(model.config, "multi_quantile_head", False):
                            y = out["quantile_outputs"][:, 4, :k, :]  # [1, k, C]
                        else:
                            y = out["prediction_outputs"][:, :k, :]  # [1, k, C]

                        y_np = y.squeeze(0).cpu().numpy().T  # [C, k]
                        # e = np.mean(np.abs(y_np - tgt))

                        # seasonal error
                        se = calculate_seasonal_error(ctx.reshape(-1), freq=freq)

                        # mase
                        m_eval = mase_metric()
                        m_eval.update(
                            {
                                "label": tgt.reshape(-1),
                                "0.5": y_np.reshape(-1),
                                "seasonal_error": se,
                            }
                        )
                        e = float(m_eval.get())

                        per_model_err_lists[mi].append(e)

                windows_used += 1

            # Aggregate errors per model for this series
            for mi in range(M):
                if len(per_model_err_lists[mi]) == 0:
                    # rare path
                    ERR[mi, i] = 1e3
                else:
                    ERR[mi, i] = np.median(per_model_err_lists[mi])
                WIN_USED[mi, i] = windows_used

        # ---- LOCAL weights: inverse error ----
        CONF = 1.0 / (ERR + eps)  # [M, N]
        W_local = CONF / (CONF.sum(axis=0, keepdims=True) + eps)

        # ---- GLOBAL weights: median error per model ----
        global_err = np.median(ERR, axis=1)  # [M]
        global_conf = 1.0 / (global_err + eps)
        W_global = global_conf / (global_conf.sum() + eps)  # [M]

        # automatic per-series shrinkage (Empirical Bayes)
        diff = W_local - W_global[:, None]  # [M, N]
        V = np.mean(diff**2, axis=0)  # [N]

        windows_per_series = np.maximum(WIN_USED.max(axis=0), 1)
        err_var = np.var(ERR, axis=0)  # [N]
        sigma2 = err_var / windows_per_series
        # sigma2 = err_var

        lambda_i = V / (V + sigma2 + eps)  # [N]

        # ---- shrink local toward global ----
        lambda_shrink = self.ensemble_shrink_lambda
        # W = lambda_shrink * W_local + (1.0 - lambda_shrink) * W_global[:, None]
        W = (lambda_i[None, :] * W_local) + (
            (1.0 - lambda_i)[None, :] * W_global[:, None]
        )

        # normalize again
        W /= W.sum(axis=0, keepdims=True) + eps

        info = {
            "k": k,
            "max_windows": max_windows,
            "shrink_lambda": float(lambda_shrink),
            "global_err": global_err.tolist(),
            "global_w": W_global.tolist(),
            "windows_used_mean_per_model": WIN_USED.mean(axis=1).tolist(),
            "err_mean_per_model": ERR.mean(axis=1).tolist(),
            "err_median_per_model": np.median(ERR, axis=1).tolist(),
        }

        return W, info

    def predict(
        self,
        test_data_input: InputDataset,
        batch_size: int = 64,
    ):
        """Predict on test data.

        Runs predictions for ALL finetuned models in self.ttm_list, and (optionally)
        ALL zeroshot copies in self.ttm_zeroshot_list. Then applies ft_zs_ensemble_mode
        ONCE across the entire pool (FT + ZS together), while logging model participation
        and weights.
        """

        # ---- NEW: init per-run logs ----
        if hasattr(self, "_init_run_logs"):
            self._init_run_logs()

        test_data_input_scaled = self._process_time_series(
            test_data_input, truncate=False
        )

        if self.scale:
            if self.rolling_norm:
                self.scaler = StandardScalingGluonTSDataset()
                self.scaler.fit(test_data_input_scaled)
            test_data_input_scaled = self.scaler.transform(test_data_input_scaled)

        test_data_input_scaled = list(test_data_input_scaled)

        # -------- helper: full predict pipeline for a single model --------
        def _single_model_predict(model, test_data_input_scaled, batch_size):
            test_data_input_scaled_copy = copy.deepcopy(test_data_input_scaled)
            local_bs = batch_size

            while True:
                try:
                    print("bs = ", local_bs)
                    forecast_samples = []
                    series_ids = []

                    for batch in tqdm(
                        batcher(test_data_input_scaled, batch_size=local_bs)
                    ):
                        batch_ttm = {}
                        adjusted_batch_raw = []
                        past_observed_mask = []

                        for idx, entry in enumerate(batch):
                            series_ids.append(entry["item_id"])

                            if len(entry["target"].shape) == 1:
                                entry["target"] = entry["target"].reshape(1, -1)

                            if self.force_short_context:
                                entry["target"] = entry["target"][
                                    :, -self.min_context_mult * self.prediction_length :
                                ]

                            entry_context_length = entry["target"].shape[1]
                            num_channels = entry["target"].shape[0]

                            # Pad
                            if entry_context_length < model.config.context_length:
                                logger.debug("Using zero filling for padding.")
                                padding = torch.zeros(
                                    (
                                        num_channels,
                                        model.config.context_length
                                        - entry_context_length,
                                    )
                                )
                                adjusted_entry = torch.cat(
                                    (
                                        padding,
                                        torch.tensor(impute_series(entry["target"])),
                                    ),
                                    dim=1,
                                )
                                mask = torch.ones(adjusted_entry.shape)
                                mask[:, : padding.shape[1]] = 0

                            # Truncate
                            elif entry_context_length > model.config.context_length:
                                adjusted_entry = torch.tensor(
                                    impute_series(
                                        entry["target"][
                                            :, -model.config.context_length :
                                        ]
                                    )
                                )
                                mask = torch.ones(adjusted_entry.shape)

                            else:
                                adjusted_entry = torch.tensor(
                                    impute_series(entry["target"])
                                )
                                mask = torch.ones(adjusted_entry.shape)

                            adjusted_batch_raw.append(adjusted_entry)
                            past_observed_mask.append(mask.bool())

                        # --- pad to common length across batch if needed ---
                        lengths = [t.shape[1] for t in adjusted_batch_raw]
                        max_len = max(lengths)

                        if len(set(lengths)) > 1:
                            for i, t in enumerate(adjusted_batch_raw):
                                cur_len = t.shape[1]
                                if cur_len < max_len:
                                    pad_len = max_len - cur_len
                                    pad = torch.zeros(
                                        t.shape[0],
                                        pad_len,
                                        dtype=t.dtype,
                                        device=t.device,
                                    )
                                    adjusted_batch_raw[i] = torch.cat([pad, t], dim=1)

                            for i, m in enumerate(past_observed_mask):
                                cur_len = m.shape[-1]
                                if cur_len < max_len:
                                    pad_len = max_len - cur_len
                                    pad_shape = list(m.shape)
                                    pad_shape[-1] = pad_len
                                    pad = torch.zeros(
                                        pad_shape,
                                        dtype=m.dtype,
                                        device=m.device,
                                    )
                                    past_observed_mask[i] = torch.cat([pad, m], dim=-1)

                        batch_ttm["past_values"] = (
                            torch.stack(adjusted_batch_raw)
                            .permute(0, 2, 1)
                            .to(self.device)
                        )

                        if self.use_mask:
                            batch_ttm["past_observed_mask"] = (
                                torch.stack(past_observed_mask)
                                .permute(0, 2, 1)
                                .to(self.device)
                            )

                        if model.config.resolution_prefix_tuning:
                            freq_map = get_freq_mapping()
                            batch_ttm["freq_token"] = (
                                torch.ones((batch_ttm["past_values"].shape[0]))
                                * freq_map[self.freq]
                            ).to(self.device)

                        # -------- long horizon / recursive case --------
                        if self.prediction_length > TTM_MAX_FORECAST_HORIZON:
                            batch_ttm["return_loss"] = False

                            recursive_steps = int(
                                np.ceil(
                                    self.prediction_length
                                    / model.config.prediction_length
                                )
                            )

                            if (
                                hasattr(model.config, "multi_quantile_head")
                                and model.config.multi_quantile_head
                            ):
                                predict_outputs = torch.empty(
                                    len(batch), 9, 0, num_channels
                                ).to(self.device)
                            else:
                                predict_outputs = torch.empty(
                                    len(batch), 0, num_channels
                                ).to(self.device)

                            with torch.no_grad():
                                for _ in range(recursive_steps):
                                    model_outputs = model(**batch_ttm)

                                    if (
                                        hasattr(model.config, "multi_quantile_head")
                                        and model.config.multi_quantile_head
                                    ):
                                        batch_ttm["past_values"] = torch.cat(
                                            [
                                                batch_ttm["past_values"],
                                                model_outputs["quantile_outputs"][
                                                    :, 4, ...
                                                ],
                                            ],
                                            dim=1,
                                        )[:, -model.config.context_length :, :]
                                    else:
                                        batch_ttm["past_values"] = torch.cat(
                                            [
                                                batch_ttm["past_values"],
                                                model_outputs["prediction_outputs"],
                                            ],
                                            dim=1,
                                        )[:, -model.config.context_length :, :]

                                    if self.use_mask:
                                        batch_ttm["past_observed_mask"] = torch.cat(
                                            [
                                                batch_ttm["past_observed_mask"],
                                                torch.ones(
                                                    model_outputs[
                                                        "prediction_outputs"
                                                    ].shape
                                                )
                                                .bool()
                                                .to(self.device),
                                            ],
                                            dim=1,
                                        )[:, -model.config.context_length :, :]

                                    if (
                                        hasattr(model.config, "multi_quantile_head")
                                        and model.config.multi_quantile_head
                                    ):
                                        predict_outputs = torch.cat(
                                            [
                                                predict_outputs,
                                                model_outputs["quantile_outputs"][
                                                    ...,
                                                    : model.config.prediction_length,
                                                    :,
                                                ],
                                            ],
                                            dim=2,
                                        )
                                    else:
                                        predict_outputs = torch.cat(
                                            [
                                                predict_outputs,
                                                model_outputs["prediction_outputs"][
                                                    :,
                                                    : model.config.prediction_length,
                                                    :,
                                                ],
                                            ],
                                            dim=1,
                                        )

                            predict_outputs = predict_outputs[
                                ..., : self.prediction_length, :
                            ]

                        # -------- short horizon / single-shot case --------
                        else:
                            model_outputs = model(**batch_ttm)
                            if (
                                hasattr(model.config, "multi_quantile_head")
                                and model.config.multi_quantile_head
                            ):
                                predict_outputs = model_outputs.quantile_outputs
                            else:
                                predict_outputs = model_outputs.prediction_outputs

                        forecast_samples.append(predict_outputs.detach().cpu().numpy())

                    if len(forecast_samples) == 0:
                        logger.warning(
                            "No batches produced in _single_model_predict; returning empty forecasts."
                        )
                        return np.array([]), series_ids

                    forecast_samples = np.concatenate(forecast_samples)

                    if self.scale:
                        if self.past_feat_dynamic_real_exist:
                            forecast_samples = self.scaler.inverse_transform(
                                forecast_samples,
                                series_ids,
                                self.prediction_channel_indices,
                            )
                        else:
                            forecast_samples = self.scaler.inverse_transform(
                                forecast_samples, series_ids
                            )

                    if self.prediction_length > TTM_MAX_FORECAST_HORIZON:
                        forecast_samples = forecast_samples[
                            ..., : self.num_prediction_channels
                        ]

                    # --- in-sample driven quantiles (no multi-quantile head) ---
                    if self.insample_forecast and (
                        not hasattr(model.config, "multi_quantile_head")
                        or (
                            hasattr(model.config, "multi_quantile_head")
                            and not model.config.multi_quantile_head
                        )
                    ):
                        point_forecasts = np.expand_dims(forecast_samples, 1)
                        self.quantiles = np.array(
                            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                        )

                        b, seq_len, no_channels = forecast_samples.shape

                        if self.insample_errors is None:
                            dummy_errors_ = []
                            unq_series_ids = list(np.unique(series_ids))
                            for _ in unq_series_ids:
                                dummy_errors_.append(np.ones((seq_len, no_channels)))
                            self.insample_errors = pd.DataFrame(
                                {"item_id": unq_series_ids, "errors": dummy_errors_}
                            ).set_index("item_id")["errors"]
                            logger.warning(
                                "`insample_errors` is `None`. Using a dummy error of `np.ones()`"
                            )

                        if (
                            self.insample_errors.iloc[0].shape[0]
                            < self.prediction_length
                        ):
                            for i in range(len(self.insample_errors)):
                                self.insample_errors.iloc[i] = np.concatenate(
                                    (
                                        self.insample_errors.iloc[i],
                                        self.insample_errors.iloc[i][
                                            -(
                                                self.prediction_length
                                                - self.insample_errors.iloc[i].shape[0]
                                            ) :,
                                            :,
                                        ],
                                    )
                                )

                        dataset = ForecastDataset(
                            forecast_samples,
                            series_ids,
                            self.insample_errors,
                            point_forecasts,
                            self.quantiles,
                        )
                        dataloader = DataLoader(
                            dataset,
                            batch_size=local_bs,
                            shuffle=False,
                            collate_fn=lambda x: (
                                np.stack([i[0] for i in x]),
                                np.stack([i[1] for i in x]),
                                np.stack([i[2] for i in x]),
                            ),
                        )

                        all_quantile_forecasts = self.compute_quantile_forecasts(
                            dataloader, self.quantiles
                        )

                    # --- direct multi-quantile head case ---
                    if (
                        hasattr(model.config, "multi_quantile_head")
                        and model.config.multi_quantile_head
                    ):
                        all_quantile_forecasts = np.concatenate(
                            (
                                forecast_samples,
                                np.expand_dims(forecast_samples[:, 4, :, :], axis=1),
                            ),
                            axis=1,
                        )

                    forecast_samples_final = np.array(all_quantile_forecasts)
                    if forecast_samples_final.shape[-1] == 1:
                        forecast_samples_final = np.squeeze(
                            forecast_samples_final, axis=-1
                        )

                    return forecast_samples_final, series_ids

                except torch.cuda.OutOfMemoryError:
                    print(
                        f"OutOfMemoryError at batch_size {local_bs}, reducing to {local_bs // 2}"
                    )
                    local_bs //= 2
                    test_data_input_scaled = copy.deepcopy(test_data_input_scaled_copy)

        # -------- Collect forecasts for ALL models (FT + optional ZS) --------
        forecasts_all = []
        model_ids = []
        series_ids = None

        # Predict from Base models
        for mi, model in enumerate(self.ttm_list):
            mk = self.model_keys[mi] if hasattr(self, "model_keys") else None
            mid = f"FT::{mk}" if mk is not None else f"FT::model_{mi}"
            logger.debug(f"[Predict] {mid}")

            f, ids = _single_model_predict(model, test_data_input_scaled, batch_size)
            if f.size > 0:
                forecasts_all.append(f)
                model_ids.append(mid)

            if series_ids is None:
                series_ids = ids

        # Ensemble with Zeroshot models if base models were finetuned and ensemble enabled
        if (
            self.ft_zs_ensemble
            and hasattr(self, "ttm_zeroshot_list")
            and self.ttm_zeroshot_list is not None
        ):
            for mi, model in enumerate(self.ttm_zeroshot_list):
                mk = self.model_keys[mi] if hasattr(self, "model_keys") else None
                mid = f"ZS::{mk}" if mk is not None else f"ZS::model_{mi}"
                logger.debug(f"[Predict] {mid}")

                f, _ = _single_model_predict(model, test_data_input_scaled, batch_size)
                if f.size > 0:
                    forecasts_all.append(f)
                    model_ids.append(mid)

        if len(forecasts_all) == 0:
            raise RuntimeError("No forecasts produced by any model.")

        stacked_all = np.stack(forecasts_all, axis=0)  # [M, N, Q+1, H, C?]
        M = stacked_all.shape[0]
        num_series = len(test_data_input)
        logger.info(
            f"[Predict] Dataset={self.ds_name}, Num series={num_series}, NM={M}"
        )

        # -------- Ensemble across ALL models --------
        if self.ft_zs_ensemble_mode == "median" or num_series > 50:
            # Median only over primary FT and primary ZS (the "first items")
            # Ordering invariant:
            #   model_ids: [FT::main, FT::mapped..., ZS::main, ZS::mapped...]
            ft0 = 0
            zs0 = (
                len(self.ttm_list)
                if (self.ft_zs_ensemble and self.ttm_zeroshot_list)
                else None
            )

            if zs0 is not None and zs0 < M:
                idxs = [ft0, zs0]
                stacked_2 = stacked_all[idxs]  # [2, N, Q+1, H, C?]
                forecast_samples = np.median(stacked_2, axis=0)

                # log weights only for those 2 models (others are effectively 0)
                weights = [0.0] * M
                weights[ft0] = 0.5
                weights[zs0] = 0.5

                if hasattr(self, "_set_ensemble_log"):
                    self._set_ensemble_log(
                        model_ids=model_ids,
                        mode="median_primary_ft_zs",
                        weights=weights,
                        extra={"used_model_ids": [model_ids[ft0], model_ids[zs0]]},
                    )

                # print("median aggregate across PRIMARY FT+ZS only")

            else:
                # fallback: only FT primary exists
                forecast_samples = stacked_all[ft0]  # [N, Q+1, H, C?]
                weights = [0.0] * M
                weights[ft0] = 1.0

                if hasattr(self, "_set_ensemble_log"):
                    self._set_ensemble_log(
                        model_ids=model_ids,
                        mode="primary_ft_only",
                        weights=weights,
                        extra={"used_model_ids": [model_ids[ft0]]},
                    )

                # print("primary FT only (no ZS available)")

        elif self.ft_zs_ensemble_mode == "mean":
            forecast_samples = np.mean(stacked_all, axis=0)
            equal_w = [1.0 / M] * M
            if hasattr(self, "_set_ensemble_log"):
                self._set_ensemble_log(
                    model_ids=model_ids, mode="mean", weights=equal_w, extra=None
                )
            print("mean aggregate across ALL models")

        elif self.ft_zs_ensemble_mode == "w_mean":
            # Multi-model uncertainty-weighted mean using quantile width (0.1..0.9)
            q_low, q_high, eps = "0.1", "0.9", 1e-6
            qk = self.quantile_keys
            if q_low not in qk or q_high not in qk:
                logger.warning("w_mean fallback → normal mean due to missing quantiles")
                forecast_samples = np.mean(stacked_all, axis=0)
                equal_w = [1.0 / M] * M
                if hasattr(self, "_set_ensemble_log"):
                    self._set_ensemble_log(
                        model_ids=model_ids,
                        mode="w_mean_fallback_mean",
                        weights=equal_w,
                        extra=None,
                    )
            else:
                idx_low = qk.index(q_low)
                idx_high = qk.index(q_high)

                # width: [M, N, H, C?]
                width = np.abs(stacked_all[:, :, idx_high] - stacked_all[:, :, idx_low])

                # scalar uncertainty per (model, series): [M, N]
                width_scalar = width.mean(axis=tuple(range(2, width.ndim)))
                conf = 1.0 / (width_scalar + eps)  # [M, N]
                w = conf / (
                    conf.sum(axis=0, keepdims=True) + eps
                )  # normalize over M per series

                # broadcast to stacked_all rank: [M, N, 1, 1, ...]
                w_b = w.reshape([M, w.shape[1]] + [1] * (stacked_all.ndim - 2))
                forecast_samples = (w_b * stacked_all).sum(axis=0)

                if hasattr(self, "_set_ensemble_log"):
                    self._set_ensemble_log(
                        model_ids=model_ids, mode="w_mean", weights=w, extra=None
                    )

            print("weighted mean aggregate across ALL models")

        elif self.ft_zs_ensemble_mode == "backtest_mean":
            models_all = self.ttm_list + (self.ttm_zeroshot_list or [])

            W, info = self.rolling_backtest_weights_all_models(
                test_data_input_scaled=test_data_input_scaled,
                models=models_all,
                context_length=self.context_length,
                prediction_length=self.prediction_length,
                device=self.device,
                freq=self.freq,
                use_mask=self.use_mask,
                max_windows=3,
                eps=1e-6,
            )

            # Weighted aggregation
            M = stacked_all.shape[0]
            N = stacked_all.shape[1]
            W_b = W.reshape([M, N] + [1] * (stacked_all.ndim - 2))
            forecast_samples = (W_b * stacked_all).sum(axis=0)
            # print("forecast_samples", forecast_samples)
            # Logging
            if hasattr(self, "_set_ensemble_log"):
                self._set_ensemble_log(
                    model_ids=model_ids,
                    mode="backtest_mean_all_models",
                    weights=W,
                    extra=info,
                )

            # print("rolling backtest weighted ensemble across ALL models")

        else:
            raise Exception("Invalid ft_zs_ensemble_mode")

        # -------- convert to GluonTS QuantileForecast objects --------
        sample_forecasts = []
        for item, ts in zip(forecast_samples, test_data_input):
            forecast_start_date = ts["start"] + len(ts["target"])
            sample_forecasts.append(
                QuantileForecast(
                    forecast_arrays=item,
                    start_date=forecast_start_date,
                    forecast_keys=self.quantile_keys,
                    item_id=ts["item_id"],
                )
            )

        if self.plot_predictions:
            plot_forecast(
                test_data_input,
                self.test_data_label,
                forecast_samples,
                self.prediction_length,
                self.ds_name,
                self.term,
                self.out_dir,
                probabilistic=self.insample_forecast,
                quantile_keys=self.quantile_keys,
            )

        return sample_forecasts
