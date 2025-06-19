# Copyright authors of TSPulse: Dual Space Tiny Pre-trained Models for Rapid Time-series Analysis

from typing import Optional, Union

import numpy as np
import torch
from transformers.utils import logging


logger = logging.get_logger(__name__)


def _reduce(metric, reduction="mean", axis=None):
    if reduction == "mean":
        return np.nanmean(metric, axis=axis)
    elif reduction == "sum":
        return np.nansum(metric, axis=axis)
    elif reduction == "none":
        return metric


def mse(
    y: np.ndarray,
    y_hat: np.ndarray,
    reduction: str = "mean",
    axis: Optional[int] = None,
) -> Union[float, np.ndarray]:
    delta_y = np.square(y - y_hat)
    return _reduce(delta_y, reduction=reduction, axis=axis)


def mask_contiguous_with_token(tensor, mask_percentage, patch_length, generator):
    patch_length = 8
    b, s, c = tensor.shape
    num_patches = s // patch_length

    if s % patch_length != 0:
        raise ValueError("Sequence length (s) must be divisible by the patch length (K).")

    tensor = tensor.transpose(-1, -2)  # b, c, s

    num_patches_to_mask_per_sample = int(num_patches * mask_percentage)

    rand_indices = torch.rand((b, c, num_patches), device=tensor.device, generator=generator).argsort(dim=2)

    mask_patches = torch.ones((b, c, num_patches), dtype=torch.bool, device=tensor.device)

    batch_mask = torch.ones(b, dtype=torch.bool, device=tensor.device)

    mask_indices = rand_indices[:, :, :num_patches_to_mask_per_sample]
    mask_patches[batch_mask] = mask_patches[batch_mask].scatter(-1, mask_indices[batch_mask], False)

    patch_mask = mask_patches.unsqueeze(-1).expand(-1, -1, -1, patch_length)

    mask = ~patch_mask.reshape(b, c, s // patch_length, patch_length).permute(0, 2, 3, 1).reshape(b, s, c)

    return mask


def hybrid_masking_with_token(tensor, mask_percentage, patch_size, num_full_patches_to_mask, generator):
    patch_size = 8
    B, T, C = tensor.shape
    device = tensor.device
    total_masks = int(mask_percentage * T)

    if num_full_patches_to_mask * patch_size > total_masks:
        logger.warning(
            f"[hybrid_masking_with_token] num_full_patches_to_mask={num_full_patches_to_mask} "
            f"× patch_size={patch_size} > total_masks={total_masks}. Setting to 0."
        )
        num_full_patches_to_mask = 0

    # === Patch info ===
    patch_ids_full = torch.arange(T, device=device) // patch_size  # [T]
    num_patches = T // patch_size

    rand_vals = torch.rand(B, C, num_patches, device=device, generator=generator)
    _, top_patch_ids = torch.topk(rand_vals, k=num_full_patches_to_mask, dim=2)  # [B, C, K]
    selected_patch_ids = top_patch_ids.unsqueeze(1).expand(B, T, C, num_full_patches_to_mask)
    patch_id_exp = patch_ids_full.view(1, T, 1, 1).expand(B, T, C, num_full_patches_to_mask)
    full_patch_mask = (patch_id_exp == selected_patch_ids).any(-1)  # [B, T, C]

    full_patch_counts = full_patch_mask.sum(dim=1)  # [B, C]
    total_masks_per_channel = int(mask_percentage * T)
    remaining_masks = (total_masks_per_channel - full_patch_counts).clamp(min=0)  # [B, C]

    rand_scores = torch.rand(B, T, C, device=device, generator=generator)
    rand_scores[full_patch_mask] = float("inf")
    sorted_scores, sorted_indices = torch.sort(rand_scores, dim=1)
    time_range = torch.arange(T, device=device).view(1, T, 1).expand(B, T, C)
    topk_mask = time_range < remaining_masks.view(B, 1, C)
    additional_mask = torch.zeros_like(topk_mask)
    additional_mask.scatter_(1, sorted_indices, topk_mask)
    mask = full_patch_mask | additional_mask  # [B, T, C]

    return mask


def mask_generate(generator, x, patch_len, mask_rate, mask_type="hybrid", num_full_patches_for_hybrid_mask=4):
    if mask_type == "hybrid":
        num_full_patches_for_hybrid_mask = int((num_full_patches_for_hybrid_mask) * (mask_rate / 0.125))
        mask = hybrid_masking_with_token(
            x, mask_rate, patch_len, num_full_patches_for_hybrid_mask, generator
        )  # in mask tensor currently missing position are 1
    elif mask_type == "block":
        mask = mask_contiguous_with_token(x, mask_rate, patch_len, generator)
    else:
        raise ValueError("Masking_strategy_not_implemented")

    return mask
