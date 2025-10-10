# Copyright contributors to the TSFM project
#
import math
from typing import Literal, Optional, Union

import torch
from torch.utils.data import Dataset

from ..modeling_tspulse import TSPulseForClassification, TSPulseForReconstruction


def patchwise_stitched_reconstruction(
    model,
    past_values,
    patch_size,
    keys_to_stitch,
    keys_to_aggregate,
    reconstruct_start,
    reconstruct_end,
    debug=False,
):
    """
    Performs patchwise reconstruction within a specified time window on a multivariate time-series tensor.
    Only patches whose **start indices fall within** [reconstruct_start, reconstruct_end) are masked and reconstructed.
    Results are then stitched or aggregated across the specified keys.

    Args:
        model: A callable that accepts `past_values` and `past_observed_mask` and returns a dict of outputs.
        past_values (torch.Tensor): Input tensor of shape [B, L, C], where
                                    B = batch size,
                                    L = sequence length,
                                    C = number of input channels.
        patch_size (int): Size of each non-overlapping patch.
        keys_to_stitch (List[str]): Keys in the model output dict that should be reconstructed and stitched
                                    back into full time-series format (shape [B, L, C]).
        keys_to_aggregate (List[str]): Keys in the model output dict that should be aggregated (mean-pooled)
                                       across patches (resulting in shape [B, ...]).
        reconstruct_start (int): Start index (inclusive) of the reconstruction window.
        reconstruct_end (int): End index (exclusive) of the reconstruction window.
        debug (bool): If True, also returns patch-level outputs and patch indices.

    Returns:
        result_dict (dict): Dictionary containing:
            - For each key in `keys_to_stitch`: Reconstructed and stitched tensor of shape [B, L, C],
              with reconstructed values only in the patchwise regions; rest is filled with NaNs.
            - For each key in `keys_to_aggregate`: Aggregated (mean) tensor of shape [B, ...].

        If `debug=True`, also returns:
            patch_outputs (dict):
                - For each stitched key: Tensor of shape [B, num_selected_patches, patch_size, C],
                  which contains patch-level reconstructed outputs.
                - For each aggregated key: Tensor of shape [B, num_selected_patches, ...],
                  representing the output for each patch before aggregation.
                - "patch_starts_selected" (torch.Tensor): 1D tensor of shape [num_selected_patches]
                  with the start positions of each selected patch (in original time indices).

    Raises:
        ValueError: If no patches fall within the reconstruction window.
    """

    B, L, C = past_values.shape
    device = past_values.device
    num_patches = L // patch_size

    patch_indices = torch.arange(num_patches, device=device)
    patch_starts = patch_indices * patch_size
    # patch_ends = patch_starts + patch_size

    # Select only patches fully inside the reconstruct window
    valid_mask = (patch_starts >= reconstruct_start) & (patch_starts < reconstruct_end)

    selected_patch_indices = patch_indices[valid_mask]
    num_selected_patches = selected_patch_indices.shape[0]

    if num_selected_patches == 0:
        raise ValueError("No patches fall entirely within the reconstruction window.")

    # Step 1: Create expanded inputs → shape: [B * num_selected_patches, L, C]
    past_values_expanded = past_values.unsqueeze(1).repeat(1, num_selected_patches, 1, 1)
    past_values_expanded = past_values_expanded.view(B * num_selected_patches, L, C)

    # Step 2: Create past_observed_mask: [B * num_selected_patches, L, C]
    past_observed_mask = torch.ones_like(past_values, dtype=torch.bool)
    past_observed_mask = past_observed_mask.unsqueeze(1).repeat(1, num_selected_patches, 1, 1)
    past_observed_mask = past_observed_mask.view(B * num_selected_patches, L, C)

    patch_starts_selected = patch_starts[valid_mask]
    for i, start in enumerate(patch_starts_selected):
        b_indices = torch.arange(B, device=device)
        idx_range = slice(start.item(), start.item() + patch_size)
        flat_idx = b_indices * num_selected_patches + i
        past_observed_mask[flat_idx, idx_range, :] = 0  # Mask the patch

    # Step 3: Forward pass
    model_outputs = model(
        past_values=past_values_expanded,
        past_observed_mask=past_observed_mask,
    )

    # Step 4: Prepare indices for patch extraction
    indices = torch.arange(patch_size, device=device).unsqueeze(0) + patch_starts_selected.view(-1, 1)
    indices_expanded = indices.unsqueeze(-1).expand(-1, -1, C)  # [num_selected_patches, patch_size, C]
    indices_expanded = indices_expanded.repeat(B, 1, 1)  # [B * num_selected_patches, patch_size, C]

    # Step 5: Initialize result dict
    result_dict = {key: torch.full_like(past_values, float("nan")) for key in keys_to_stitch}
    patch_outputs = {}

    # Step 6: Stitch keys
    for key in keys_to_stitch:
        output = model_outputs[key]  # [B * num_selected_patches, L, C]
        patches = torch.gather(output, dim=1, index=indices_expanded)  # [B * num_selected_patches, patch_size, C]
        patches = patches.view(B, num_selected_patches, patch_size, C)

        for i, start in enumerate(patch_starts_selected):
            result_dict[key][:, start : start + patch_size, :] = patches[:, i]

        if debug:
            patch_outputs[key] = patches  # [B, num_selected_patches, patch_size, C]

    # Step 7: Aggregate keys
    for key in keys_to_aggregate:
        output = model_outputs[key]  # [B * num_selected_patches, ...]
        out_shape = output.shape[1:]
        output = output.view(B, num_selected_patches, *out_shape)
        mean_output = output.mean(dim=1)
        result_dict[key] = mean_output
        if debug:
            patch_outputs[key] = output  # [B, num_selected_patches, ...]

    if debug:
        patch_outputs["patch_starts_selected"] = patch_starts_selected
        return result_dict, patch_outputs
    else:
        return result_dict


class PatchMaskingDatasetWrapper(Dataset):
    def __init__(self, base_dataset, window_length, patch_length, window_position="first"):
        """
        A dataset wrapper for fine-tuning TSPulse on time-series anomaly detection (AD) tasks
        using patch-level masking.

        In AD fine-tuning, TSPulse is trained to reconstruct specific patches of the input sequence,
        given the context of the surrounding time steps. This wrapper enables systematic patch masking
        across a fixed-size window (first or last) of the input sequence, allowing the model to learn local patterns
        and detect deviations during reconstruction.

        Each original time-series sample is expanded into multiple samples, one for each patch in the
        selected window. For each of these, a different patch is masked.
        using a boolean `past_observed_mask`, while the rest of the input remains visible to the model.

        Specifically, for each input sample of shape [T, C], the wrapper:
        - selects a fixed window of length `window_length` from either the start or end of the sequence
        - splits the window into `ceil(window_length / patch_length)` non-overlapping patches
        - generates multiple copies of the sample, each with a different patch marked as unobserved
            in the `past_observed_mask`, while all other time steps remain visible

        Args:
            base_dataset: The original dataset
            window_length: Number of time steps to apply patch masking to
            patch_length: Length of each patch
            window_position: "first" or "last" — which part of the sequence to apply masking on
        """
        assert window_position in [
            "first",
            "last",
        ], "window_position must be 'first' or 'last'"
        self.base_dataset = base_dataset
        self.window_length = window_length
        self.patch_length = patch_length
        self.window_position = window_position
        self.num_patches = math.ceil(window_length / patch_length)
        self.total_len = len(base_dataset) * self.num_patches

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        base_idx = idx // self.num_patches
        patch_idx = idx % self.num_patches

        item = self.base_dataset[base_idx]
        past_values = item["past_values"]
        T, C = past_values.shape

        # Determine window start position based on window_position
        if self.window_position == "first":
            window_start = 0
        else:  # "last"
            window_start = max(0, T - self.window_length)

        # Calculate actual patch position within full time series
        patch_start = window_start + patch_idx * self.patch_length
        patch_end = min(patch_start + self.patch_length, T)

        # Create full-size mask, mask only selected patch in the selected window
        past_observed_mask = torch.ones((T, C), dtype=torch.bool)
        past_observed_mask[patch_start:patch_end] = 0

        return {
            **item,
            "past_observed_mask": past_observed_mask,
        }


def get_embeddings(
    model: Union[TSPulseForReconstruction, TSPulseForClassification],
    past_values: torch.Tensor,
    past_observed_mask: Optional[torch.Tensor] = None,
    component: Literal["backbone", "decoder"] = "decoder",
    mode: Literal["time", "fft", "register", "full"] = "register",
) -> torch.Tensor:
    """
    Obtain embeddings from a TSPulse model.

    Args:
        model: A TSPulse model object with a callable interface that accepts `past_values` and `past_observed_mask`.
        past_values (torch.Tensor): Input tensor of shape [B, L, C], where
                                    B = batch size,
                                    L = sequence length,
                                    C = number of input channels.
        past_observed_mask (Optional[torch.Tensor]): Mask tensor of the same shape as `past_values`, indicating observed (1.0) vs. missing (0.0) values.
        component (str): Component to use to get the embedding. Allowed values are "backbone" and "decoder".
        mode (str): Specifies the type of embeddings to extract. One of:
            - "time": Extracts time-domain embeddings.
            - "fft": Extracts frequency-domain (FFT) embeddings.
            - "register": Extracts register token embeddings.
            - "full": Returns the full embedding without slicing.

    Returns:
        embeddings (torch.Tensor): Tensor of shape [B, C, D], where D depends on the selected mode.
    """

    num_reg_tokens = model.config.patch_register_tokens
    embeddings = model(past_values, past_observed_mask=past_observed_mask)
    if component == "backbone":
        d_model = model.config.d_model_layerwise[-1]
        num_patches = model.config.num_patches_layerwise[-1]

        embeddings = embeddings["backbone_hidden_state"]  # [B, C, D]
    elif component == "decoder":
        d_model = model.config.decoder_d_model_layerwise[-1]
        num_patches = model.config.decoder_num_patches_layerwise[-1]

        embeddings = embeddings["decoder_hidden_state"]  # [B, C, D]
    else:
        raise ValueError(f"Invalid component: {component}. Choose 'backbone' or 'decoder'.")

    time_emb_size = fft_emb_size = (num_patches // 2) * d_model
    reg_emb_size = num_reg_tokens * d_model

    if mode == "time":
        return embeddings[:, :, :time_emb_size]
    elif mode == "fft":
        return embeddings[:, :, time_emb_size : (time_emb_size + fft_emb_size)]
    elif mode == "register":
        return embeddings[:, :, -reg_emb_size:]
    elif mode == "full":
        return embeddings
    else:
        raise ValueError(f"Invalid mode: {mode}. Choose from 'time', 'fft', 'register', 'full'.")
