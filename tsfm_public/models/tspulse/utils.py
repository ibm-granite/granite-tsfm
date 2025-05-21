# Copyright contributors to the TSFM project
#
import math

import torch
from torch.utils.data import DataLoader, Dataset


def patchwise_stitched_reconstruction_vectorized_multikey(
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


# class PatchMaskingDatasetWrapper(Dataset):
#     def __init__(self, base_dataset, window_length, patch_length):
#         self.base_dataset = base_dataset
#         self.window_length = window_length
#         self.patch_length = patch_length
#         self.num_patches = math.ceil(window_length / patch_length)
#         self.total_len = len(base_dataset) * self.num_patches

#     def __len__(self):
#         return self.total_len

#     def __getitem__(self, idx):
#         base_idx = idx // self.num_patches
#         patch_idx = idx % self.num_patches

#         item = self.base_dataset[base_idx]
#         past_values = item["past_values"]
#         T, C = past_values.shape

#         past_observed_mask = torch.ones((T, C), dtype=torch.bool)

#         start = patch_idx * self.patch_length
#         end = min(start + self.patch_length, T)

#         past_observed_mask[start:end] = 0

#         return {
#             **item,
#             "past_observed_mask": past_observed_mask,
#         }


# def test_patchmaskingdatasetwrapper():
#     class DummyTimeSeriesDataset(Dataset):
#         def __init__(self, num_samples=5, T=512, C=2):
#             self.T = T
#             self.C = C
#             self.data = [
#                 {
#                     "past_values": torch.arange(i * T * C, (i + 1) * T * C)
#                     .view(T, C)
#                     .float(),
#                     "label": i,
#                 }
#                 for i in range(num_samples)
#             ]

#         def __len__(self):
#             return len(self.data)

#         def __getitem__(self, idx):
#             return self.data[idx]

#     window_length = 100
#     patch_length = 16
#     base_dataset = DummyTimeSeriesDataset(num_samples=5, T=512, C=2)
#     wrapper = PatchMaskingDatasetWrapper(
#         base_dataset, window_length=window_length, patch_length=patch_length
#     )
#     loader = DataLoader(wrapper, batch_size=1, shuffle=False)
#     breakpoint()
#     num_patches = math.ceil(window_length / patch_length)
#     total_len = len(base_dataset) * num_patches

#     assert len(wrapper) == total_len

#     prev_pv = None
#     pv_count = 0
#     mask_positions = []

#     for i, batch in enumerate(loader):
#         past_values = batch["past_values"][0]
#         past_observed_mask = batch["past_observed_mask"][0]

#         if prev_pv is None:
#             prev_pv = past_values
#             pv_count = 1
#         elif torch.equal(prev_pv, past_values):
#             pv_count += 1
#         else:
#             assert (
#                 pv_count == num_patches
#             ), f"Expected {num_patches} reps, got {pv_count}"
#             assert sorted(mask_positions) == list(range(len(mask_positions)))
#             prev_pv = past_values
#             pv_count = 1
#             mask_positions = []

#         T, C = past_values.shape
#         assert past_observed_mask.shape == (T, C)

#         # Check where mask is False (i.e. masked)
#         mask = past_observed_mask == 0
#         masked_rows = (mask.any(dim=1)).nonzero(as_tuple=True)[0]
#         assert masked_rows.numel() > 0, "Each sample must have some masked patch"

#         start = masked_rows[0].item()
#         patch_idx = start // patch_length
#         mask_positions.append(patch_idx)

#         # Validate the mask length
#         expected_masked_len = min(patch_length, T - start)
#         actual_masked_len = (
#             (mask[start : start + expected_masked_len].any(dim=1)).sum().item()
#         )
#         assert (
#             actual_masked_len == expected_masked_len
#         ), f"Expected {expected_masked_len} rows masked, got {actual_masked_len}"

#     assert (
#         pv_count == num_patches
#     ), f"Expected {num_patches} reps at end, got {pv_count}"
#     assert sorted(mask_positions) == list(range(len(mask_positions)))


def test_patchmaskingdatasetwrapper():
    class DummyTimeSeriesDataset(Dataset):
        def __init__(self, num_samples=5, T=512, C=2):
            self.T = T
            self.C = C
            self.data = [
                {
                    "past_values": torch.arange(i * T * C, (i + 1) * T * C).view(T, C).float(),
                    "label": i,
                }
                for i in range(num_samples)
            ]

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    window_length = 100
    patch_length = 16
    num_patches = math.ceil(window_length / patch_length)
    total_samples = 5

    for window_position in ["first", "last"]:
        base_dataset = DummyTimeSeriesDataset(num_samples=total_samples, T=512, C=2)
        wrapper = PatchMaskingDatasetWrapper(
            base_dataset,
            window_length=window_length,
            patch_length=patch_length,
            window_position=window_position,
        )
        loader = DataLoader(wrapper, batch_size=1, shuffle=False)
        assert len(wrapper) == total_samples * num_patches

        prev_pv = None
        pv_count = 0
        mask_positions = []

        for i, batch in enumerate(loader):
            past_values = batch["past_values"][0]
            past_observed_mask = batch["past_observed_mask"][0]

            if prev_pv is None:
                prev_pv = past_values
                pv_count = 1
            elif torch.equal(prev_pv, past_values):
                pv_count += 1
            else:
                # Check that patch indices were covered in order: always LTR
                expected = list(range(num_patches))
                assert (
                    mask_positions == expected
                ), f"Incorrect patch order for {window_position}: got {mask_positions}, expected {expected}"
                assert pv_count == num_patches, f"Expected {num_patches} reps, got {pv_count}"
                prev_pv = past_values
                pv_count = 1
                mask_positions = []

            T, C = past_values.shape
            assert past_observed_mask.shape == (T, C)

            # Check where mask is False (i.e. masked)
            mask = past_observed_mask == 0
            masked_rows = (mask.any(dim=1)).nonzero(as_tuple=True)[0]
            assert masked_rows.numel() > 0, "Each sample must have some masked patch"

            start = masked_rows[0].item()
            patch_idx = start % T // patch_length
            mask_positions.append(patch_idx)

            # Validate that the patch falls inside the expected window
            if window_position == "first":
                assert start >= 0 and start < window_length
            else:  # "last"
                assert start >= T - window_length and start < T

            # Validate the mask length
            expected_masked_len = min(patch_length, T - start)
            actual_masked_len = (mask[start : start + expected_masked_len].any(dim=1)).sum().item()
            assert (
                actual_masked_len == expected_masked_len
            ), f"Expected {expected_masked_len} rows masked, got {actual_masked_len}"

        # Final check for last group
        expected = list(range(num_patches))
        assert (
            mask_positions == expected
        ), f"Incorrect patch order for {window_position}: got {mask_positions}, expected {expected}"
        assert pv_count == num_patches, f"Expected {num_patches} reps at end, got {pv_count}"


def test_patchwise_stitched_reconstruction_vectorized_multikey():
    import math

    from .modeling_tspulse import TSPulseForReconstruction

    # Load pre-trained model
    model = TSPulseForReconstruction.from_pretrained(
        "./tspulse_model",
        fft_time_add_forecasting_pt_loss=False,
        num_input_channels=4,
        mask_type="user",
    ).to("cuda")
    model.eval()

    B, L, C = 2, 512, 4  # 4 channels with 1x, 2x, 3x, 4x base frequency
    base_freq = 1.0

    t = torch.linspace(0, 2 * math.pi, L).unsqueeze(1)  # [L, 1]
    waves = []

    for c in range(C):
        freq = base_freq * (c + 1)
        wave = torch.sin(freq * t)  # [L, 1]
        waves.append(wave)

    # Stack along channel dimension → [L, C], then expand to [B, L, C]
    past_values = torch.cat(waves, dim=1).unsqueeze(0).repeat(B, 1, 1)  # [B, L, C]

    patch_size = 16
    patchwise_stitched_reconstruction_vectorized_multikey(
        model,
        past_values=past_values.to("cuda"),
        patch_size=patch_size,
        keys_to_stitch=["reconstruction_outputs"],
        keys_to_aggregate=[
            # # "forecast_output",
            # "fft_reconstruction_outputs",
            # "original_past_values_fft",
            # # "future_values",
            # "original_fft_softmax",
            # "fft_softmax_preds",
        ],
        reconstruct_start=0,
        reconstruct_end=100,  # to get reconstruction of first 100 points.
        debug=False,
    )


if __name__ == "__main__":
    test_patchmaskingdatasetwrapper()
