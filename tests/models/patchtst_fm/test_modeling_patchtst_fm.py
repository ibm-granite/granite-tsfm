# Copyright contributors to the TSFM project
#
# This code is based on the test code for PatchTSMixer in the HuggingFace Transformers Library:
# https://github.com/huggingface/transformers/blob/main/tests/models/patchtsmixer/test_modeling_patchtsmixer.py
"""Testing suite for the PyTorch PatchTST-FM model."""

import unittest

import torch
from transformers.trainer_utils import set_seed

from tsfm_public.models.patchtst_fm import (
    PatchTSTFMConfig,
    PatchTSTFMForPrediction,
)


set_seed(42)


class PatchTSTFMFunctionalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup method: Called once before test-cases execution"""
        cls.params = {}
        cls.params.update(
            context_length=8192,
            prediction_length=64,
            d_patch=16,
            d_model=1024,
            n_head=16,
            n_layer=20,
            norm_first=True,
            pretrain_mask_ratio=0.4,
            pretrain_mask_cont=8,
            num_quantile=99,
        )

    def test_forecast_single_vs_list(self):
        """Test that forecasting with a single tensor and list of tensors produces equal results."""
        # Test parameters
        batch_size = 8
        context_length = 128
        forecast_length = 64
        num_channels = 1

        # Create random input data
        set_seed(42)
        input_data = torch.randn(batch_size, context_length, num_channels)

        # Create a smaller config for faster testing
        test_config = PatchTSTFMConfig(
            context_length=512,  # Smaller than default for faster testing
            prediction_length=forecast_length,
            d_patch=16,
            d_model=128,  # Smaller for faster testing
            n_head=4,
            n_layer=2,  # Fewer layers for faster testing
            norm_first=True,
            pretrain_mask_ratio=0.4,
            pretrain_mask_cont=8,
            num_quantile=99,
        )

        # Initialize model
        model = PatchTSTFMForPrediction(test_config)
        model.eval()

        # Test 1: Single tensor input
        with torch.no_grad():
            output_single = model(
                inputs=input_data,
                prediction_length=forecast_length,
            )

        # Test 2: List of tensors input (split batch into list)
        input_list = [i for i in input_data]

        with torch.no_grad():
            output_list = model(
                inputs=input_list,
                prediction_length=forecast_length,
            )

        # Verify outputs
        self.assertIsNotNone(output_single.quantile_outputs)
        self.assertIsNotNone(output_list.quantile_outputs)

        # Check that single tensor output has correct shape
        # Expected: (batch_size, num_quantiles, forecast_length, num_channels)
        expected_shape = (batch_size, test_config.num_quantile, forecast_length, num_channels)
        self.assertEqual(output_single.quantile_outputs.shape, expected_shape)

        # Check that list output has correct number of elements
        self.assertEqual(len(output_list.quantile_outputs), batch_size)

        # Check that each element in list has correct shape
        # Expected: (1, num_quantiles, forecast_length, num_channels) for each element
        for i, sample in enumerate(output_list.quantile_outputs):
            self.assertEqual(sample.shape, (test_config.num_quantile, forecast_length, num_channels))

        # Concatenate list outputs to compare with single tensor output
        output_list_concat = torch.stack(output_list.quantile_outputs, dim=0)

        # Verify that both approaches produce equal results
        torch.testing.assert_close(
            output_single.quantile_outputs,
            output_list_concat,
            rtol=1e-4,
            atol=1e-4,
            msg="Single tensor and list of tensors should produce equal forecasts",
        )


if __name__ == "__main__":
    unittest.main()
