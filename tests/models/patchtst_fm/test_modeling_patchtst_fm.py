# Copyright contributors to the TSFM project
#
# This code is based on the test code for PatchTSMixer in the HuggingFace Transformers Library:
# https://github.com/huggingface/transformers/blob/main/tests/models/patchtsmixer/test_modeling_patchtsmixer.py
"""Testing suite for the PyTorch PatchTST-FM model."""

import unittest

import torch
from einops import rearrange
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
        num_channels = 2

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
                past_values=input_data,
                prediction_length=forecast_length,
            )

        # Test 2: List of tensors input (split batch into list)
        input_list = list(input_data)

        with torch.no_grad():
            output_list = model(
                past_values=input_list,
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

    def test_channel_independence(self):
        """Test that multivariate forecasts maintain channel independence.

        This test verifies that the model treats each channel independently by:
        1. Creating a test tensor with 4 channels
        2. Reshaping it to treat channels as additional batch dimensions
        3. Verifying that forecasts match when properly reshaped back

        If channels are truly independent, then:
        - input shape (batch=2, time=128, channels=4) should produce the same per-channel
          forecasts as input shape (batch=8, time=128, channels=1) when we reshape
          (2, 4) -> (8, 1) by treating each channel as a separate batch element.
        """
        # Test parameters
        batch_size = 2
        context_length = 128
        forecast_length = 64
        num_channels = 4

        # Create random input data
        set_seed(42)
        input_data = torch.randn(batch_size, context_length, num_channels)

        # Create a smaller config for faster testing
        test_config = PatchTSTFMConfig(
            context_length=512,
            prediction_length=forecast_length,
            d_patch=16,
            d_model=128,
            n_head=4,
            n_layer=2,
            norm_first=True,
            pretrain_mask_ratio=0.4,
            pretrain_mask_cont=8,
            num_quantile=99,
        )

        # Initialize model
        model = PatchTSTFMForPrediction(test_config)
        model.eval()

        # Test 1: Forecast with original shape (batch_size, context_length, num_channels)
        with torch.no_grad():
            output_original = model(
                past_values=input_data,
                prediction_length=forecast_length,
            )

        # Test 2: Reshape input to treat each channel as a separate batch element
        # Reshape from (batch_size, context_length, num_channels) to (batch_size * num_channels, context_length, 1)
        input_reshaped = rearrange(input_data, "B C N -> (B N) C 1")

        with torch.no_grad():
            output_reshaped = model(
                past_values=input_reshaped,
                prediction_length=forecast_length,
            )

        # Reshape output back to original format for comparison
        # output_reshaped shape: (batch_size * num_channels, num_quantiles, forecast_length, 1)
        # Reshape to: (batch_size, num_channels, num_quantiles, forecast_length, 1)
        # Then permute to: (batch_size, num_quantiles, forecast_length, num_channels)
        output_reshaped_back = rearrange(output_reshaped.quantile_outputs, "(B N) Q F 1 -> B Q F N", B=batch_size)

        # Verify shapes match
        self.assertEqual(
            output_original.quantile_outputs.shape, output_reshaped_back.shape, "Shape mismatch after reshaping"
        )

        # Verify that both approaches produce equal results
        torch.testing.assert_close(
            output_original.quantile_outputs,
            output_reshaped_back,
            rtol=1e-4,
            atol=1e-4,
            msg="Forecasts should be identical when channels are reshaped as separate batch elements",
        )

    def test_forecast_list_of_1d_tensors(self):
        """Test that forecasting with a list of 1D tensors produces correct output shapes.

        This test verifies that when the model receives a list of 1D tensors (univariate time series),
        it produces outputs with the correct shape for each element in the list.
        """
        # Test parameters
        num_series = 5
        context_length = 128
        forecast_length = 64

        # Create random input data as a list of 1D tensors
        set_seed(42)
        input_list = [torch.randn(context_length) for _ in range(num_series)]

        # Create a smaller config for faster testing
        test_config = PatchTSTFMConfig(
            context_length=512,
            prediction_length=forecast_length,
            d_patch=16,
            d_model=128,
            n_head=4,
            n_layer=2,
            norm_first=True,
            pretrain_mask_ratio=0.4,
            pretrain_mask_cont=8,
            num_quantile=99,
        )

        # Initialize model
        model = PatchTSTFMForPrediction(test_config)
        model.eval()

        # Forecast with list of 1D tensors
        with torch.no_grad():
            output = model(
                past_values=input_list,
                prediction_length=forecast_length,
            )

        # Verify output is not None
        self.assertIsNotNone(output.quantile_outputs)

        # Check that output is a list with correct number of elements
        self.assertIsInstance(output.quantile_outputs, list)
        self.assertEqual(len(output.quantile_outputs), num_series)

        # Check that each element in the output list has the correct shape
        # For 1D input (univariate), each output should have shape (num_quantiles, forecast_length, 1)
        # since the model treats 1D input as having 1 channel
        for i, forecast in enumerate(output.quantile_outputs):
            expected_shape = (test_config.num_quantile, forecast_length, 1)
            self.assertEqual(
                forecast.shape,
                expected_shape,
                f"Output element {i} has incorrect shape. Expected {expected_shape}, got {forecast.shape}",
            )

        # Verify that stacking the list produces a valid tensor
        stacked_output = torch.stack(output.quantile_outputs, dim=0)
        expected_stacked_shape = (num_series, test_config.num_quantile, forecast_length, 1)
        self.assertEqual(
            stacked_output.shape,
            expected_stacked_shape,
            f"Stacked output has incorrect shape. Expected {expected_stacked_shape}, got {stacked_output.shape}",
        )


if __name__ == "__main__":
    unittest.main()
