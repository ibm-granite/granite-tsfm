# Copyright contributors to the TSFM project
#
# This code is based on the test code for PatchTSMixer in the HuggingFace Transformers Library:
# https://github.com/huggingface/transformers/blob/main/tests/models/patchtsmixer/test_modeling_patchtsmixer.py
""" Testing suite for the PyTorch TinyTimeMixer model. """

# Standard
import itertools

# import torchinfo
import unittest

import numpy as np

# Third Party
from parameterized import parameterized
from transformers import is_torch_available
from transformers.testing_utils import require_torch


# # Local
# from ...test_configuration_common import ConfigTester
# from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
# from ...test_pipeline_mixin import PipelineTesterMixin


# # from transformers.tests.test_configuration_common import ConfigTester
# from transformers.tests.test_modeling_common import (
#     ModelTesterMixin,
#     floats_tensor,
#     ids_tensor,
# )
# # from transformers.tests.test_pipeline_mixin import PipelineTesterMixin


TOLERANCE = 1e-4

if is_torch_available():
    # Third Party
    import torch

    # Local
    from tsfm_public.models.tinytimemixer import (
        TinyTimeMixerConfig,
        TinyTimeMixerForPrediction,
        TinyTimeMixerModel,
    )
    from tsfm_public.models.tinytimemixer.modeling_tinytimemixer import (
        TinyTimeMixerForPredictionOutput,
    )


@require_torch
class TinyTimeMixerFunctionalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup method: Called once before test-cases execution"""
        cls.params = {}
        cls.params.update(
            context_length=32,
            patch_length=8,
            num_input_channels=3,
            patch_stride=8,
            d_model=64,
            expansion_factor=2,
            num_layers=3,
            adaptive_patching_levels=0,
            dropout=0.2,
            mode="common_channel",  # common_channel,  mix_channel
            gated_attn=True,
            norm_mlp="LayerNorm",
            head_dropout=0.2,
            prediction_length=64,
            # num_labels=3,
            scaling="std",
            use_positional_encoding=False,
            self_attn=False,
            self_attn_heads=1,
            num_parallel_samples=4,
            decoder_num_layers=1,
            decoder_d_model=32,
            decoder_adaptive_patching_levels=0,
            decoder_raw_residual=False,
            decoder_mode="mix_channel",
            use_decoder=True,
        )

        cls.num_patches = (
            max(cls.params["context_length"], cls.params["patch_length"]) - cls.params["patch_length"]
        ) // cls.params["patch_stride"] + 1

        # batch_size = 32
        batch_size = 2
        cls.batch_size = batch_size
        int(cls.params["prediction_length"] / cls.params["patch_length"])

        cls.data = torch.rand(
            batch_size,
            cls.params["context_length"],
            cls.params["num_input_channels"],
        )

        cls.freq_token = torch.ones(batch_size)

        cls.enc_data = torch.rand(
            batch_size,
            cls.params["num_input_channels"],
            cls.num_patches,
            cls.params["patch_length"],
        )

        cls.enc_output = torch.rand(
            batch_size,
            cls.params["num_input_channels"],
            cls.num_patches,
            cls.params["d_model"],
        )

        cls.dec_output = torch.rand(
            batch_size,
            cls.params["num_input_channels"],
            cls.num_patches,
            cls.params["decoder_d_model"],
        )

        cls.flat_enc_output = torch.rand(
            batch_size,
            cls.num_patches,
            cls.params["d_model"],
        )

        cls.correct_pred_output = torch.rand(
            batch_size,
            cls.params["prediction_length"],
            cls.params["num_input_channels"],
        )
        # cls.correct_regression_output = torch.rand(
        #     batch_size, cls.params["num_targets"]
        # )

        cls.correct_pretrain_output = torch.rand(
            batch_size,
            cls.params["num_input_channels"],
            cls.num_patches,
            cls.params["patch_length"],
        )

        cls.correct_forecast_output = torch.rand(
            batch_size,
            cls.params["prediction_length"],
            cls.params["num_input_channels"],
        )

        cls.correct_sel_forecast_output = torch.rand(batch_size, cls.params["prediction_length"], 2)

    def test_patchmodel(self):
        config = TinyTimeMixerConfig(**self.__class__.params)
        mdl = TinyTimeMixerModel(config)
        output = mdl(self.__class__.data)
        self.assertEqual(output.last_hidden_state.shape, self.__class__.enc_output.shape)
        self.assertEqual(output.patch_input.shape, self.__class__.enc_data.shape)

    # def test_forecast_head(self):
    #     config = TinyTimeMixerConfig(**self.__class__.params)
    #     head = TinyTimeMixerForPredictionHead(
    #         config=config,
    #     )
    #     # output = head(self.__class__.enc_output, raw_data = self.__class__.correct_pretrain_output)
    #     output = head(self.__class__.enc_output)

    #     self.assertEqual(output.shape, self.__class__.correct_forecast_output.shape)

    def check_module(
        self,
        task,
        params=None,
        output_hidden_states=True,
    ):
        config = TinyTimeMixerConfig(**params)
        if task == "forecast":
            mdl = TinyTimeMixerForPrediction(config)
            if (
                "target_channel_filtered" in params
                and params["target_channel_filtered"]
                and params["prediction_channel_indices"] is not None
            ):
                target_input = self.__class__.correct_sel_forecast_output
            else:
                target_input = self.__class__.correct_forecast_output

            if config.prediction_channel_indices is not None:
                target_output = self.__class__.correct_sel_forecast_output
            else:
                target_output = target_input

            if config.prediction_filter_length is not None:
                target_output = target_output[:, : config.prediction_filter_length, :]

            if "target_pred_length_filtered" in params and params["target_pred_length_filtered"]:
                target_input = target_input[:, : config.prediction_filter_length, :]

            # ref_samples = target_output.unsqueeze(1).expand(-1, config.num_parallel_samples, -1, -1)

        else:
            print("invalid task")

        enc_output = self.__class__.enc_output

        if config.use_decoder:
            dec_output = self.__class__.dec_output
        else:
            dec_output = enc_output

        if target_input is None:
            output = mdl(
                self.__class__.data,
                output_hidden_states=output_hidden_states,
            )
        else:
            output = mdl(
                self.__class__.data,
                future_values=target_input,
                output_hidden_states=output_hidden_states,
                freq_token=self.__class__.freq_token,
            )

        if isinstance(output.prediction_outputs, tuple):
            for t in output.prediction_outputs:
                self.assertEqual(t.shape, target_output.shape)
        else:
            self.assertEqual(output.prediction_outputs.shape, target_output.shape)

        # self.assertEqual(output.last_hidden_state.shape, enc_output.shape)

        # if output_hidden_states is True:
        #     self.assertEqual(len(output.hidden_states), params["num_layers"])

        # else:
        #     self.assertEqual(output.hidden_states, None)

        self.assertEqual(output.loss.item() < np.inf, True)

        enc_output_shape = list(enc_output.shape)
        dec_output_shape = list(dec_output.shape)
        if config.resolution_prefix_tuning:
            enc_output_shape[-2] += 1
            dec_output_shape[-2] += 1
        self.assertEqual(list(output.backbone_hidden_state.shape), enc_output_shape)
        self.assertEqual(list(output.decoder_hidden_state.shape), dec_output_shape)

        # self.assertEqual(output.backbone_hidden_state.shape, enc_output.shape)
        # self.assertEqual(output.decoder_hidden_state.shape, dec_output.shape)

    @parameterized.expand(
        list(
            itertools.product(
                ["common_channel", "mix_channel"],
                [True, False],
                [True, False, "mean", "std"],
                [True, False],
                [None, [0, 2]],
                ["mse", "mae"],
                [3, 4],
                [True, False],
            )
        )
    )
    def test_forecast(
        self,
        mode,
        self_attn,
        scaling,
        gated_attn,
        prediction_channel_indices,
        loss,
        prediction_filter_length,
        target_pred_length_filtered,
    ):
        params = self.__class__.params.copy()
        params.update(
            mode=mode,
            self_attn=self_attn,
            scaling=scaling,
            prediction_channel_indices=prediction_channel_indices,
            gated_attn=gated_attn,
            loss=loss,
            prediction_filter_length=prediction_filter_length,
            target_pred_length_filtered=target_pred_length_filtered,
            target_channel_filtered=False,
        )

        self.check_module(task="forecast", params=params)

    @parameterized.expand(
        list(
            itertools.product(
                [None, [0, 2]],
                [True, False],
                [True, False],
                ["common_channel", "mix_channel"],
                [0, 2],
                [-1, 3],
                [True, False],
            )
        )
    )
    def test_forecast_decoder(
        self,
        prediction_channel_indices,
        use_decoder,
        decoder_raw_residual,
        decoder_mode,
        adaptive_patching_levels,
        decoder_adaptive_patching_levels,
        resolution_prefix_tuning,
    ):
        params = self.__class__.params.copy()
        params.update(
            mode="mix_channel",
            self_attn=False,
            scaling=True,
            prediction_channel_indices=prediction_channel_indices,
            gated_attn=True,
            use_decoder=use_decoder,
            decoder_raw_residual=decoder_raw_residual,
            decoder_mode=decoder_mode,
            adaptive_patching_levels=adaptive_patching_levels,
            decoder_adaptive_patching_levels=decoder_adaptive_patching_levels,
            resolution_prefix_tuning=resolution_prefix_tuning,
            target_channel_filtered=False,
        )

        self.check_module(task="forecast", params=params)

    def forecast_full_module(self, params=None, output_hidden_states=False, return_dict=None):
        config = TinyTimeMixerConfig(**params)
        mdl = TinyTimeMixerForPrediction(config)
        target_val = self.__class__.correct_forecast_output

        target_input = self.__class__.correct_forecast_output

        if config.prediction_channel_indices is not None:
            target_val = self.__class__.correct_sel_forecast_output

        enc_output = self.__class__.enc_output

        if config.use_decoder:
            dec_output = self.__class__.dec_output
        else:
            dec_output = enc_output

        if config.prediction_filter_length is not None:
            target_val = target_val[:, : config.prediction_filter_length, :]

        if "target_pred_length_filtered" in params and params["target_pred_length_filtered"]:
            target_input = target_input[:, : config.prediction_filter_length, :]

        output = mdl(
            self.__class__.data,
            future_values=target_input,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            freq_token=self.__class__.freq_token,
        )
        # print(mdl)
        # from torchsummary import summary

        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # mdl = mdl.to(device)
        # summary(mdl, self.__class__.data.shape)

        if isinstance(output, tuple):
            output = TinyTimeMixerForPredictionOutput(*output)

        if config.loss == "mse":
            self.assertEqual(output.prediction_outputs.shape, target_val.shape)
        enc_output_shape = list(enc_output.shape)
        dec_output_shape = list(dec_output.shape)
        if config.resolution_prefix_tuning:
            enc_output_shape[-2] += 1
            dec_output_shape[-2] += 1
        self.assertEqual(list(output.backbone_hidden_state.shape), enc_output_shape)
        self.assertEqual(list(output.decoder_hidden_state.shape), dec_output_shape)

        # if output_hidden_states is True:
        #     print("ooo", len(output.hidden_states))
        #     self.assertEqual(len(output.hidden_states), params["num_layers"])

        # else:
        #     self.assertEqual(output.hidden_states, None)

        self.assertEqual(output.loss.item() < np.inf, True)

    def test_forecast_full(self):
        self.check_module(task="forecast", params=self.__class__.params, output_hidden_states=True)
        # self.forecast_full_module(self.__class__.params, output_hidden_states = True)

    def test_forecast_full_2(self):
        params = self.__class__.params.copy()
        params.update(
            mode="mix_channel",
            adaptive_patching_levels=3,
            num_layers=2,
            resolution_prefix_tuning=False,
            prediction_filter_length=63,
            target_pred_length_filtered=False,
            loss="mse",
        )
        self.forecast_full_module(params, output_hidden_states=True)

    def test_forecast_full_2_with_return_dict(self):
        params = self.__class__.params.copy()
        params.update(
            mode="mix_channel",
        )
        self.forecast_full_module(params, output_hidden_states=True, return_dict=False)

    def test_forecast_full_3(self):
        params = self.__class__.params.copy()
        params.update(
            mode="mix_channel",
        )
        self.forecast_full_module(params, output_hidden_states=True)

    def test_forecast_full_5(self):
        params = self.__class__.params.copy()
        params.update(
            self_attn=True,
            use_positional_encoding=True,
            positional_encoding="sincos",
        )
        self.forecast_full_module(params, output_hidden_states=True)

    def test_forecast_full_4(self):
        params = self.__class__.params.copy()
        params.update(
            mode="mix_channel",
            prediction_channel_indices=[0, 2],
        )
        self.forecast_full_module(params)

    def test_forecast_full_distributional(self):
        params = self.__class__.params.copy()
        params.update(
            mode="mix_channel",
            prediction_channel_indices=[0, 2],
            loss="mae",
        )

        self.forecast_full_module(params)

    def test_forecast_full_distributional_2(self):
        params = self.__class__.params.copy()
        params.update(
            mode="mix_channel",
            prediction_channel_indices=[0, 2],
            loss="mae",
        )
        self.forecast_full_module(params)

    def test_forecast_full_distributional_3(self):
        params = self.__class__.params.copy()
        params.update(
            mode="mix_channel",
            # prediction_channel_indices=[0, 2],
            loss="mae",
        )
        self.forecast_full_module(params)

    def test_forecast_full_distributional_4(self):
        params = self.__class__.params.copy()
        params.update(
            mode="mix_channel",
            # prediction_channel_indices=[0, 2],
            loss="mae",
        )
        self.forecast_full_module(params)
