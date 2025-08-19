# Copyright contributors to the TSFM project
#

"""Testing suite for the PyTorch TSPulse model."""

import itertools
import math
import unittest

import numpy as np
import torch
from parameterized import parameterized
from torch.utils.data import DataLoader, Dataset

from tsfm_public.models.tspulse import (
    TSPulseConfig,
    TSPulseForClassification,
    TSPulseForReconstruction,
)
from tsfm_public.models.tspulse.utils.helpers import (
    PatchMaskingDatasetWrapper,
    get_embeddings,
    patchwise_stitched_reconstruction,
)


TOLERANCE = 1e-4


class TSPulseFunctionalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup method: Called once before test-cases execution"""
        cls.params = {}
        cls.total_embedding_size = 64

        cls.params.update(
            context_length=16,
            patch_length=2,
            mask_block_length=2,
            num_input_channels=3,
            patch_stride=2,
            expansion_factor=2,
            num_layers=2,
            dropout=0.1,
            mode="common_channel",  # common_channel,  mix_channel
            gated_attn=True,
            norm_mlp="LayerNorm",
            head_dropout=0.1,
            scaling="std",
            use_positional_encoding=False,
            self_attn=False,
            self_attn_heads=1,
            num_parallel_samples=4,
            decoder_num_layers=2,
            decoder_mode="mix_channel",
            d_model_layerwise_scale=[1, 0.75],  # 0.5, 0.5],
            num_patches_layerwise_scale=[1, 0.75],  # , 0.5, 0.5],
            decoder_num_patches_layerwise_scale=[0.75, 1],  #  0.75, 1],
            decoder_d_model_layerwise_scale=[0.75, 1],  # 0.75, 1],
            num_channels_layerwise_scale=[1, 1],  # 1, 1],  # [1, 0.75, 0.5, 0.5],
            decoder_num_channels_layerwise_scale=[
                1,
                1,
            ],  # 1, 1],  # [0.5, 0.5, 0.75, 1],
            d_model=8,
            decoder_d_model=8,
            num_targets=5,
            head_aggregation="max_pool",
            fuse_fft=True,
            patch_register_tokens=2,
            channel_register_tokens=None,
            fft_mask_ratio=0.2,
            fft_mask_strategy="random",
            use_learnable_mask_token=True,
            prediction_length=4,
            fft_time_add_forecasting_pt_loss=False,
            channel_mix_init="identity",
            reconstruction_loss_weight=1,
            masked_reconstruction_loss_weight=1,
            register_mixer_layers=1,
            head_gated_attention_activation="softmax",
            gated_attention_activation="softmax",
            head_attention=False,
            head_reduce_channels=None,
            fft_applied_on="scaled_ts",
        )

        cls.num_patches = (
            max(cls.params["context_length"], cls.params["patch_length"]) - cls.params["patch_length"]
        ) // cls.params["patch_stride"] + 1

        # batch_size = 32
        batch_size = 32
        cls.batch_size = batch_size

        cls.data = torch.rand(
            batch_size,
            cls.params["context_length"],
            cls.params["num_input_channels"],
        )

        cls.enc_data = torch.rand(
            batch_size,
            cls.params["num_input_channels"],
            cls.num_patches,
            cls.params["patch_length"],
        )

        cls.enc_output = torch.rand(
            batch_size,
            int(cls.params["num_channels_layerwise_scale"][-1] * cls.params["num_input_channels"]),
            int(cls.params["num_patches_layerwise_scale"][-1] * cls.num_patches),
            int(cls.params["d_model_layerwise_scale"][-1] * cls.params["d_model"]),
        )

        cls.enc_output_raw = torch.rand(
            batch_size,
            cls.params["num_input_channels"],
            cls.num_patches,
            cls.params["d_model"],
        )

        cls.dec_output = torch.rand(
            batch_size,
            int(cls.params["decoder_num_channels_layerwise_scale"][-1] * cls.params["num_input_channels"]),
            int(cls.params["decoder_num_patches_layerwise_scale"][-1] * cls.num_patches),
            int(cls.params["decoder_d_model_layerwise_scale"][-1] * cls.params["d_model"]),
        )

        cls.correct_reconstruction_output = torch.rand(
            batch_size,
            cls.params["context_length"],
            cls.params["num_input_channels"],
        )

        cls.loc = torch.rand(
            batch_size,
            1,
            cls.params["num_input_channels"],
        )

        cls.correct_classification_output = torch.rand(
            batch_size,
            cls.params["num_targets"],
        )

        cls.future_values = torch.rand(
            batch_size,
            cls.params["prediction_length"],
            cls.params["num_input_channels"],
        )

        cls.correct_classification_classes = torch.randint(0, cls.params["num_targets"], (batch_size,))

    def check_module(
        self,
        task,
        params=None,
        output_hidden_states=True,
        compression=True,
    ):
        config = TSPulseConfig(**params)
        if task == "reconstruction":
            mdl = TSPulseForReconstruction(config)
            target_input = self.__class__.correct_reconstruction_output
            target_output = target_input
        elif task == "classification":
            config.loss = "cross_entropy"
            mdl = TSPulseForClassification(config)
            target_output = self.__class__.correct_classification_output
            target_input = self.__class__.correct_classification_classes

        else:
            print("invalid task")

        if compression:
            enc_output = self.__class__.enc_output
            enc_output_shape = list(enc_output.shape)

            dec_output = self.__class__.dec_output
            dec_output_shape = list(dec_output.shape)
        else:
            enc_output = self.__class__.enc_output_raw
            dec_output = enc_output
            enc_output_shape = list(enc_output.shape)
            dec_output_shape = list(dec_output.shape)

        if config.mode == "common_channel" or task in ["classification"]:
            enc_output_shape[1] = config.num_input_channels  # no compression for these cases

        if config.fuse_fft:
            enc_output_shape[2] *= 2
            dec_output_shape[2] *= 2

        if config.patch_register_tokens is not None:
            enc_output_shape[2] += config.patch_register_tokens

            if task == "reconstruction" or (
                task in ["classification"] and config.classification_mode == "full_embedding"
            ):
                dec_output_shape[2] += config.patch_register_tokens

            if task in ["classification"] and config.classification_mode == "short_embedding":
                dec_output_shape[2] = config.patch_register_tokens

            if task in ["classification"] and config.classification_mode == "time_with_short_fft_embedding":
                dec_output_shape[2] = (dec_output_shape[2] // 2) + config.patch_register_tokens

            if task in [
                "classification",
            ] and config.classification_mode in ["fft_embedding", "time_embedding"]:
                dec_output_shape[2] = dec_output_shape[2] // 2

        if config.channel_register_tokens is not None:
            enc_output_shape[1] += config.channel_register_tokens

        if config.channel_virtual_expand_scale > 1:
            enc_output_shape[1] *= config.channel_virtual_expand_scale

        enc_output = torch.rand(tuple(enc_output_shape)).flatten(start_dim=2)
        dec_output = torch.rand(tuple(dec_output_shape)).flatten(start_dim=2)

        cat_samples = None
        if "categorical_vocab_size_list" in params and params["categorical_vocab_size_list"]:
            b = self.__class__.batch_size
            cat_samples = [torch.randint(0, a, (b, 1)) for a in params["categorical_vocab_size_list"]]
            cat_samples = torch.stack(cat_samples, dim=1).squeeze()

        if task == "reconstruction":
            output = mdl(
                self.__class__.data,
                future_values=self.__class__.future_values,
                output_hidden_states=output_hidden_states,
            )

            self.assertEqual(output.reconstruction_outputs.shape, target_output.shape)
            self.assertEqual(output.backbone_hidden_state.shape, enc_output.shape)
            self.assertEqual(output.past_values.shape, target_output.shape)

            if config.fuse_fft:
                if config.fft_weight > 0:
                    self.assertEqual(output.fft_reconstruction_outputs.shape, target_output.shape)
                    self.assertEqual(output.original_past_values_fft.shape, target_output.shape)
                if config.fft_original_signal_loss_weight > 0:
                    self.assertEqual(output.reconstructed_ts_from_fft.shape, target_output.shape)
                if config.fft_mask_ratio is not None and config.fft_mask_ratio > 0:
                    self.assertEqual(output.fft_mask.shape, target_output.shape)
                if config.enable_fft_prob_loss:
                    if config.fft_prob_length is None:
                        fft_prob_length = config.context_length // 2
                    else:
                        fft_prob_length = config.fft_prob_length

                    self.assertEqual(
                        output.fft_softmax_preds.shape,
                        target_output[:, :fft_prob_length, :].shape,
                    )
                    self.assertEqual(
                        output.original_fft_softmax.shape,
                        target_output[:, :fft_prob_length, :].shape,
                    )

            if config.fft_time_add_forecasting_pt_loss:
                self.assertEqual(output.future_values.shape, self.__class__.future_values.shape)
                self.assertEqual(output.forecast_output.shape, self.__class__.future_values.shape)

            if config.mask_ratio is not None and config.mask_ratio > 0:
                self.assertEqual(output.masked_past_values.shape, target_output.shape)
                self.assertEqual(output.mask.shape, target_output.shape)

            self.assertEqual(output.fft_loss.item() < np.inf, True)
            self.assertEqual(output.reconstruction_loss.item() < np.inf, True)
            self.assertEqual(output.forecast_loss.item() < np.inf, True)
            self.assertEqual(output.reconstructed_ts_from_fft_loss.item() < np.inf, True)

            self.assertEqual(output.masked_reconstruction_loss.item() < np.inf, True)

            if config.fuse_fft:
                pass

        elif task in ["classification"]:
            output = mdl(
                self.__class__.data,
                output_hidden_states=output_hidden_states,
                target_values=target_input,
                static_categorical_values=cat_samples,
            )
            self.assertEqual(output.prediction_outputs.shape, target_output.shape)

        self.assertEqual(output.loss.item() < np.inf, True)

        self.assertEqual(output.backbone_hidden_state.shape, enc_output.shape)

        self.assertEqual(output.loc.shape, self.__class__.loc.shape)
        self.assertEqual(output.scale.shape, self.__class__.loc.shape)
        self.assertEqual(output.decoder_hidden_state.shape, dec_output.shape)

    @parameterized.expand(
        list(
            itertools.product(
                ["common_channel", "mix_channel"],
                ["common_channel", "mix_channel"],
                [True, False],
                [2],
                ["dc", "last"],
                ["mask", "full", "mask_and_full"],
                [True, False],
                [True, False],
                [0, 1],
                ["revin", "std"],
                ["scaled_ts", "raw_ts"],
            )
        )
    )
    def test_reconstruction_2(
        self,
        mode,
        decoder_mode,
        fuse_fft,
        patch_register_tokens,
        fft_remove_component,
        loss_apply_mode,
        enable_fft_prob_loss,
        fft_time_add_forecasting_pt_loss,
        fft_original_signal_loss_weight,
        scaling,
        fft_applied_on,
    ):
        params = self.__class__.params.copy()
        params.update(
            mode=mode,
            decoder_mode=decoder_mode,
            fuse_fft=fuse_fft,
            patch_register_tokens=patch_register_tokens,
            fft_remove_component=fft_remove_component,
            loss_apply_mode=loss_apply_mode,
            enable_fft_prob_loss=enable_fft_prob_loss,
            fft_time_add_forecasting_pt_loss=fft_time_add_forecasting_pt_loss,
            fft_original_signal_loss_weight=fft_original_signal_loss_weight,
            scaling=scaling,
            fft_applied_on=fft_applied_on,
        )

        self.check_module(task="reconstruction", params=params)

        params.update(
            num_patches_layerwise_scale=None,
            num_channels_layerwise_scale=None,
            d_model_layerwise_scale=None,
            decoder_num_patches_layerwise_scale=None,
            decoder_num_channels_layerwise_scale=None,
            decoder_d_model_layerwise_scale=None,
        )
        self.check_module(
            task="reconstruction",
            params=params,
            output_hidden_states=False,
            compression=False,
        )

    @parameterized.expand(
        list(
            itertools.product(
                [True, False],
                [0.2, None],
                [0.2, None],
                ["var_hybrid", "block", "random", "hybrid"],
                [
                    True,
                    False,
                ],
                [
                    True,
                    False,
                ],
                [
                    True,
                    False,
                ],
                [
                    1,
                    3,
                ],
                [
                    True,
                    False,
                ],
            )
        )
    )
    def test_reconstruction_3(
        self,
        fuse_fft,
        mask_ratio,
        fft_mask_ratio,
        mask_type,
        use_learnable_mask_token,
        channel_consistent_masking,
        fft_time_consistent_masking,
        channel_virtual_expand_scale,
        batch_aware_masking,
    ):
        params = self.__class__.params.copy()
        params.update(
            fuse_fft=fuse_fft,
            mask_ratio=mask_ratio,
            fft_mask_ratio=fft_mask_ratio,
            mask_type=mask_type,
            use_learnable_mask_token=use_learnable_mask_token,
            channel_consistent_masking=channel_consistent_masking,
            fft_time_consistent_masking=fft_time_consistent_masking,
            channel_virtual_expand_scale=channel_virtual_expand_scale,
            batch_aware_masking=batch_aware_masking,
        )
        # print(params)
        self.check_module(task="reconstruction", params=params)
        self.check_module(task="reconstruction", params=params, output_hidden_states=False)

    @parameterized.expand(
        list(
            itertools.product(
                ["random", "magnitude"],
                ["plain", "log"],
                [2, None],
                [0, 0.2, None],
                [0, 0.2, None],
                [True, False],
                [True, False],
            )
        )
    )
    def test_reconstruction_4(
        self,
        fft_mask_strategy,
        fft_prob_mode,
        fft_prob_length,
        mask_ratio,
        fft_mask_ratio,
        use_learnable_mask_token,
        revin_affine,
    ):
        params = self.__class__.params.copy()
        params.update(
            fft_mask_strategy=fft_mask_strategy,
            fft_prob_mode=fft_prob_mode,
            fft_prob_length=fft_prob_length,
            mask_ratio=mask_ratio,
            fft_mask_ratio=fft_mask_ratio,
            use_learnable_mask_token=use_learnable_mask_token,
            scaling="revin",
            revin_affine=revin_affine,
        )

        self.check_module(task="reconstruction", params=params)

    @parameterized.expand(
        list(
            itertools.product(
                ["common_channel", "mix_channel"],
                [True, False, "mean", "std"],
                [True, False],
                ["common_channel", "mix_channel"],
                [True, False],
                ["zero", "identity"],
                [None, 3],
                [True, False],
                [None, [{"head_dropout": 0.3}, {"head_dropout": 0.1}]],
                [True, False],
            )
        )
    )
    def test_reconstruction_1(
        self,
        mode,
        scaling,
        gated_attn,
        decoder_mode,
        fuse_fft,
        channel_mix_init,
        register_mixer_layers,
        free_channel_flow,
        hydra_class_head,
        hydra_class_attention,
    ):
        params = self.__class__.params.copy()
        params.update(
            mode=mode,
            scaling=scaling,
            gated_attn=gated_attn,
            decoder_mode=decoder_mode,
            fuse_fft=fuse_fft,
            channel_mix_init=channel_mix_init,
            register_mixer_layers=register_mixer_layers,
            free_channel_flow=free_channel_flow,
            hydra_class_head=hydra_class_head,
            hydra_class_attention=hydra_class_attention,
        )

        self.check_module(task="reconstruction", params=params)

    @parameterized.expand(
        list(
            itertools.product(
                ["common_channel", "mix_channel"],
                [True, False],
                [True, False, "mean", "std"],
                [True, False],
                ["common_channel", "mix_channel"],
                ["max_pool", "avg_pool", None],
                [True, False],
                [2],
                [None],
                [
                    "full_embedding",
                    "long_embedding",
                    "short_embedding",
                    "time_embedding",
                    "fft_embedding",
                    "time_with_short_fft_embedding",
                ],
                [1, 3],
                [True, False],
            )
        )
    )
    def test_classification(
        self,
        mode,
        self_attn,
        scaling,
        gated_attn,
        decoder_mode,
        pooling,
        fuse_fft,
        patch_register_tokens,
        channel_register_tokens,
        classification_mode,
        channel_virtual_expand_scale,
        disable_mask_in_classification_eval,
    ):
        params = self.__class__.params.copy()
        params.update(
            mode=mode,
            self_attn=self_attn,
            scaling=scaling,
            gated_attn=gated_attn,
            decoder_mode=decoder_mode,
            head_aggregation=pooling,
            fuse_fft=fuse_fft,
            patch_register_tokens=patch_register_tokens,
            channel_register_tokens=channel_register_tokens,
            classification_mode=classification_mode,
            channel_virtual_expand_scale=channel_virtual_expand_scale,
            disable_mask_in_classification_eval=disable_mask_in_classification_eval,
        )

        self.check_module(task="classification", params=params)

    def test_reconstruction_full_2(self):
        params = self.__class__.params.copy()
        params.update(
            {
                "context_length": 16,
                "patch_length": 2,
                "mask_block_length": 2,
                "num_input_channels": 3,
                "patch_stride": 2,
                "expansion_factor": 2,
                "num_layers": 2,
                "dropout": 0.1,
                "mode": "common_channel",
                "gated_attn": True,
                "norm_mlp": "LayerNorm",
                "head_dropout": 0.1,
                "scaling": "std",
                "use_positional_encoding": False,
                "self_attn": False,
                "self_attn_heads": 1,
                "num_parallel_samples": 4,
                "decoder_num_layers": 2,
                "decoder_mode": "mix_channel",
                "d_model_layerwise_scale": [1, 0.75],
                "num_patches_layerwise_scale": [1, 0.75],
                "decoder_num_patches_layerwise_scale": [0.75, 1],
                "decoder_d_model_layerwise_scale": [0.75, 1],
                "num_channels_layerwise_scale": [1, 1],
                "decoder_num_channels_layerwise_scale": [1, 1],
                "d_model": 8,
                "decoder_d_model": 8,
                "num_targets": 5,
                "head_aggregation": "max_pool",
                "fuse_fft": True,
                "patch_register_tokens": 2,
                "channel_register_tokens": None,
                "fft_mask_ratio": 0.2,
                "fft_mask_strategy": "random",
                "use_learnable_mask_token": True,
                "prediction_length": 4,
                "fft_time_add_forecasting_pt_loss": False,
                "channel_mix_init": "identity",
                "reconstruction_loss_weight": 1,
                "masked_reconstruction_loss_weight": 1,
                "register_mixer_layers": 1,
                "head_gated_attention_activation": "softmax",
                "gated_attention_activation": "softmax",
                "head_attention": False,
                "head_reduce_channels": None,
                "mask_ratio": 0.2,
                "mask_type": "var_hybrid",
                "channel_consistent_masking": True,
                "fft_time_consistent_masking": True,
                "channel_virtual_expand_scale": 3,
                "batch_aware_masking": True,
            }
        )
        self.check_module(
            task="reconstruction",
            params=params,
            output_hidden_states=True,
        )

    def test_classification_2(self):
        params = self.__class__.params.copy()
        params.update(
            mode="mix_channel",
            # num_layers=4,
            # decoder_num_layers=4,
            fuse_fft=True,
            mask_ratio=0.3,
            fft_mask_ratio=0.3,
            channel_register_tokens=None,
            patch_register_tokens=2,
        )

        self.check_module(
            task="classification",
            params=params,
            output_hidden_states=True,
        )

    @parameterized.expand(
        list(
            itertools.product(
                [True, False],
            )
        )
    )
    def test_hybrid_mask(self, channel_consistent_masking):
        params = self.__class__.params.copy()
        params.update(
            mode="mix_channel",
            fuse_fft=True,
            mask_ratio=0.3,
            fft_mask_ratio=0.3,
            channel_register_tokens=None,
            patch_register_tokens=2,
            mask_type="hybrid",
            num_full_patches_for_hybrid_mask=1,
            channel_consistent_masking=channel_consistent_masking,
        )

        mdl = TSPulseForReconstruction(TSPulseConfig(**params))
        x = self.__class__.data
        patch_size = mdl.config.patch_length
        mask_token_values = mdl.backbone.time_masker.mask_token.clone()
        # future_values=self.__class__.future_values,
        masked_tensor, mask = mdl.backbone.time_masker.hybrid_masking_with_token(
            x,
            params["mask_ratio"],
            patch_size=patch_size,
            num_full_patches_to_mask=params["num_full_patches_for_hybrid_mask"],
        )
        self.assertEqual(masked_tensor.shape, x.shape)
        self.assertEqual(mask.shape, x.shape)

        T = x.shape[1]
        B = x.shape[0]
        C = x.shape[2]
        patch_ids_relative = torch.arange(T) % patch_size  # [T]
        patch_pos = patch_ids_relative.view(1, T, 1).expand(B, T, C)  # [B, T, C]

        expected_token = mask_token_values[patch_pos]  # [B, T, C]
        # Check if all masked values are replaced correctly
        self.assertTrue(torch.allclose(masked_tensor[mask], expected_token[mask]))

        total_elements = B * T * C
        masked_elements = mask.sum().item()
        expected = int(total_elements * params["mask_ratio"])
        self.assertTrue(abs(masked_elements - expected) / total_elements < 0.1)  # within 10%

        masked_tensor, mask = mdl.backbone.time_masker.hybrid_masking_with_token(
            x,
            1,
            patch_size=patch_size,
            num_full_patches_to_mask=params["num_full_patches_for_hybrid_mask"],
        )

        self.assertTrue(torch.all(mask))

        masked_tensor, mask = mdl.backbone.time_masker.hybrid_masking_with_token(
            x,
            0,
            patch_size=patch_size,
            num_full_patches_to_mask=params["num_full_patches_for_hybrid_mask"],
        )

        self.assertTrue(torch.equal(masked_tensor, x))
        self.assertEqual(mask.sum().item(), 0)

    @parameterized.expand(
        list(
            itertools.product(
                [True, False],
            )
        )
    )
    def test_variable_hybrid_mask(self, channel_consistent_masking):
        params = self.__class__.params.copy()
        params.update(
            mode="mix_channel",
            fuse_fft=True,
            mask_ratio=0.3,
            fft_mask_ratio=0.3,
            channel_register_tokens=None,
            patch_register_tokens=2,
            mask_type="var_hybrid",
            full_patch_mask_percentage=0.5,
            channel_consistent_masking=channel_consistent_masking,
        )

        mdl = TSPulseForReconstruction(TSPulseConfig(**params))
        x = self.__class__.data
        patch_size = mdl.config.patch_length
        mask_token_values = mdl.backbone.time_masker.mask_token.clone()

        # === Base call
        masked_tensor, mask = mdl.backbone.time_masker.variable_length_hybrid_masking_with_token(
            tensor=x,
            mask_percentage=params["mask_ratio"],
            patch_size=patch_size,
            full_patch_mask_percentage=params["full_patch_mask_percentage"],
        )
        self.assertEqual(masked_tensor.shape, x.shape)
        self.assertEqual(mask.shape, x.shape)

        T = x.shape[1]
        B = x.shape[0]
        C = x.shape[2]
        patch_ids_relative = torch.arange(T) % patch_size  # [T]
        patch_pos = patch_ids_relative.view(1, T, 1).expand(B, T, C)  # [B, T, C]

        expected_token = mask_token_values[patch_pos]  # [B, T, C]
        self.assertTrue(torch.allclose(masked_tensor[mask], expected_token[mask]))

        total_elements = B * T * C

        masked_elements = mask.sum().item()
        actual_ratio = masked_elements / total_elements
        expected_ratio = params["mask_ratio"]

        self.assertTrue(
            actual_ratio <= expected_ratio + 0.1,
            f"Actual mask ratio {actual_ratio:.4f} > expected {expected_ratio}",
        )

    def test_past_observed_mask(self):
        params = self.__class__.params.copy()
        params.update(
            mode="mix_channel",
            fuse_fft=True,
            mask_ratio=0.3,
            fft_mask_ratio=0.3,
            channel_register_tokens=None,
            patch_register_tokens=2,
            mask_type="hybrid",
            num_full_patches_for_hybrid_mask=1,
        )

        mdl = TSPulseForReconstruction(TSPulseConfig(**params))
        x = self.__class__.data
        B, T, C = x.shape
        patch_size = mdl.config.patch_length
        mask_token_values = mdl.backbone.time_masker.mask_token.clone()

        past_observed_mask = torch.ones(B, T, C, dtype=torch.bool)
        past_observed_mask[:, 0:4, :] = False  # Mark patch 0 (t=0..3) as missing

        model_output = mdl(x, past_observed_mask=past_observed_mask)
        masked_tensor = model_output.masked_past_values
        mask = model_output.mask

        patch_ids_relative = torch.arange(T) % patch_size
        patch_pos = patch_ids_relative.view(1, T, 1).expand(B, T, C)
        expected_token = mask_token_values[patch_pos]
        # === Tests ===
        self.assertEqual(masked_tensor.shape, x.shape)
        self.assertEqual(mask.shape, x.shape)

        # Check masked positions were replaced by correct tokens
        self.assertTrue(torch.allclose(masked_tensor[mask], expected_token[mask]))

        # Check unmasked positions remain unchanged
        self.assertTrue(torch.allclose(masked_tensor[~mask], x[~mask]))

    def test_patchwise_stitched_reconstruction(self):
        params = self.__class__.params.copy()

        params.update(
            context_length=512,
            patch_length=16,
            patch_stride=16,
            fft_time_add_forecasting_pt_loss=False,
            num_input_channels=4,
            mask_type="user",
        )

        model = TSPulseForReconstruction(TSPulseConfig(**params))

        # # Load pre-trained model
        # model = TSPulseForReconstruction.from_pretrained(
        #     "./tspulse_model",
        #     fft_time_add_forecasting_pt_loss=False,
        #     num_input_channels=4,
        #     mask_type="user",
        # ).to("cuda")
        # model.eval()

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

        patch_size = params["patch_length"]
        patchwise_stitched_reconstruction(
            model,
            past_values=past_values,
            patch_size=patch_size,
            keys_to_stitch=["reconstruction_outputs", "fft_reconstruction_outputs"],
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

    def test_patchmaskingdatasetwrapper(self):
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

        for window_position in ["last", "first"]:
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

                # FIX: Normalize start to window to get correct patch_idx
                window_start = 0 if window_position == "first" else T - window_length
                relative_start = start - window_start
                patch_idx = relative_start // patch_length
                mask_positions.append(patch_idx)

                # Ensure mask is inside selected window
                assert window_start <= start < window_start + window_length

                # Validate expected masked length
                expected_masked_len = min(patch_length, T - start)
                actual_masked_len = (mask[start : start + expected_masked_len].any(dim=1)).sum().item()
                assert (
                    actual_masked_len == expected_masked_len
                ), f"Expected {expected_masked_len} rows masked, got {actual_masked_len}"

            # Final flush for last group
            expected = list(range(num_patches))
            assert (
                mask_positions == expected
            ), f"Incorrect patch order for {window_position}: got {mask_positions}, expected {expected}"
            assert pv_count == num_patches, f"Expected {num_patches} reps at end, got {pv_count}"

    def test_get_embeddings(self):
        params = self.__class__.params.copy()
        params.update(
            context_length=512,
            patch_length=8,
            patch_stride=8,
            d_model=24,
            decoder_d_model=24,
            patch_register_tokens=10,
            num_input_channels=1,
            mask_type="user",
            d_model_layerwise_scale=[1, 1],
            num_patches_layerwise_scale=[1, 1],
        )
        past_values = torch.randn((1, 512, 1))  # [B, T, C]

        # num_patches=64*2 ((512-8)/8 + 1)=64, concat([time, fft]))
        # backbone_d_model=decoder_d_model=24, time_num_patches=fft_num_patches=64
        model = TSPulseForReconstruction(TSPulseConfig(**params))
        component = "backbone"
        assert get_embeddings(model, past_values, component=component, mode="time").shape[2] == 1536  # 64*24
        assert get_embeddings(model, past_values, component=component, mode="fft").shape[2] == 1536
        assert get_embeddings(model, past_values, component=component, mode="register").shape[2] == 240  # 10*24
        component = "decoder"
        assert get_embeddings(model, past_values, component=component, mode="time").shape[2] == 1536
        assert get_embeddings(model, past_values, component=component, mode="fft").shape[2] == 1536
        assert get_embeddings(model, past_values, component=component, mode="register").shape[2] == 240

        params.update(
            d_model_layerwise_scale=[1, 0.75],  # backbone_d_model_layerwise = [24, 18]
        )
        # backbone_d_model=18
        model = TSPulseForReconstruction(TSPulseConfig(**params))
        component = "backbone"
        assert get_embeddings(model, past_values, component=component, mode="time").shape[2] == 1152  # 64*18
        assert get_embeddings(model, past_values, component=component, mode="fft").shape[2] == 1152
        assert get_embeddings(model, past_values, component=component, mode="register").shape[2] == 180  # 10*18
        # decoder_d_model=24, i.e., d_model_layerwise_scale does not affect decoder_d_model
        component = "decoder"
        assert get_embeddings(model, past_values, component=component, mode="time").shape[2] == 1536
        assert get_embeddings(model, past_values, component=component, mode="fft").shape[2] == 1536
        assert get_embeddings(model, past_values, component=component, mode="register").shape[2] == 240

        params.update(
            num_patches_layerwise_scale=[1, 0.75],  # backbone_num_patches_layerwise = [128, 96]
        )
        # backbone_d_model=18, time_num_patches=48 (=96/2),
        model = TSPulseForReconstruction(TSPulseConfig(**params))
        component = "backbone"
        assert get_embeddings(model, past_values, component=component, mode="time").shape[2] == 864  # 48*18
        assert get_embeddings(model, past_values, component=component, mode="fft").shape[2] == 864
        assert get_embeddings(model, past_values, component=component, mode="register").shape[2] == 180

        params.update(
            d_model=8,
            decoder_d_model=8,
            patch_register_tokens=8,
        )
        model = TSPulseForReconstruction(TSPulseConfig(**params))
        component = "decoder"
        assert get_embeddings(model, past_values, component=component, mode="register").shape[2] == 64  # 8*8
