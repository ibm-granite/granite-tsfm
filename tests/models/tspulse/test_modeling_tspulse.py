# Copyright contributors to the TSFM project
#

"""Testing suite for the PyTorch TSPulse model."""

import itertools

# import torchinfo
import unittest

import numpy as np
import torch
from parameterized import parameterized

from tsfm_public.models.tspulse import (
    TSPulseConfig,
    TSPulseForClassificationOrRegression,
    TSPulseForReconstruction,
)


# from tsfm.models.tspulse.modeling_tspulse import TSPulseFFTMasker, TSPulseMasking


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

        # cls.params["patch_stride"] = int(cls.params["stride_ratio"] * cls.params["patch_length"])
        # cls.params["d_model"] = int(cls.params["d_model_scale"] * cls.params["patch_length"])
        # cls.params["decoder_d_model"] = int(cls.params["decoder_d_model_scale"] * cls.params["patch_length"])

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
        # cls.small_input = torch.randn(
        #     6, 12, 1
        # )  # [B, T, C] (B=6, T=12, C=1), divisible by 4

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

        cls.correct_regression_output = torch.rand(
            batch_size,
            cls.params["num_targets"],
        )

        cls.future_values = torch.rand(
            batch_size,
            cls.params["prediction_length"],
            cls.params["num_input_channels"],
        )

        cls.correct_classification_classes = torch.randint(0, cls.params["num_targets"], (batch_size,))

    # def test_patchmodel(self):
    #     config = TSPulseConfig(**self.__class__.params)
    #     mdl = TSPulseModel(config)
    #     output = mdl(self.__class__.data)
    #     self.assertEqual(output.last_hidden_state.shape, self.__class__.enc_output.shape)
    #     self.assertEqual(output.patch_input.shape, self.__class__.enc_data.shape)

    # def test_forecast_head(self):
    #     config = TSPulseConfig(**self.__class__.params)
    #     head = TSPulseForReconstructionHead(
    #         config=config,
    #     )
    #     # output = head(self.__class__.enc_output, raw_data = self.__class__.correct_pretrain_output)
    #     output = head(self.__class__.enc_output)

    #     self.assertEqual(output.shape, self.__class__.correct_reconstruction_output.shape)

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
            mdl = TSPulseForClassificationOrRegression(config)
            target_output = self.__class__.correct_classification_output
            target_input = self.__class__.correct_classification_classes

        elif task == "regression":
            mdl = TSPulseForClassificationOrRegression(config)
            target_output = self.__class__.correct_regression_output
            target_input = self.__class__.correct_regression_output

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

        if config.mode == "common_channel" or task in ["classification", "regression"]:
            enc_output_shape[1] = config.num_input_channels  # no compression for these cases

        if config.fuse_fft:
            enc_output_shape[2] *= 2
            dec_output_shape[2] *= 2

        if config.patch_register_tokens is not None:
            enc_output_shape[2] += config.patch_register_tokens

            if task == "reconstruction" or (
                task in ["classification", "regression"] and config.classification_mode == "full_embedding"
            ):
                dec_output_shape[2] += config.patch_register_tokens

            if task in ["classification", "regression"] and config.classification_mode == "short_embedding":
                dec_output_shape[2] = config.patch_register_tokens

            if (
                task in ["classification", "regression"]
                and config.classification_mode == "time_with_short_fft_embedding"
            ):
                dec_output_shape[2] = (dec_output_shape[2] // 2) + config.patch_register_tokens

            if task in [
                "classification",
                "regression",
            ] and config.classification_mode in ["fft_embedding", "time_embedding"]:
                dec_output_shape[2] = dec_output_shape[2] // 2

        if config.channel_register_tokens is not None:
            enc_output_shape[1] += config.channel_register_tokens
            # dec_output_shape[1] += config.channel_register_tokens

        if config.channel_virtual_expand_scale > 1:
            enc_output_shape[1] *= config.channel_virtual_expand_scale

            # dec_output_shape[1] += config.channel_register_tokens
        enc_output = torch.rand(tuple(enc_output_shape)).flatten(start_dim=2)
        dec_output = torch.rand(tuple(dec_output_shape)).flatten(start_dim=2)

        cat_samples = None
        if "categorical_vocab_size_list" in params and params["categorical_vocab_size_list"]:
            b = self.__class__.batch_size
            cat_samples = [torch.randint(0, a, (b, 1)) for a in params["categorical_vocab_size_list"]]
            # for i in cat_samples:
            #     print(i.shape,"jjj")
            cat_samples = torch.stack(cat_samples, dim=1).squeeze()
            # print(cat_samples.shape)
            # print(cat_samples)

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
            # if mdl.config.variational is True:
            #     samples = mdl.samples(target_output.shape[0])
            #     self.assertEqual(samples.shape, target_output.shape)

        elif task in ["classification", "regression"]:
            output = mdl(
                self.__class__.data,
                output_hidden_states=output_hidden_states,
                target_values=target_input,
                static_categorical_values=cat_samples,
            )
            self.assertEqual(output.prediction_outputs.shape, target_output.shape)

        self.assertEqual(output.loss.item() < np.inf, True)

        # enc_output_shape = list(enc_output.shape)
        # dec_output_shape = list(dec_output.shape)
        # self.assertEqual(list(output.backbone_hidden_state.shape), enc_output.shape)
        self.assertEqual(output.backbone_hidden_state.shape, enc_output.shape)
        # self.assertEqual(list(output.decoder_hidden_state.shape), dec_output_shape)
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

        # print(
        #     mode,
        #     decoder_mode,
        #     fuse_fft,
        #     patch_register_tokens,
        #     fft_remove_component,
        #     loss_apply_mode,
        #     enable_fft_prob_loss,
        #     fft_time_add_forecasting_pt_loss,
        #     fft_original_signal_loss_weight,
        # )
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
        # self.check_module(
        #     task="reconstruction", params=params, output_hidden_states=False
        # )

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
        # self.check_module(
        #     task="reconstruction", params=params, output_hidden_states=False
        # )

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
    def test_classification_or_regression(
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
            # decoder_num_layers=2,
            # decoder_d_model_layerwise=[0.25, 0.125],
            # decoder_num_patches_layerwise_scale=[0.25, 0.25],
            # num_channels_layerwise_scale=None,
            # decoder_num_channels_layerwise_scale=[1, 0.5],
        )

        # print(
        #     mode,
        #     self_attn,
        #     scaling,
        #     gated_attn,
        #     decoder_mode,
        #     pooling,
        #     fuse_fft,
        #     patch_register_tokens,
        #     channel_register_tokens,
        #     classification_mode,
        #     channel_virtual_expand_scale,
        #     # use_channel_register_tokens_for_classification_head,
        # )
        self.check_module(task="classification", params=params)
        self.check_module(task="regression", params=params)
        # self.check_module(task="classification", params=params, output_hidden_states=False)

    # def test_classification_sample(self):
    #     params = self.__class__.params.copy()
    #     params.update(
    #         inject_statistics=True,
    #         head_aggregation="max_pool",
    #         decoder_mode="mix_channel",
    #         mode="mix_channel",
    #         decoder_num_layers=2,
    #         decoder_d_model_layerwise=[0.25, 0.125],
    #         decoder_num_patches_layerwise_scale=[0.25, 0.25],
    #         num_channels_layerwise_scale=None,
    #         decoder_num_channels_layerwise_scale=[1, 0.5],
    #     )
    #     self.check_module(task="classification", params=params)

    # @parameterized.expand(
    #     list(
    #         itertools.product(
    #             [[5, 10, 29], None],
    #             [[1, 0.5], None],
    #             ["max_pool", "avg_pool"],
    #             [True, False],
    #         )
    #     )
    # )
    # def test_classification_categorical(
    #     self,
    #     categorical_vocab_size_list,
    #     decoder_num_channels_layerwise_scale,
    #     head_aggregation,
    #     inject_statistics,
    # ):
    #     params = self.__class__.params.copy()
    #     params.update(
    #         inject_statistics=inject_statistics,
    #         head_aggregation=head_aggregation,
    #         decoder_mode="mix_channel",
    #         mode="mix_channel",
    #         decoder_num_layers=2,
    #         decoder_d_model_layerwise=[0.25, 0.125],
    #         decoder_num_patches_layerwise_scale=[0.25, 0.25],
    #         num_channels_layerwise_scale=None,
    #         decoder_num_channels_layerwise_scale=decoder_num_channels_layerwise_scale,
    #         categorical_vocab_size_list=categorical_vocab_size_list,
    #     )
    #     self.check_module(task="classification", params=params)

    # def reconstruct_full_module(
    #     self, params=None, output_hidden_states=False, return_dict=None
    # ):
    #     config = TSPulseConfig(**params)

    #     mdl = TSPulseForReconstruction(config)
    #     target_val = self.__class__.correct_reconstruction_output

    #     if config.mode == "common_channel":
    #         enc_output = self.__class__.enc_output_cc
    #     else:
    #         enc_output = self.__class__.enc_output

    #     dec_output = self.__class__.dec_output

    #     output = mdl(
    #         self.__class__.data,
    #         output_hidden_states=output_hidden_states,
    #         return_dict=return_dict,
    #     )

    #     # print(mdl)
    #     # from torchsummary import summary

    #     # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #     # mdl = mdl.to(device)
    #     # summary(mdl, self.__class__.data.shape)

    #     if isinstance(output, tuple):
    #         output = TSPulseForReconstructionOutput(*output)

    #     if config.loss == "mse":
    #         self.assertEqual(output.reconstruction_outputs.shape, target_val.shape)
    #     enc_output_shape = list(enc_output.shape)
    #     dec_output_shape = list(dec_output.shape)

    #     if "total_embedding_size" in params and params["total_embedding_size"]:
    #         enc_output_shape = tuple(enc_output.shape)
    #         total_embedding_size = params["total_embedding_size"]

    #         if mdl.config.mode == "mix_channel":
    #             total_embedding_size = (
    #                 total_embedding_size * mdl.config.num_input_channels
    #             )
    #         # print(enc_output_shape)
    #         enc_output_shape = list(enc_output_shape[:-1] + (total_embedding_size,))
    #         enc_output = torch.rand(enc_output_shape)

    #     print("---", output.backbone_hidden_state.shape, enc_output_shape)
    #     self.assertEqual(list(output.backbone_hidden_state.shape), enc_output_shape)
    #     self.assertEqual(list(output.decoder_hidden_state.shape), dec_output_shape)

    #     # if output_hidden_states is True:
    #     #     print("ooo", len(output.hidden_states))
    #     #     self.assertEqual(len(output.hidden_states), params["num_layers"])

    #     # else:
    #     #     self.assertEqual(output.hidden_states, None)

    #     self.assertEqual(output.loss.item() < np.inf, True)

    # def test_reconstruction_full(self):
    #     self.check_module(
    #         task="reconstruction",
    #         params=self.__class__.params,
    #         output_hidden_states=True,
    #     )
    #     # self.reconstruct_full_module(self.__class__.params, output_hidden_states = True)

    def test_reconstruction_full_2(self):
        params = self.__class__.params.copy()
        # params.update(
        #     mode="mix_channel",
        #     # num_layers=4,
        #     # decoder_num_layers=4,
        #     decoder_mode="mix_channel",
        #     fuse_fft=True,
        #     mask_ratio=0.3,
        #     fft_mask_ratio=0.3,
        #     mask_type="point",
        # )
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

    # def test_masking(self):
    #     params = self.__class__.params.copy()
    #     params.update(
    #         mode="mix_channel",
    #         fuse_fft=True,
    #         mask_ratio=0.7,
    #         fft_mask_ratio=0.7,
    #         channel_register_tokens=None,
    #         patch_register_tokens=2,
    #         mask_type="block",
    #     )
    #     config = TSPulseConfig(**params)
    #     x = self.__class__.small_input  # Shape: [6, T, C]
    #     device = x.device

    #     def check_masking(masker, label):
    #         masked_x, mask = masker(x)
    #         assert masked_x.shape == x.shape
    #         assert mask.shape == x.shape
    #         if label == "full":
    #             assert mask.any(), "Expected some masking in full mode"
    #         elif label == "odd":
    #             for i in range(x.size(0)):
    #                 assert (
    #                     mask[i].any() if i % 2 == 1 else not mask[i].any()
    #                 ), f"{label} batch error at {i}"
    #         elif label == "even":
    #             for i in range(x.size(0)):
    #                 assert (
    #                     mask[i].any() if i % 2 == 0 else not mask[i].any()
    #                 ), f"{label} batch error at {i}"

    #     for mode in ["full", "odd", "even"]:
    #         masker = TSPulseMasking(config, device=device, batch_mode=mode)
    #         check_masking(masker, label=mode)

    #     fft_input = (
    #         torch.fft.fft(x.squeeze(-1), norm="ortho").unsqueeze(-1).to(device)
    #     )  # [B, T, 1]
    #     fft_input = torch.cat([fft_input.real, fft_input.imag], dim=1)  # [B, 2T, 1]

    #     for mode in ["full", "odd", "even"]:
    #         fft_masker = TSPulseFFTMasker(config, batch_mode=mode)
    #         masked_fft, fft_mask = fft_masker(fft_input.clone())
    #         assert masked_fft.shape == fft_input.shape
    #         assert fft_mask.shape == fft_input.shape
    #         if mode == "full":
    #             assert fft_mask.any(), "Expected some masking in full mode"
    #         elif mode == "odd":
    #             for i in range(fft_input.size(0)):
    #                 assert (
    #                     fft_mask[i].any() if i % 2 == 1 else not fft_mask[i].any()
    #                 ), f"FFT odd batch error at {i}"
    #         elif mode == "even":
    #             for i in range(fft_input.size(0)):
    #                 assert (
    #                     fft_mask[i].any() if i % 2 == 0 else not fft_mask[i].any()
    #                 ), f"FFT even batch error at {i}"

    # def test_masking_old(self):
    #     params = self.__class__.params.copy()
    #     params.update(
    #         mode="mix_channel",
    #         # num_layers=4,
    #         # decoder_num_layers=4,
    #         fuse_fft=True,
    #         mask_ratio=0.7,
    #         fft_mask_ratio=0.7,
    #         channel_register_tokens=None,
    #         patch_register_tokens=2,
    #         mask_type="block",
    #     )
    #     config = TSPulseConfig(**params)

    #     masker = TSPulseMasking(
    #         config, device=self.__class__.small_input.device, batch_mode="full"
    #     )
    #     x = self.__class__.small_input
    #     masked_x, mask = masker(x)

    #     assert masked_x.shape == x.shape
    #     assert mask.shape == x.shape
    #     assert mask.any(), "At least some elements should be masked in 'full' mode"

    #     masker = TSPulseMasking(
    #         config, device=self.__class__.small_input.device, batch_mode="odd"
    #     )

    #     masked_x, mask = masker(x)

    #     assert masked_x.shape == x.shape
    #     assert mask.shape == x.shape
    #     # Only odd indices [1, 3, 5] should be masked
    #     for i in range(6):
    #         if i % 2 == 0:
    #             assert not mask[i].any(), f"Even sample {i} should NOT be masked"
    #         else:
    #             assert mask[i].any(), f"Odd sample {i} should be masked"

    #     masker = TSPulseMasking(
    #         config, device=self.__class__.small_input.device, batch_mode="even"
    #     )

    #     masked_x, mask = masker(x)

    #     assert masked_x.shape == x.shape
    #     assert mask.shape == x.shape
    #     # Only odd indices [1, 3, 5] should be masked
    #     for i in range(6):
    #         if i % 2 == 1:
    #             assert not mask[i].any(), f"Even sample {i} should NOT be masked"
    #         else:
    #             assert mask[i].any(), f"Odd sample {i} should be masked"

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

    # def test_explicit_masking(self):
    #     params = self.__class__.params.copy()
    #     params.update(
    #         mode="mix_channel",
    #         fuse_fft=True,
    #         mask_ratio=0.3,
    #         fft_mask_ratio=0.3,
    #         channel_register_tokens=None,
    #         patch_register_tokens=2,
    #     )

    #     mdl = TSPulseForReconstruction(TSPulseConfig(**params))

    #     # ---- CASE 1: 1D Shared Mask for All Batch Samples ----
    #     shared_mask = torch.tensor([0, 1, 2])  # Mask patches 0, 1, 2 for all samples
    #     output_shared = mdl(
    #         self.__class__.correct_reconstruction_output,
    #         explicit_mask_positions=shared_mask,
    #     )

    #     b, l, c = output_shared.mask.shape
    #     patch_size = mdl.config.patch_length
    #     num_masked_positions = len(shared_mask) * patch_size
    #     expected_shared = (
    #         torch.arange(l, device=output_shared.mask.device).unsqueeze(0).unsqueeze(-1)
    #     )  # [1, l, 1]
    #     expected_shared = (
    #         expected_shared < num_masked_positions
    #     )  # True for first k*patch_len, False after

    #     expected_shared = expected_shared.expand(b, l, c)
    #     # Ensure all batch samples are masked identically
    #     assert torch.all(output_shared.mask == expected_shared), "Shared mask failed"

    #     # ---- CASE 2: 2D Per-Sample Mask ----
    #     per_sample_mask = torch.tensor(
    #         [
    #             [0, 1],  # Sample 0 masks patch 0 and 1
    #             [2, 3],  # Sample 1 masks patch 2 and 3
    #         ],
    #         device=output_shared.mask.device,
    #     )

    #     output_per_sample = mdl(
    #         self.__class__.correct_reconstruction_output[:2],
    #         explicit_mask_positions=per_sample_mask,
    #     )

    #     b2, l2, c2 = output_per_sample.mask.shape

    #     expected_per_sample = torch.zeros(
    #         (b2, l2, c2), dtype=torch.bool, device=output_per_sample.mask.device
    #     )

    #     for i, masked_patches in enumerate(per_sample_mask):
    #         for patch_idx in masked_patches:
    #             start = patch_idx * patch_size
    #             end = (patch_idx + 1) * patch_size
    #             expected_per_sample[i, start:end, :] = True

    #     assert torch.all(
    #         output_per_sample.mask == expected_per_sample
    #     ), "Per-sample mask failed"

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

        # masked_elements = mask.sum().item()
        # expected = int(total_elements * params["mask_ratio"])
        # self.assertTrue(abs(masked_elements - expected) / total_elements < 0.1)

        # === Test with mask_percentage = 1.0
        # masked_tensor, mask = (
        #     mdl.backbone.time_masker.variable_length_hybrid_masking_with_token(
        #         tensor=x,
        #         mask_percentage=1.0,
        #         patch_size=patch_size,
        #         full_patch_mask_percentage=params["full_patch_mask_percentage"],
        #     )
        # )
        # self.assertTrue(torch.all(mask))

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

        # # future_values=self.__class__.future_values,
        # masked_tensor, mask = mdl.backbone.time_masker(
        #     x, past_observed_mask=past_observed_mask
        # )

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

    # def test_explict_masking(self):
    #     params = self.__class__.params.copy()
    #     params.update(
    #         mode="mix_channel",
    #         # num_layers=4,
    #         # decoder_num_layers=4,
    #         fuse_fft=True,
    #         mask_ratio=0.3,
    #         fft_mask_ratio=0.3,
    #         channel_register_tokens=None,
    #         patch_register_tokens=2,
    #     )

    #     mdl = TSPulseForReconstruction(TSPulseConfig(**params))
    #     output = mdl(
    #         self.__class__.correct_reconstruction_output,
    #         explicit_mask_positions=torch.tensor([0, 1, 2]),
    #     )
    #     b, l, c = output.mask.shape
    #     expected = (
    #         torch.arange(l, device=output.mask.device).unsqueeze(0).unsqueeze(-1)
    #     )  # [1, l, 1]
    #     expected = expected < 6  # [1, l, 1] --> True for first k, False after
    #     assert torch.all(output.mask == expected)

    # self.reconstruct_full_module(params, output_hidden_states=True)

    # def test_components(self):
    #     params = self.__class__.params.copy()
    #     params.update(
    #         mode="mix_channel",
    #         num_layers=4,
    #         decoder_num_layers=4,
    #         d_model_layerwise_compression_scale=[1, 0.75, 0.5, 0.25],
    #         decoder_d_model_layerwise_expansion_scale=[0.25, 0.5, 0.75, 1],
    #     )

    #     config = TSPulseConfig(**params)

    #     encoder = TSPulseModel(config)
    #     decoder = TSPulseDecoderWithReconstructionHead(config)

    #     encoder_output = encoder(self.__class__.data)

    #     reconstructed_output = decoder(
    #         decoder_input=encoder_output.last_hidden_flatten_state, loc=encoder_output.loc, scale=encoder_output.scale
    #     )
    #     self.assertEqual(self.__class__.data.shape, reconstructed_output.reconstruction_outputs.shape)
    #     enc_output_shape = list(self.__class__.enc_output.shape)
    #     # self.assertEqual(list(encoder_output.last_hidden_flatten_state.shape), enc_output_shape)
