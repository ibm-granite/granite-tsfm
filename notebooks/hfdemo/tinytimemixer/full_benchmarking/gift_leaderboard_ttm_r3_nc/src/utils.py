# Copyright contributors to the TSFM project
#
import argparse
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from gluonts.time_feature.seasonality import get_seasonality
from transformers import Trainer
from transformers.utils import logging


logger = logging.get_logger(__name__)

CUSTOM_SEASONALITIES = {
    "S": 60,  # 3600,  # 1 hour
    "s": 60,  # 3600,  # 1 hour
    "T": 60,  # 1440,  # 1 day
    "min": 60,  # 1440,  # 1 day
    "H": 24,  # 1 day
    "h": 24,  # 1 day
    "D": 1,  # 1 day
    "W": 1,  # 1 week
    "M": 12,
    "ME": 12,
    "B": 5,
    "Q": 4,
    "QE": 4,
}


class CustomMASETrainer(Trainer):
    def __init__(self, *args, freq=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.seasonal_period = get_seasonality(freq=freq, seasonalities=CUSTOM_SEASONALITIES)
        logger.info(f"Finetuning with MASE loss with freq={freq}, seasonality = {self.seasonal_period}")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute the MASE loss using the seasonal denominator based on in-sample values.
        """
        # Extract in-sample values from inputs
        in_sample_values = inputs["past_values"]  # Ensure in-sample values are passed in the batch

        # Compute the denominator (mean absolute seasonal error)
        diffs = torch.abs(
            in_sample_values[:, self.seasonal_period :, :] - in_sample_values[:, : -self.seasonal_period, :]
        )
        scale_factor = torch.mean(diffs)

        # Forward pass
        outputs = model(**inputs)
        preds = outputs.prediction_outputs
        labels = inputs["future_values"]

        # Compute the numerator of MASE
        numerator = torch.mean(torch.abs(preds - labels))

        # Compute MASE
        mase_loss = numerator / scale_factor

        return (mase_loss, outputs) if return_outputs else mase_loss


def plot_forecast(
    test_data_input,
    test_data_label,
    forecast_samples,
    prediction_length,
    ds_name,
    term,
    out_dir,
    probabilistic: bool = False,
    quantile_keys=[],
):
    test_data_input_list = list(test_data_input)
    test_data_label_list = list(test_data_label)
    ds_name = ds_name.replace("/", "__")
    plot_dir = f"{out_dir}/{ds_name}__{term}__{prediction_length}"
    num_plots = 10
    plot_context = int(4 * prediction_length)
    channel = 0

    # Set a more beautiful style
    plt.style.use("seaborn-v0_8-whitegrid")

    # Adjust figure size and subplot spacing
    assert num_plots >= 1
    num_plots = min(num_plots, len(forecast_samples))
    fig, axs = plt.subplots(num_plots, 1, figsize=(10, 2 * num_plots))
    if num_plots == 1:
        axs = [axs]

    indices = np.random.choice(len(forecast_samples), size=num_plots, replace=False)

    for i, index in enumerate(indices):
        sample = test_data_input_list[index]
        label = test_data_label_list[index]
        if len(sample["target"].shape) == 1:
            sample["target"] = sample["target"].copy()
            sample["target"] = sample["target"].reshape(1, -1)
        if label["target"].ndim == 1:
            label["target"] = label["target"].reshape(1, -1)
        feasible_plot_context = min(plot_context, sample["target"].shape[1])
        ts_y_hat = np.arange(feasible_plot_context, feasible_plot_context + prediction_length)
        if probabilistic:
            if forecast_samples.ndim == 4:
                y_hat = forecast_samples[index, :, :, channel]
            else:
                y_hat = forecast_samples[index, :, :]

        else:
            if len(forecast_samples.shape) == 3:
                y_hat = forecast_samples[index, :, channel]
            else:
                y_hat = forecast_samples[index, :]

        ts_y = np.arange(feasible_plot_context + prediction_length)
        y = label["target"][channel, :]
        x = sample["target"][channel, -feasible_plot_context:]
        y = np.concatenate((x, y), axis=0)
        border = feasible_plot_context
        plot_title = f"Example {indices[i]}"

        # Plot predicted values with a dashed line
        if probabilistic:
            for qi in [0, 8, 9]:
                axs[i].plot(
                    ts_y_hat,
                    y_hat[qi, :],
                    label=f"Predicted: q_{quantile_keys[qi]}",
                    linestyle="--",
                    # color="orange",
                    linewidth=2,
                )
        else:
            axs[i].plot(
                ts_y_hat,
                y_hat,
                label="Predicted",
                linestyle="--",
                color="orange",
                linewidth=2,
            )

        # Plot true values with a solid line
        axs[i].plot(ts_y, y, label="True", linestyle="-", color="blue", linewidth=2)

        # Plot horizon border
        if border is not None:
            axs[i].axvline(x=border, color="r", linestyle="-")

        axs[i].set_title(plot_title)
        axs[i].legend()

    # Adjust overall layout
    plt.tight_layout()

    # Save the plot
    if plot_dir is not None:
        plot_filename = f"ch_{str(channel)}.pdf"
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, plot_filename))


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args():
    # ## Important arguments

    # Create the parser
    parser = argparse.ArgumentParser(description="A simple argparse example.")

    # Add arguments
    parser.add_argument(
        "--out_dir",
        "-od",
        type=str,
        help="Out dir prefix name",
        required=False,
        default="ttm",
    )
    parser.add_argument(
        "--model_path",
        "-mp",
        type=str,
        help="Path to TTM model card",
        required=False,
        default="ibm-granite/granite-timeseries-ttm-r2",
    )
    parser.add_argument(
        "--decoder_mode",
        "-dm",
        type=str,
        help="Decoder mode",
        required=False,
        default="common_channel",
    )
    parser.add_argument(
        "--model_dict_path",
        "-mdp",
        type=str,
        help="Model dict path",
        required=False,
        default=None,
    )

    parser.add_argument(
        "--model_map_path",
        "-mmp",
        type=str,
        help="Model map path",
        required=False,
        default=None,
    )

    parser.add_argument(
        "--few_shot_data_limit_config",
        "-fsdlc",
        type=str,
        help="Few shot data limit config path",
        required=False,
        default=None,
    )

    parser.add_argument(
        "--use_zs_anchor",
        "-uza",
        action="store_true",
        help="Enable zero-shot anchor regularization during fine-tuning.",
    )

    parser.add_argument(
        "--disable_extra_point_weightage",
        "-depw",
        action="store_true",
        help="disable_extra_point_weightage",
    )

    parser.add_argument(
        "--enable_smart_rolling",
        "-esr",
        action="store_true",
        help="enable_smart_rolling",
    )

    parser.add_argument(
        "--zs_anchor_weight",
        "-aw",
        type=float,
        default=0.01,
        help="Weight of zero-shot anchor penalty applied to the 0.5 quantile forecast.",
    )

    parser.add_argument(
        "--head_dropout",
        "-hdr",
        type=float,
        help="Head dropout",
        required=False,
        default=0.2,
    )
    parser.add_argument(
        "--fewshot_fraction",
        "-ff",
        type=float,
        help="Fewshot fraction",
        required=False,
        default=0.2,
    )
    parser.add_argument(
        "--huber_delta",
        "-de",
        type=float,
        help="Huber delta",
        required=False,
        default=1.0,
    )
    parser.add_argument(
        "--quantile",
        "-q",
        type=float,
        help="Quantile for pinball loss",
        required=False,
        default=0.5,
    )
    parser.add_argument(
        "--upper_bound_fewshot_samples",
        "-ubfs",
        type=int,
        help="Upper bound fewshot samples",
        required=False,
        default=0,
    )
    # parser.add_argument(
    #     "--freeze_backbone",
    #     "-fb",
    #     type=int,
    #     help="Freeze TTM backbone",
    #     required=False,
    #     default=0,
    # )

    # parser.add_argument(
    #     "--only_prefix_tune",
    #     "-opt",
    #     type=int,
    #     help="Only prefix tune",
    #     required=False,
    #     default=0,
    # )

    parser.add_argument(
        "--bias_tune",
        "-bt",
        type=int,
        help="bias tune",
        required=False,
        default=0,
    )

    parser.add_argument(
        "--calibrator_tune",
        "-ct",
        type=int,
        help="calibrator_tune",
        required=False,
        default=0,
    )

    parser.add_argument(
        "--ft_zs_ensemble",
        "-fze",
        type=int,
        help="ft_zs_ensemble",
        required=False,
        default=0,
    )

    parser.add_argument(
        "--ft_zs_ensemble_mode",
        "-fze_mode",
        type=str,
        help="ft_zs_ensemble_mode",
        required=False,
        default="median",
    )

    parser.add_argument(
        "--ensemble_shrink_lambda",
        "-fze_ratio",
        type=float,
        help="ensemble_shrink_lambda",
        required=False,
        default=0.7,
    )

    parser.add_argument(
        "--decoder_tune",
        "-dt",
        type=int,
        help="decoder tune",
        required=False,
        default=0,
    )

    parser.add_argument(
        "--head_tune",
        "-ht",
        type=int,
        help="Head tune",
        required=False,
        default=1,
    )

    parser.add_argument(
        "--quantile_tune",
        "-qt",
        type=int,
        help="Quantile tune",
        required=False,
        default=1,
    )

    parser.add_argument(
        "--norm_tune",
        "-nt",
        type=int,
        help="Norm tune",
        required=False,
        default=0,
    )

    parser.add_argument(
        "--enable_staging",
        "-es",
        type=int,
        help="Enable staging",
        required=False,
        default=0,
    )

    parser.add_argument(
        "--rolling_norm",
        "-rn",
        type=int,
        help="Rolling norm",
        required=False,
        default=0,
    )

    parser.add_argument(
        "--patch_tune",
        "-pat",
        type=int,
        help="Patch tune",
        required=False,
        default=0,
    )

    parser.add_argument(
        "--prefix_tune",
        "-prt",
        type=int,
        help="prefix_tune",
        required=False,
        default=0,
    )

    parser.add_argument(
        "--backbone_tune",
        "-bkt",
        type=int,
        help="backbone_tune",
        required=False,
        default=0,
    )

    parser.add_argument(
        "--batch_size",
        "-bs",
        type=int,
        help="Batch Size",
        required=False,
        default=None,
    )

    parser.add_argument(
        "--fewshot_max_samples",
        "-fms",
        type=int,
        help="fewshot_max_samples",
        required=False,
        default=None,
    )

    parser.add_argument(
        "--valid_max_samples",
        "-vms",
        type=int,
        help="valid_max_samples",
        required=False,
        default=None,
    )

    parser.add_argument(
        "--fewshot_location",
        "-fl",
        type=str,
        help="Fewshot location: rand/end/start",
        required=False,
        default="rand",
    )

    parser.add_argument(
        "--ttm_version",
        "-tv",
        type=str,
        help="ttm_version",
        required=False,
        default="test",
    )

    parser.add_argument(
        "--loss",
        "-l",
        type=str,
        help="Finetune loss",
        required=False,
        default="mae",
    )
    parser.add_argument(
        "--learning_rate",
        "-lr",
        type=float,
        help="Learning rate",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--force_short_context",
        "-fsc",
        type=int,
        help="Force short context for short-term series",
        required=False,
        default=0,
    )
    parser.add_argument(
        "--use_exogs",
        "-ue",
        type=int,
        help="Use any available past exogeneous series",
        required=False,
        default=0,
    )
    parser.add_argument(
        "--use_mask",
        "-um",
        type=int,
        help="Use past_observed_mask in TTM",
        required=False,
        default=0,
    )
    parser.add_argument(
        "--min_context_mult",
        "-mcm",
        type=int,
        help="Minimum context (multiple of horizon) needed for forcing short context",
        required=False,
        default=4,
    )
    parser.add_argument(
        "--num_epochs",
        "-ne",
        type=int,
        help="Number of finetune epochs",
        required=False,
        default=20,
    )
    parser.add_argument(
        "--insample_forecast",
        "-if",
        type=int,
        help="insample prediction to generate quantile forcasts",
        required=False,
        default=1,
    )
    parser.add_argument(
        "--insample_use_train",
        "-iut",
        type=int,
        help="Use training data to calculate insample error statistics",
        required=False,
        default=1,
    )
    parser.add_argument(
        "--use_valid_from_train",
        "-uvft",
        type=int,
        help="Use validation from train",
        required=False,
        default=1,
    )
    parser.add_argument(
        "--mem_per_proc",
        "-mpp",
        type=int,
        help="Memory per ray job (in GB)",
        required=False,
        default=80,
    )
    parser.add_argument(
        "--automate_fewshot_fraction",
        "-aff",
        type=int,
        help="Automatically decide fewshot fraction",
        required=False,
        default=1,
    )
    parser.add_argument(
        "--automate_fewshot_fraction_threshold",
        "-afft",
        type=int,
        help="Threshold on number of samples to automatically decide fewshot fraction",
        required=False,
        default=200,
    )
    parser.add_argument(
        "--short_datasets",
        "-sd",
        type=str,
        help="String of short dataset names separated by spaces",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--med_long_datasets",
        "-mld",
        type=str,
        help="String of medium/long dataset names separated by spaces",
        required=False,
        default=None,
    )

    parser.add_argument(
        "--plot_predictions",
        "-pp",
        type=int,
        help="Plot Pred",
        required=False,
        default=0,
    )

    parser.add_argument(
        "--auto_sampler",
        "-as",
        type=int,
        help="Auto Sampler",
        required=False,
        default=0,
    )

    parser.add_argument(
        "--calibrate",
        "-cl",
        type=int,
        help="Calivrate",
        required=False,
        default=0,
    )

    parser.add_argument(
        "--affine_quantile_adapter",
        "-aqa",
        type=int,
        help="affine_quantile_adapter",
        required=False,
        default=0,
    )

    parser.add_argument(
        "--ensemble_point_w",
        type=float,
        default=1.0,
        help="Weight for point accuracy (MASE)",
    )

    parser.add_argument(
        "--ensemble_pinball_w",
        type=float,
        default=0.0,
        help="Weight for quantile pinball loss",
    )

    parser.add_argument(
        "--ensemble_width_w",
        type=float,
        default=0.3,
        help="Penalty weight for prediction interval width",
    )

    parser.add_argument(
        "--ensemble_cover_w",
        type=float,
        default=0.4,
        help="Penalty weight for coverage calibration",
    )

    parser.add_argument(
        "--ensemble_stability_w",
        type=float,
        default=0.1,
        help="Penalty weight for instability across windows",
    )

    parser.add_argument(
        "--ensemble_softmax_temp",
        type=float,
        default=10.0,
        help="Softmax temperature for model weighting (higher = sharper selection)",
    )

    parser.add_argument(
        "--use_lite",
        "-ul",
        type=int,
        help="Use Lite TTM R3 Models",
        required=False,
        default=0,
    )

    parser.add_argument(
        "--torch_matmul_precision",
        "-tmp",
        type=str,
        help="Precision for torch float32 matrix multiplication (highest, high, medium)",
        required=False,
        default="highest",
    )

    if "ipykernel" in sys.modules:  # Check if running in a Jupyter environment
        # For Jupyter, provide default arguments
        args = parser.parse_args("")
    else:
        args = parser.parse_args()  # Normal command-line parsing

    args.upper_bound_fewshot_samples = bool(args.upper_bound_fewshot_samples)
    # args.freeze_backbone = bool(args.freeze_backbone)
    args.force_short_context = bool(args.force_short_context)
    args.use_exogs = bool(args.use_exogs)
    args.use_mask = bool(args.use_mask)
    args.insample_forecast = bool(args.insample_forecast)
    args.insample_use_train = bool(args.insample_use_train)
    args.use_valid_from_train = bool(args.use_valid_from_train)
    args.automate_fewshot_fraction = bool(args.automate_fewshot_fraction)
    args.plot_predictions = bool(args.plot_predictions)
    args.calibrate = bool(args.calibrate)
    args.auto_sampler = bool(args.auto_sampler)
    args.affine_quantile_adapter = bool(args.affine_quantile_adapter)
    args.use_lite = bool(args.use_lite)

    return args


def delete_if_empty_csv(file_path):
    # Check if file exists
    if os.path.exists(file_path):
        # Check if file is empty (0 bytes)
        if os.path.getsize(file_path) == 0:
            os.remove(file_path)
            print(f"Deleted empty file: {file_path}")
        else:
            print(f"File is not empty: {file_path}")
    else:
        print(f"File does not exist: {file_path}")
