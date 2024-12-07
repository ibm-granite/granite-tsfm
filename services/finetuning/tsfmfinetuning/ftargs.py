import argparse
import tempfile
from pathlib import Path


def argparser():
    parser = argparse.ArgumentParser()

    # ################Input data
    parser.add_argument(
        "--payload",
        "-p",
        type=Path,
        help="A json file containing the service request payload.",
        required="true",
    )
    parser.add_argument(
        "--target_dir",
        "-d",
        type=Path,
        help="A target directory where the fine-tuned model should be written.",
        default=tempfile.gettempdir(),
    )
    parser.add_argument(
        "--model_name",
        "-n",
        type=str,
        help="The name that should be used for the finetuned model.",
        required=True,
    )
    parser.add_argument(
        "--config_file",
        "-c",
        type=str,
        help="The tsfm configuration file (yaml).",
    )
    # #################Model archicture
    arch_choices = ["ttm"]
    model_arch_type = parser.add_mutually_exclusive_group(
        required=True,
    )
    model_arch_type.add_argument(
        "--model_arch",
        "-a",
        choices=arch_choices,
        help=f"""The base model architecture. The following are currently supported: {",".join(arch_choices)}""",
    )

    # ####################Task type
    task_choices = ["forecasting"]
    model_task_choices = parser.add_mutually_exclusive_group(required=True)
    model_task_choices.add_argument(
        "--task_type",
        "-t",
        choices=task_choices,
        help=f"""The base model architecture. The following are currently supported: {",".join(task_choices)}""",
    )
    return parser
