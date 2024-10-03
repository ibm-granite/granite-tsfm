# Standard
import argparse
import json
import tempfile
from pathlib import Path

import yaml

from tsfmfinetuning import TSFM_CONFIG_FILE
from tsfmfinetuning.finetuning import FinetuningRuntime
from tsfmfinetuning.ftpayloads import TinyTimeMixerForecastingTuneInput


# remote container space
def main():
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

    # ###########################3
    args, _ = parser.parse_known_args()

    # reconstruct our input object
    # these come from a mutually exclusive group
    payload: dict = json.loads(args.json_payload)
    # this will give us param validation

    if args.model_arch_type == "ttm" and args.model_task_choices == "forecasting":
        input: TinyTimeMixerForecastingTuneInput = TinyTimeMixerForecastingTuneInput(**payload)
        config_file = args.config_file if args.config_file else TSFM_CONFIG_FILE
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)
        ftr: FinetuningRuntime = FinetuningRuntime(config=config)
        ftr.finetuning(input=input, tuned_model_name=args.model_name, output_dir=args.target_dir)

    else:
        raise NotImplementedError(f"model arch/task type not implemented {args.model_arch_type} {args.task_type}")


if __name__ == "__main__":
    main()
