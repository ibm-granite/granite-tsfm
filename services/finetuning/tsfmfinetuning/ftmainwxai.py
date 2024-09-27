# Standard
import argparse
import json
import tempfile
from pathlib import Path

from .ftcommon import forecasting_tuning_to_local
from .ftpayloads import TinyTimeMixerForecastingTuneInput, TuneOutput


# remote container space
def main():
    parser = argparse.ArgumentParser()

    # ################Input data
    parser.add_argument(
        "--input_params_json",
        "-i",
        type=Path,
        help="A json file containing fine-tuning configuration parameters.",
        required="true",
    )
    parser.add_argument(
        "--target_dir",
        "-t",
        type=Path,
        help="A target directory for the finetuned model.",
        default=tempfile.gettempdir(),
    )
    parser.add_argument(
        "--model_name", "-m", type=str, help="The name that should be used for the finetuned model.", required=True
    )
    # #################Model archicture
    arch_choices = ["ttm"]
    model_arch_type = parser.add_mutually_exclusive_group(
        help=f"""The base model architecture. The following are currently supported: {",".join(arch_choices)}"""
    )
    model_arch_type.add_argument("--model_arch", "-a", choices=arch_choices, required=True)

    # ####################Task type
    task_choices = ["forecasting"]
    task_choices = parser.add_mutually_exclusive_group(
        help=f"""The base model architecture. The following are currently supported: {",".join(task_choices)}"""
    )
    task_choices.add_argument("--task_type", "-t", choices=task_choices, required=True)

    # ###########################3
    args, _ = parser.parse_known_args()

    # reconstruct our input object
    # these come from a mutually exclusive group
    params: dict = json.loads(args.input_params_json)
    # this will give us param validation

    if args.model_arch_type == "ttm" and args.task_type == "forecasting":
        tune_input: TinyTimeMixerForecastingTuneInput = TinyTimeMixerForecastingTuneInput(**params)
        tune_output: TuneOutput = forecasting_tuning_to_local(
            input=tune_input, target_dir=args.target_dir, model_name=args.model_name
        )
        print(tune_output)
    else:
        raise NotImplementedError(f"model arch/task type not implemented {args.model_arch_type} {args.task_type}")


if __name__ == "__main__":
    main()
