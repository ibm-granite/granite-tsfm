# Standard
import json
import traceback

import yaml

from tsfmfinetuning import TSFM_CONFIG_FILE
from tsfmfinetuning.finetuning import FinetuningRuntime
from tsfmfinetuning.ftargs import argparser
from tsfmfinetuning.ftpayloads import TinyTimeMixerForecastingTuneInput


# remote container space
def main() -> int:
    # ###########################3
    args, _ = argparser().parse_known_args()

    # reconstruct our input object
    # these come from a mutually exclusive group
    payload: dict = json.load(open(args.payload, "r"))
    # this will give us param validation

    if args.model_arch == "ttm" and args.task_type == "forecasting":
        input: TinyTimeMixerForecastingTuneInput = TinyTimeMixerForecastingTuneInput(**payload)
        config_file = args.config_file if args.config_file else TSFM_CONFIG_FILE
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)
        ftr: FinetuningRuntime = FinetuningRuntime(config=config)
        ftr.finetuning(input=input, tuned_model_name=args.model_name, output_dir=args.target_dir)
        return 0

    else:
        raise NotImplementedError(f"model arch/task type not implemented {args.model_arch_type} {args.task_type}")


if __name__ == "__main__":
    try:
        exit(main())
    except Exception as e:
        traceback.print_exception(e)
        exit(1)
