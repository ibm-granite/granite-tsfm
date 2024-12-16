# Copyright contributors to the TSFM project
#

# Standard
import json
import os
from datetime import datetime

# Third Party
from transformers import TrainerCallback


class FileLoggingCallback(TrainerCallback):
    """Exports metrics, e.g., training loss to a file in the checkpoint directory."""

    def __init__(self, logs_filename):
        self.training_logs_filename = logs_filename

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Checks if this log contains keys of interest, e.g., loss, and if so, creates
        training_logs.jsonl in the model output dir (if it doesn't already exist),
        appends the subdict of the log & dumps the file.
        """
        # All processes get the logs from this node; only update from process 0.
        if not state.is_world_process_zero:
            return

        log_file_path = os.path.join(args.output_dir, self.training_logs_filename)
        if logs is not None and "loss" in logs and "epoch" in logs:
            self._track_loss("loss", "training_loss", log_file_path, logs, state)
        elif logs is not None and "eval_loss" in logs and "epoch" in logs:
            self._track_loss("eval_loss", "validation_loss", log_file_path, logs, state)

    def _track_loss(self, loss_key, log_name, log_file, logs, state):
        try:
            # Take the subdict of the last log line; if any log_keys aren't part of this log
            # object, assume this line is something else, e.g., train completion, and skip.
            log_obj = {
                "name": log_name,
                "data": {
                    "epoch": round(logs["epoch"], 2),
                    "step": state.global_step,
                    "value": logs[loss_key],
                    "timestamp": datetime.isoformat(datetime.now()),
                },
            }
        except KeyError:
            return

        # append the current log to the jsonl file
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"{json.dumps(log_obj, sort_keys=True)}\n")
