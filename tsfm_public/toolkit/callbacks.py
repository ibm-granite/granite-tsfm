"""Some basic callbacks for training with HF Trainer"""

import time

import numpy as np
from transformers import TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments


class TrackingCallback(TrainerCallback):
    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        verbose: bool = False,
        **kwargs,
    ):
        self.all_epoch_times = []
        self.train_start_time = time.time()
        self.verbose = verbose
        return super().on_train_begin(args, state, control, **kwargs)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self.train_end_time = time.time()
        self.mean_epoch_time = np.mean(self.all_epoch_times)
        self.total_train_time = self.train_end_time - self.train_start_time
        self.best_eval_metric = state.best_metric
        print(
            f"[{self.__class__.__name__}] Mean Epoch Time = {self.mean_epoch_time} seconds, Total Train Time = {self.total_train_time}"
        )
        return super().on_train_end(args, state, control, **kwargs)

    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self.epoch_start_time = time.time()
        return super().on_epoch_begin(args, state, control, **kwargs)

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self.epoch_end_time = time.time()
        self.last_epoch_time = self.epoch_end_time - self.epoch_start_time
        if self.verbose:
            print(f"[{self.__class__.__name__}] Epoch Time = {self.last_epoch_time} seconds")
        self.all_epoch_times.append(self.last_epoch_time)
        return super().on_epoch_end(args, state, control, **kwargs)
