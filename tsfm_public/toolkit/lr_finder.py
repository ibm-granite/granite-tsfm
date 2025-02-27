# Copyright contributors to the TSFM project
#
"""Functions to identify candidate learning rates"""

import inspect
import os
import uuid
from cmath import inf
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from transformers import PreTrainedModel
from transformers.data.data_collator import default_data_collator
from transformers.trainer_utils import RemoveColumnsCollator
from transformers.utils import logging


logger = logging.get_logger(__name__)


def join_path_file(file: Union[str, Path], path: Union[str, Path], ext: str = ""):
    "Return `path/file` if file is a string or a `Path`, file otherwise"
    if not isinstance(file, (str, Path)):
        return file
    if not isinstance(path, Path):
        path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{file}{ext}"


def get_model(model: Union[nn.Module, PreTrainedModel]):
    "Return the model maybe wrapped inside `model`."
    return model.module if isinstance(model, (DistributedDataParallel, nn.DataParallel)) else model


def save_model(
    path,
    model: Union[nn.Module, PreTrainedModel],
    opt: Optional[Optimizer],
    with_opt: bool = True,
    pickle_protocol: int = 2,
):
    "Save `model` to `file` along with `opt` (if available, and if `with_opt`)"
    if opt is None:
        with_opt = False
    state = get_model(model).state_dict()
    if with_opt:
        state = {"model": state, "opt": opt.state_dict()}
    torch.save(state, path, pickle_protocol=pickle_protocol)


def load_model(
    path, model, opt: Optional[Optimizer] = None, with_opt: bool = False, device: str = "cpu", strict: bool = True
) -> nn.Module:
    "load the saved model"
    state = torch.load(path, map_location=device)
    if not opt:
        with_opt = False
    model_state = state["model"] if with_opt else state
    get_model(model).load_state_dict(model_state, strict=strict)
    if with_opt:
        opt.load_state_dict(state["opt"])
    model = model.to(device)


class LinearLR(_LRScheduler):
    """Linearly increases the learning rate between two boundaries over a number of iterations.

    Arguments:
        optimizer (torch.optim.Optimizer): wrapped optimizer.
        end_lr (float): the final learning rate.
        num_iter (int): the number of iterations over which the test occurs.
        last_epoch (int, optional): the index of last epoch. Default: -1.
    """

    def __init__(self, optimizer: Optimizer, end_lr: float, num_iter: int, last_epoch: int = -1):
        self.end_lr = end_lr
        if num_iter <= 1:
            raise ValueError("`num_iter` must be larger than 1")
        self.num_iter = num_iter
        super(LinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        r = (self.last_epoch + 1) / (self.num_iter - 1)
        return [base_lr + r * (self.end_lr - base_lr) for base_lr in self.base_lrs]


class ExponentialLR(_LRScheduler):
    """Exponentially increases the learning rate between two boundaries over a number of iterations.

    Arguments:
        optimizer (torch.optim.Optimizer): wrapped optimizer.
        end_lr (float): the final learning rate.
        num_iter (int): the number of iterations over which the test occurs.
        last_epoch (int, optional): the index of last epoch. Default: -1.
    """

    def __init__(self, optimizer: Optimizer, end_lr: float, num_iter: int, last_epoch: int = -1):
        self.end_lr = end_lr
        self.last_epoch = last_epoch
        if num_iter <= 1:
            raise ValueError("`num_iter` must be larger than 1")
        self.num_iter = num_iter
        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        r = (self.last_epoch + 1) / (self.num_iter - 1)
        return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]


def valley(lrs: List[float], losses: List[float]) -> float:
    "Suggests a learning rate from the longest valley and returns its index"
    n = len(losses)
    max_start, max_end = 0, 0

    # find the longest valley
    lds = [1] * n
    for i in range(1, n):
        for j in range(0, i):
            if (losses[i] < losses[j]) and (lds[i] < lds[j] + 1):
                lds[i] = lds[j] + 1
            if lds[max_end] < lds[i]:
                max_end = i
                max_start = max_end - lds[max_end]

    sections = (max_end - max_start) / 3
    idx = max_start + int(sections) + int(sections / 2)

    return float(lrs[idx])


class LRFinder:
    def __init__(
        self,
        model: Union[nn.Module, PreTrainedModel],
        opt: Optimizer,
        device: str = "cuda",
        start_lr: float = 1e-7,
        end_lr: float = 10,
        num_iter: int = 100,
        step_mode: str = "exp",
        beta: float = 0.98,
        suggestion: str = "valley",
        enable_prefix_tuning: bool = False,
    ):
        self.model = model
        self.opt = opt
        self.device = device
        self.start_lr, self.end_lr = start_lr, end_lr
        self.num_iter = num_iter
        self.step_mode = step_mode
        if beta >= 1:
            raise ValueError("`num_iter` must be smaller than 1")
        else:
            self.beta = beta
        self.suggestion = suggestion
        self.recorder = {}
        self.enable_prefix_tuning = enable_prefix_tuning

    def save(self, fname: Union[str, Path], path: Union[str, Path], **kwargs) -> str:
        """
        Save model and optimizer state (if `with_opt`) to `self.path/file`
        """
        fname = join_path_file(fname, path, ext=".pth")
        save_model(fname, self.model, getattr(self, "opt", None), **kwargs)
        return fname

    def load(
        self, fname: Union[str, Path], with_opt: bool = False, device: str = "cuda", strict: bool = True, **kwargs
    ):
        """
        load the model
        """
        if not torch.cuda.is_available():
            device = "cpu"
        load_model(fname, self.model, self.opt, with_opt, device=device, strict=strict)

    def before_fit(self):
        self.model.to(self.device)
        self.loss_func = nn.MSELoss()

        self.losses, self.lrs = [], []
        self.best_loss, self.aver_loss = inf, 0
        self.train_iter = 0

        # save model to load back after fitting
        uid = uuid.uuid4().hex
        self.temp_path = self.save("current_{}".format(uid), "temp", with_opt=False)
        # set base_lr for the optimizer
        self.set_lr(self.start_lr)

        # check num_iter
        if not self.num_iter:
            self.num_iter = len(self.dls.train)
        # if self.num_iter > len(self.dls.train): self.num_iter = len(self.dls.train)

        # Initialize the proper learning rate policy
        if self.step_mode.lower() == "exp":
            self.scheduler = ExponentialLR(self.opt, self.end_lr, self.num_iter)
        elif self.step_mode.lower() == "linear":
            self.scheduler = LinearLR(self.opt, self.end_lr, self.num_iter)
        else:
            raise ValueError("Unsupported `step_mode`.")

        self.model.train()

    def train_batch(self, batch: torch.Tensor):
        # forward + get loss + backward + optimize
        pred, self.loss = self.train_step(batch)
        # zero the parameter gradients
        self.opt.zero_grad()
        # gradient
        self.loss.backward()
        # update weights
        self.opt.step()

    def process_data(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = x.to(self.device), y.to(self.device)
        return x, y

    def train_step(self, batch: Union[Dict[str, Any], torch.Tensor]) -> Tuple[torch.Tensor, float]:
        # get the inputs

        if isinstance(batch, dict):
            self.xb, self.yb = batch["past_values"], batch["future_values"]
            signature = inspect.signature(self.model.forward)
            signature_args = list(signature.parameters.keys())

            args = {k: batch[k].to(self.device) for k in signature_args if k in batch}
            pred_outputs = self.model(**args)
            pred, loss = pred_outputs.prediction_outputs, pred_outputs.loss

        else:
            self.xb, self.yb = batch[0], batch[1]
            self.xb, self.yb = self.process_data(self.xb, self.yb)
            pred = self.model(self.xb)
            loss = self.loss_func(self.yb, pred)

        return pred, loss

    def after_batch_train(self):
        self.train_iter += 1
        self.scheduler.step()
        self.lrs.append(self.scheduler.get_last_lr()[0])

        # update smooth loss
        self.smoothing(self.beta)
        if self.smoothed_loss < self.best_loss:
            self.best_loss = self.smoothed_loss
        # Stop if the loss is exploding
        if self.smoothed_loss > 4 * self.best_loss:
            raise KeyboardInterrupt  # stop fit method
        if self.train_iter > self.num_iter:
            raise KeyboardInterrupt  # stop fit method

    def smoothing(self, beta: float):
        # Smooth the loss if beta is specified
        self.aver_loss = beta * self.aver_loss + (1 - beta) * self.loss.detach().item()
        self.smoothed_loss = self.aver_loss / (1 - beta**self.train_iter)
        self.losses.append(self.smoothed_loss)

    def fit(self, n_epochs, train_dataloder):
        try:
            for epoch in range(n_epochs):
                for batch in train_dataloder:
                    self.train_batch(batch)
                    self.after_batch_train()
        except KeyboardInterrupt:
            pass

    def after_fit(self):
        # reset the gradients
        self.opt.zero_grad()
        if self.suggestion == "valley":
            self.suggested_lr = valley(self.lrs, self.losses)
        else:
            raise ValueError(f"Unsupported lr suggestion mechanism '{self.suggestion}'.")
        # recorder the suggested learning rate
        self.recorder["suggested_lr"] = self.suggested_lr
        # load back the model at the previous state
        self.load(self.temp_path)
        os.remove(self.temp_path)

    def set_lr(self, lrs: List[float]):
        if not isinstance(lrs, list):
            lrs = [lrs] * len(self.opt.param_groups)
        if len(lrs) != len(self.opt.param_groups):
            raise ValueError(
                "Length of `lrs` is not equal to the number of parameter groups " + "in the given optimizer"
            )
        # update lr
        for param_group, lr in zip(self.opt.param_groups, lrs):
            param_group["lr"] = lr

    def plot_lr_find(self, plot_save_dir: Optional[str] = None):
        # Third Party
        import matplotlib.pyplot as plt

        if plot_save_dir is not None:
            os.makedirs(plot_save_dir, exist_ok=True)
            save_path = os.path.join(plot_save_dir, "lr_finder.png")
        else:
            save_path = "lr_finder.png"

        fig, ax = plt.subplots(1, 1)
        ax.plot(self.lrs, self.losses)
        ax.set_ylabel("Loss")
        ax.set_xlabel("Learning Rate")
        ax.set_xscale("log")
        plt.grid()
        plt.savefig(save_path)
        plt.close()


def optimal_lr_finder(
    model: torch.nn.Module,
    dset_train: torch.utils.data.Dataset,
    optimizer: Optional[Any] = None,
    device: Optional[str] = "cuda",
    start_lr: float = 1e-7,
    end_lr: float = 10.0,
    num_iter: int = 100,
    batch_size: int = 20,
    shuffle: bool = True,
    step_mode: str = "exp",
    suggestion: str = "valley",
    plot_lr_finder: bool = False,
    plot_save_dir: Optional[str] = None,
    enable_prefix_tuning: bool = False,
):
    """Finds the optimal learning rate that can used to train a model.
    This function returns the optimal learning rate and the original model.
    The returned model should be used for subsequent training, since the original models'
    weights will be changed during this procedure. Note that sometimes the return learning
    rate can be very low. In that case, it is suggested to manually set the learning rate
    instead of using this function.
    Example usage.
    ```python
    >>> from tsfm.utils.lr_finder import optimal_lr_finder
    >>> LR, model = optimal_lr_finder(model, dset_train, batch_size=20)
    >>> print("OPTIMAL SUGGESTED LEARNING RATE =", LR)
    ```

    Args:
        model (`nn.Module`): The pytorch model.
        dset_train (`torch.utils.data.Dataset`): The train dataset.
        optimizer (`torch.optim.optimizer.Optimizer`, *optional*, Defaults to None.): Optimizer.
        device (`str`, *optional*, Defaults to "cuda".): Device to use ("cpu"/"cuda").
        start_lr (`float`, *optional*, Defaults to 1e-7.): Start learning rate in search space.
        end_lr (`float`, *optional*, Defaults to 10.): End learning rate in search space.
        num_iter (`int`, *optional*, Defaults to 100.): Number of batch updates.
        batch_size (`int`, *optional*, Defaults to 20.): Batch size for the dataloader.
        shuffle (`bool`, *optional*, Defaults to False.): Whether to shuffle the dataloder.
        step_mode (`str`, *optional*, Defaults to "exp".): Type of learning rate scheduler ("exp"/"linear").
        suggestion (`str`, *optional*, Defaults to "valley".): Learning rate suggestion method (only "valley" is suggested currently).
        plot_lr_finder (`bool`, *optional*, Defaults to False.): Plot the search results.
        plot_save_dir (`str`, *optional*, Defaults to `None`.): Plot the search results.

    Returns:
        `float`: The optimal learning rate.
        `nn.Module`: The original model. This returned model should be used for subsequent training.
    """

    logger.info(
        "LR Finder: Running learning rate (LR) finder algorithm. If the suggested LR is very low, we suggest setting the LR manually."
    )

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        logger.info(f"LR Finder: Using GPU:{device}.")
    else:
        logger.info("LR Finder: Using CPU.")
        device = torch.device("cpu")

    # create the right collator in the style of HF
    signature = inspect.signature(model.forward)
    signature_columns = list(signature.parameters.keys())
    data_collator = default_data_collator

    remove_columns_collator = RemoveColumnsCollator(
        data_collator=data_collator,
        signature_columns=signature_columns,
        logger=None,
        description=None,
        model_name=model.__class__.__name__,
    )

    dl_train = DataLoader(
        dataset=dset_train, batch_size=batch_size, shuffle=shuffle, collate_fn=remove_columns_collator
    )

    n_epochs = num_iter // len(dl_train) + 1
    if optimizer is None:
        optimizer = AdamW(model.parameters(), 1e-4)
    lr_finder = LRFinder(
        model,
        optimizer,
        device,
        start_lr,
        end_lr,
        num_iter,
        step_mode,
        suggestion=suggestion,
        enable_prefix_tuning=enable_prefix_tuning,
    )
    lr_finder.before_fit()
    lr_finder.fit(n_epochs=n_epochs, train_dataloder=dl_train)
    lr_finder.after_fit()

    if plot_lr_finder:
        os.makedirs(plot_save_dir, exist_ok=True)
        lr_finder.plot_lr_find(plot_save_dir=plot_save_dir)

    logger.info(f"LR Finder: Suggested learning rate = {lr_finder.suggested_lr}")

    return lr_finder.suggested_lr, lr_finder.model
