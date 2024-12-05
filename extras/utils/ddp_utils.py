import os
from datetime import timedelta

import torch


def init_ddp(timeout=600):
    local_rank = int(os.environ.get("LOCAL_RANK"))
    world_size = int(os.environ.get("WORLD_SIZE"))
    rank = int(os.environ.get("RANK"))

    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        "nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
        timeout=timedelta(seconds=timeout),
    )


def is_rank_0():
    rank = torch.distributed.get_rank()
    return rank == 0
