import socket
from types import SimpleNamespace

import torch

import shared

logger = shared.fetch_main_logger()


def init_distributed():
    """
    Extracts relevant info from SLURM environment variables.
    Sets device based on local_rank.
    Defines {multi_gpu, rank, local_rank, world_size, device}
    """

    if torch.cuda.is_available():
        logger.info("This is a single GPU job")
    else:
        logger.info("This is a CPU job")
    multi_gpu = False
    world_size = 1
    rank = 0
    local_rank = 0

    logger.info(f"Rank {rank}")
    logger.info(f"World size {world_size}")
    logger.info(f"Local rank {local_rank}")
    logger.info(f"Running on host {socket.gethostname()}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    return SimpleNamespace(
        multi_gpu=multi_gpu,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
    )
