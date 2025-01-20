import time
from abc import ABC
from functools import partial
from types import SimpleNamespace
from typing import Literal, Optional

import numpy as np
import scipy as sp
import torch
import torchvision as tv
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

import shared
from src.utils.utils import RNGContext

from .prefetch_loader import PrefetchLoader

logger = shared.fetch_main_logger(apply_basic_config=True)


class BaseDataset(Dataset, ABC):
    name: str
    train_transforms: Optional[tv.transforms.Compose] = None
    val_transforms: Optional[tv.transforms.Compose] = None
    input_shape: tuple
    output_size: int


class IndexedDataset(Dataset):
    """Wraps a generic Pytorch dataset and appends the index of the sample as the last
    return value in the `__getitem__` method.
    """

    def __init__(self, dataset: BaseDataset):
        self.dataset = dataset

    @property
    def name(self):
        return self.dataset.name

    @property
    def input_shape(self):
        return self.dataset.input_shape

    @property
    def output_size(self):
        return self.dataset.output_size

    @property
    def data(self):
        if isinstance(self.dataset, torch.utils.data.Subset):
            return self.dataset.dataset.data[self.dataset.indices]
        else:
            return self.dataset.data

    @property
    def targets(self):
        if isinstance(self.dataset, torch.utils.data.Subset):
            return self.dataset.dataset.targets[self.dataset.indices]
        else:
            return self.dataset.targets

    def __getitem__(self, idx):
        standard_return = self.dataset.__getitem__(idx)
        if isinstance(standard_return, tuple):
            return (*standard_return, idx)
        else:
            return (standard_return, idx)

    def __len__(self):
        return len(self.dataset)


def find_best_num_workers(dataloader_class):
    logger.info("Finding best num_workers for dataloader")

    num_workers_to_test = list(range(0, 10, 2))
    num_workers_time = {}
    for num_workers_ix, num_workers in enumerate(num_workers_to_test):
        dataloader = dataloader_class(num_workers=num_workers)
        start = time.time()
        for epoch in range(2):
            for batch_ix, data in enumerate(dataloader, 0):
                if batch_ix > 5:
                    break
        current_time = time.time() - start
        num_workers_time[num_workers] = current_time
        logger.info(f"num_workers: {num_workers}, time: {current_time}")

        if num_workers_ix > 0:
            previous_time = num_workers_time[num_workers_to_test[num_workers_ix - 1]]
            if current_time > previous_time:
                logger.info("Latest num_workers choice caused a time increase, stopping search")
                break

    # Return key with the best time
    return min(num_workers_time, key=num_workers_time.get)


def build_dataloader(
    dataset: str,
    split: str,
    batch_size_per_gpu: int,
    num_workers: int = 0,
    use_distributed_sampler: bool = False,
    seed: int = 0,
    device: torch.device = torch.device("cpu"),
    use_prefetcher: bool = False,
):
    """
    Args:
        batch_size_per_gpu: batch size per GPU. If -1, use the full dataset as a batch.
            The value -1 is only intended to be used in single-GPU jobs.
    """

    batch_size_per_gpu = len(dataset) if batch_size_per_gpu == -1 else batch_size_per_gpu

    is_training_split = split == "train"

    if use_distributed_sampler:
        logger.info(f"Using DistributedSampler for {split} split")
        sampler = DistributedSampler(dataset, shuffle=is_training_split, seed=seed)
    elif is_training_split:
        generator = torch.Generator().manual_seed(seed)
        sampler = torch.utils.data.RandomSampler(dataset, replacement=False, generator=generator)
    else:
        logger.info(f"Using {split} dataloader with SequentialSampler")
        sampler = torch.utils.data.SequentialSampler(dataset)

    dataloader_class = partial(
        DataLoader, dataset=dataset, batch_size=batch_size_per_gpu, pin_memory=True, sampler=sampler
    )
    if num_workers is None:
        best_num_workers = find_best_num_workers(dataloader_class)
        logger.info(f"Using empirically found `num_workers={best_num_workers}` for {split} dataloader")
    else:
        best_num_workers = num_workers
        logger.info(f"Using provided `num_workers={best_num_workers}` for {split} dataloader")
    dataloader = dataloader_class(num_workers=best_num_workers)

    if use_prefetcher:
        if device.type == "cpu":
            raise ValueError("Using prefetcher with CPU runtime is not supported.")
        logger.info(f"Wrapping {split} dataloader in prefetcher")
        dataloader = PrefetchLoader(dataloader, device)

    logger.info(f"Dataloader for {split} split has {len(dataloader)} batches of size {batch_size_per_gpu}")

    return dataloader, best_num_workers
