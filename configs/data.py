import os
from functools import partial

import ml_collections as mlc

import shared
from src import datasets

MLC_PH = mlc.config_dict.config_dict.placeholder


def get_data_path():
    return os.environ.get("DATA_DIR", "~/datasets")


def basic_config(splits: list[str] = ["train", "val"]):
    config = mlc.ConfigDict()

    config.dataset_class = MLC_PH(type)

    config.dataloader_kwargs = mlc.ConfigDict()

    config.dataloader_kwargs.split_kwargs = mlc.ConfigDict()
    config.dataloader_kwargs.use_distributed_sampler = MLC_PH(bool)
    config.dataloader_kwargs.seed = MLC_PH(int)
    config.dataloader_kwargs.use_prefetcher = MLC_PH(bool)
    config.dataloader_kwargs.num_workers = MLC_PH(int)

    config.dataset_kwargs = mlc.ConfigDict()
    config.dataset_kwargs.split_kwargs = mlc.ConfigDict()

    # Automatically create fields (unpopulated) for each of the splits
    for split_name in splits:
        setattr(config.dataset_kwargs.split_kwargs, split_name, mlc.ConfigDict())
        setattr(config.dataloader_kwargs.split_kwargs, split_name, mlc.ConfigDict())
        getattr(config.dataloader_kwargs.split_kwargs, split_name).batch_size_per_gpu = MLC_PH(int)

    return config


def cifar_config(cifar_type):
    config = basic_config(splits=["train", "val"])

    config.dataset_type = cifar_type
    config.dataset_class = datasets.CIFAR10 if cifar_type == "cifar10" else datasets.CIFAR100

    config.dataloader_kwargs.use_distributed_sampler = False
    config.dataloader_kwargs.seed = 0
    config.dataloader_kwargs.use_prefetcher = False
    # config.dataloader_kwargs.num_workers = 4

    config.dataloader_kwargs.split_kwargs.train.batch_size_per_gpu = 128
    config.dataloader_kwargs.split_kwargs.val.batch_size_per_gpu = 128

    config.dataset_kwargs.split_kwargs.train.split = "train"
    config.dataset_kwargs.split_kwargs.train.data_path = get_data_path()

    config.dataset_kwargs.split_kwargs.val.split = "val"
    config.dataset_kwargs.split_kwargs.val.data_path = get_data_path()

    return config


def two_moons_config():
    config = basic_config(splits=["train", "val"])

    config.dataset_class = datasets.TwoMoonsDataset

    config.dataloader_kwargs.seed = 0
    config.dataloader_kwargs.num_workers = 0  # Tuned for local MBP run
    config.dataloader_kwargs.split_kwargs.train.batch_size_per_gpu = 512
    config.dataloader_kwargs.split_kwargs.val.batch_size_per_gpu = 128

    config.dataset_kwargs.split_kwargs.train.num_samples = 512
    config.dataset_kwargs.split_kwargs.train.noise = 0.1
    config.dataset_kwargs.split_kwargs.val.num_samples = 128
    config.dataset_kwargs.split_kwargs.val.noise = 0.1

    return config


def utkface_config():
    config = basic_config(splits=["train", "val"])

    config.dataset_class = datasets.UTKFace

    config.dataloader_kwargs.use_distributed_sampler = False
    config.dataloader_kwargs.seed = 0
    config.dataloader_kwargs.use_prefetcher = False
    # config.dataloader_kwargs.num_workers = 4

    config.dataloader_kwargs.split_kwargs.train.batch_size_per_gpu = 128
    config.dataloader_kwargs.split_kwargs.val.batch_size_per_gpu = 128

    config.dataset_kwargs.split_kwargs.train.split = "train"
    config.dataset_kwargs.split_kwargs.train.data_path = get_data_path()

    config.dataset_kwargs.split_kwargs.val.split = "val"
    config.dataset_kwargs.split_kwargs.val.data_path = get_data_path()

    return config


DATA_CONFIGS = {
    None: basic_config,
    "two_moons": two_moons_config,
    "cifar10": partial(cifar_config, cifar_type="cifar10"),
    "utkface": utkface_config,
}


def get_config(config_string=None):
    return shared.default_get_config(
        config_group_name="data", pop_key="dataset_type", preset_configs=DATA_CONFIGS, cli_cmd=config_string
    )
