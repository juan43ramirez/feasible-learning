from types import SimpleNamespace

import pydash

import shared
from src.utils import scan_namespace

from .prefetch_loader import PrefetchLoader
from .two_moons import TwoMoonsDataset
from .utils import IndexedDataset, build_dataloader
from .vision import CIFAR10, UTKFace

logger = shared.fetch_main_logger()


def build_datasets(config) -> tuple[SimpleNamespace, SimpleNamespace]:
    """
    Builds and returns datasets and metadata for a given configuration.

    Notes:
        - The final datasets are wrapped in IndexedDataset.
        - The metadata is a SimpleNamespace object containing the following attributes:
            - corrupted_train_idxs: A list of indices of the corrupted training samples.
        - Transformation like label noise and imbalance are applied at the dataset level
            and are performed in a deterministic manner based on prescribed seeds.
        - These transformations are currently only supported for classification tasks.
        - Dataset imbalance is applied _after_ label noise since it looks at the sample
            labels to sub-sample the dataset.
    """

    dataset_namespace = SimpleNamespace()
    for split in config.data.dataset_kwargs.split_kwargs:
        split_kwargs = pydash.omit(config.data.dataset_kwargs, "split_kwargs")
        split_kwargs.update(getattr(config.data.dataset_kwargs.split_kwargs, split, {}))
        _dataset = config.data.dataset_class(**split_kwargs)
        if hasattr(_dataset, "data"):
            _dataset.data = _dataset.data.to(dtype=config.task.dtype)
        setattr(dataset_namespace, split, _dataset)

    num_classes = config.data.dataset_class.output_size

    # dataset_name = config.data.dataset_type
    # data_path = config.data.dataset_kwargs.split_kwargs.train.data_path
    # use_data_augmentation = config.data.dataset_kwargs.use_data_augmentation
    # dataset_namespace, num_classes = KNOWN_DATASET_LOADERS[dataset_name](data_path, use_data_augmentation)

    dataset_metadata = SimpleNamespace(
        num_classes=num_classes,
        is_2d_dataset=isinstance(dataset_namespace.train, TwoMoonsDataset),
    )

    # Wrap all datasets in an IndexedDataset, which allows us keep track of the sample
    # indices for the multipliers.
    indexed_dataset_namespace = SimpleNamespace()
    for split, dataset in scan_namespace(dataset_namespace):
        setattr(indexed_dataset_namespace, split, IndexedDataset(dataset))

    return indexed_dataset_namespace, dataset_metadata
