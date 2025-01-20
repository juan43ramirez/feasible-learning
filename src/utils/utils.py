import random
from types import SimpleNamespace
from typing import Any, Callable, Iterator

import numpy as np
import torch


def scan_namespace(namespace: SimpleNamespace) -> Iterator[tuple[str, Any]]:
    """Iterates over the attributes of a SimpleNamespace object and yields
    a tuple with the attribute name and value."""
    for key in vars(namespace):
        yield key, getattr(namespace, key)


def extract_to_namespace(struct: dict | SimpleNamespace, extract_fn: Callable):
    """Extracts values from a dictionary or SimpleNamespace object and returns
    a new SimpleNamespace object with the extracted values."""
    if isinstance(struct, dict):
        return SimpleNamespace(**{key: extract_fn(value) for (key, value) in struct.items()})
    elif isinstance(struct, SimpleNamespace):
        return SimpleNamespace(**{key: extract_fn(getattr(struct, key)) for key in vars(struct)})
    else:
        raise ValueError(f"Unsupported type {type(struct)}")


def set_seed(seed: int):
    """Sets the seed for the random number generators used by random, numpy and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RNGContext:
    """Context manager for deterministic random number generation. This context returns
    to the previous RNG states after its closure."""

    def __init__(self, seed):
        """Initializes the RNGContext object with a seed value."""
        self.seed = seed
        self.initial_torch_seed = torch.initial_seed()
        self.initial_numpy_seed = np.random.get_state()
        self.initial_random_seed = random.getstate()

        assert self.seed is not None, "Seed must be provided to the RNGContext."
        assert isinstance(self.initial_torch_seed, int), "Torch seed must be an integer."
        assert isinstance(self.initial_numpy_seed, tuple), "Numpy seed must be a tuple."
        assert isinstance(self.initial_random_seed, tuple), "Random seed must be a tuple."

    def __enter__(self):
        """Sets the seed for the random number generators used by random, numpy and
        torch when entering the context."""
        set_seed(self.seed)

    def __exit__(self, exc_type, exc_value, traceback):
        """Resets the RNG states to their previous values when exiting the context."""
        torch.manual_seed(self.initial_torch_seed)
        torch.cuda.manual_seed_all(self.initial_torch_seed)
        torch.random.manual_seed(self.initial_torch_seed)
        np.random.set_state(self.initial_numpy_seed)
        random.setstate(self.initial_random_seed)
