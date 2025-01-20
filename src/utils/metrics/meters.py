import abc
import inspect

import torch
import torch.distributed


class Meter(abc.ABC):
    """
    Meters are used to aggregate a collection of values and provide statistics about them.
    Some meters act in a streaming fashion, updating their statistics as new values are
    added. This is the case of `AverageMeter` and `BestMeter`. Others act in a batch
    fashion, collecting all values and then computing statistics, such as `StatsMeter`.

    # .. warning::
    #     The current implementation of different meters is not compatible with
    #     _within-epoch_ checkpointing. Since we skip over the computation of forwards
    #     when resuming at the granularity of steps, we would need to checkpoint the
    #     states of the meters, which is not currently being done.
    """

    def __init__(self):
        self.reset()
        self.known_returns = None

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def update(self, val, n=1):
        pass

    @abc.abstractmethod
    def _sync(self):
        pass

    @abc.abstractmethod
    def get_result_dict(self):
        pass

    @classmethod
    def args_for_update(cls):
        update_args = inspect.getfullargspec(cls.update).args
        update_args.remove("self")
        return update_args

    @abc.abstractmethod
    def state_dict(self):
        pass

    @abc.abstractmethod
    def load_state_dict(self, state_dict):
        pass


class AverageMeter(Meter):
    """Computes and stores the average and current value of a given metric. Supports
    distributed training.
    """

    def __init__(self):
        self.known_returns = ["avg"]
        self.reset()

    def reset(self):
        self.synced_count = 0
        self.new_count = 0

        self.synced_sum = 0
        self.new_sum = 0

    def update(self, val, n=1):
        self.new_sum += val * n
        self.new_count += n

    def _sync(self):
        new_counts, new_sums = self.new_count, self.new_sum

        self.synced_count += new_counts
        self.synced_sum += new_sums

        self.new_count = 0
        self.new_sum = 0

    @property
    def avg(self):
        self._sync()
        # No need to add new_count since call to avg() forces a call to _sync()
        if self.synced_count > 0:
            return self.synced_sum / self.synced_count
        else:
            return 0

    def get_result_dict(self):
        return {"avg": self.avg}

    def state_dict(self):
        return {
            "synced_count": self.synced_count,
            "new_count": self.new_count,
            "synced_sum": self.synced_sum,
            "new_sum": self.new_sum,
        }

    def load_state_dict(self, state_dict):
        self.synced_count = state_dict["synced_count"]
        self.new_count = state_dict["new_count"]
        self.synced_sum = state_dict["synced_sum"]
        self.new_sum = state_dict["new_sum"]
        return self


class StatsMeter(Meter):
    """Stores all values in order to compute statistics of a given metric."""

    STATS_MAP = {
        "min": torch.min,
        "median": torch.median,
        "avg": torch.mean,
        "max": torch.max,
        "std": torch.std,
        "nonpos_rate": lambda x: (x <= 0).float().mean(),
        "pos_avg": lambda x: x[x > 0].mean(),
        "pointwise": lambda x: x,
    }

    def __init__(self, stats=["min", "median", "max", "std", "pointwise"]):
        self.known_returns = stats
        self.reset()

    def reset(self):
        self.all_elements = []

    def update(self, val):
        self.all_elements += [val.detach()]

    def _sync(self):
        return torch.cat(self.all_elements)

    def get_result_dict(self):
        cat_all_elements = self._sync()
        return {stat: self.STATS_MAP[stat](cat_all_elements) for stat in self.known_returns}

    def state_dict(self):
        return {"all_elements": self.all_elements}

    def load_state_dict(self, state_dict):
        self.all_elements = state_dict["all_elements"]
        return self
