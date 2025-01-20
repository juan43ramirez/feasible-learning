import abc

import torch.nn as nn


class BaseModel(nn.Module, abc.ABC):
    def __init__(self):
        super(BaseModel, self).__init__()

    @abc.abstractmethod
    def forward(self, x):
        raise NotImplementedError

    def parameter_groups(self):
        """Defines parameter groups to be used when constructing the optimizer."""
        return {"params": self.parameters()}
