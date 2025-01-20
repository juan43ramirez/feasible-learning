import os

import dotenv
import numpy as np
import torch
import wandb

dotenv.load_dotenv()


def make_wandb_histogram(values: torch.Tensor, bins=50):
    values = values.flatten().detach().cpu().numpy()
    hist = np.histogram(values, bins=bins)
    return wandb.Histogram(np_histogram=hist)


if __name__ == "__main__":

    """Test logging a wandb histogram."""

    run = wandb.init(project="test", dir=os.environ["WANDB_DIR"])

    for _step in range(10):
        values = np.random.randn(1000) + _step
        histogram = make_wandb_histogram(values)
        wandb.log({"histogram": histogram}, step=_step)
