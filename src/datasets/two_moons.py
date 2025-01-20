import torch
from sklearn.datasets import make_moons


class TwoMoonsDataset(torch.utils.data.Dataset):
    name = "two_moons"
    input_shape = (2,)
    output_size = 2

    def __init__(self, num_samples, noise, random_state=None):
        X, y = make_moons(n_samples=num_samples, noise=noise, random_state=random_state)
        self.data = torch.tensor(X).to(torch.float32)
        self.targets = y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]
