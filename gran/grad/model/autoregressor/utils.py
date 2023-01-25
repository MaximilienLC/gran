import numpy as np
import torch


class ARDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str):

        self.X, self.rew, self.done = torch.tensor(data, dtype=torch.float)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx]
