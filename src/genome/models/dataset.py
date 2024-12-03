import torch
from torch.utils.data import Dataset


class NPDataset(Dataset):

    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float)
        self.y = None
        if y:
            self.y = torch.tensor(y, dtype=torch.float).reshape((-1, 1))

    def __getitem__(self, idx):
        if self.y:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]

    def __len__(self):
        return len(self.X)
