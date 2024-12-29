from torch import utils
import torch

class NPDataset(utils.data.Dataset):

    def __init__(self, X, y):
        self.X = torch.tensor(X,dtype=torch.float)
        self.y = torch.tensor(y,dtype=torch.float).reshape((-1, 1))

    def __getitem__(self, item):
        return self.X[item], self.y[item]

    def __len__(self):
        return len(self.y)