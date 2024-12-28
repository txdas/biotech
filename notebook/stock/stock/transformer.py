import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import  KBinsDiscretizer

class AmountVocab(object):
    def __init__(self,values):

        super().__init__()
        self.kbins  = KBinsDiscretizer(n_bins=10,strategy='quantile',subsample=10000)
        self.kbins.fit(values.reshape(-1,1))
        self.bin_edges = self.kbins.bin_edges_[0]

    def __call__(self,v):
        for idx, e in enumerate(self.bin_edges):
            if v<e:
                return  idx
        return len(self.bin_edges)


class StdNormalize(object):
    def __init__(self,gap=20):
        super().__init__()
        self.gap = gap

    def __call__(self,v):
        if v<-self.gap:
            return -1
        elif v>self.gap:
            return 1
        else:
            return v/self.gap