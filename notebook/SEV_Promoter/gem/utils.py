import torch
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn import preprocessing


def r2(x,y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    return r_value**2

def set_seed(random_seed=41):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)


def one_hot_encode(df, col='seq', seq_len=44):
    # Dictionary returning one-hot encoding of nucleotides.
    nuc_d = {'a': [1, 0, 0, 0], 'c': [0, 1, 0, 0], 'g': [0, 0, 1, 0], 't': [0, 0, 0, 1], 'n': [0, 0, 0, 0]}

    # Creat empty matrix.
    vectors = np.empty([len(df), seq_len, 4])

    # Iterate through UTRs and one-hot encode
    for i, seq in enumerate(df[col].str[:seq_len]):
        seq = seq.lower()
        lst = [nuc_d[x] for x in seq]
        if seq_len > len(seq):
            lst += [nuc_d['n']] * (seq_len - len(seq))
        a = np.array(lst)
        vectors[i] = a
    return vectors


class EarlyStopping:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.best_model = None

    def early_stop(self,model, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            # self.best_model =model.copy()
            self.best_model = model
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False