import torch
from torch import nn
import torch.nn.functional  as F
import numpy as np


CODES: dict[str, int] = {
    "A": 0,
    "T": 3,
    "G": 1,
    "C": 2,
    'N': 4
}


def n2id(n: str) -> int:
    return CODES[n.upper()]


class BHI_Seq2Tensor(nn.Module):
    '''
    Encode sequences using one-hot encoding after preprocessing.
    '''
    def __init__(self):
        super().__init__()

    def forward(self, seq: str) -> torch.Tensor:
        seq_i = [n2id(x) for x in seq]
        code = torch.from_numpy(np.array(seq_i))
        code = F.one_hot(code, num_classes=5) # 5th class is N

        """
        230407 BHI edited
        one-hot encode N with 0 0 0 0
        """
        code[code[:, 4] == 1] = 0 # 0.25 => 0 # encode Ns with .25
        code = code[:, :4].float()
        return code.transpose(0, 1)