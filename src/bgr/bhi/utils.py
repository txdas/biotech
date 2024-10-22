import pandas as pd
import numpy as np
from tqdm import tqdm
from torch import Generator
from pathlib import Path
import random
import math

import torch
from torch import nn
import torch.nn.functional as F



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



def BHI_preprocess_data(data: pd.DataFrame,
                    seqsize: int,
                    plasmid_path: str | Path):
    """
    230407 BHI edited
    - Short training sequences are padded on the 3-end, 5-end, and both with nucleotides from the vector sequence to the uniform total length. (110bp)
        - 3:4:2 ratio
    - Long training sequences are trimmed from right
    """

    vector_left = 'GGTGCCTGAAACTAG'
    vector_right = 'ATG'
    seq_idx = data.columns.get_loc('seq')
    for i in tqdm(range(0, len(data))):
        # Trim from right (BHI ver.)
        if len(data.iloc[i,seq_idx]) > seqsize :
            data.iloc[i,seq_idx] = data.iloc[i,seq_idx][:seqsize]

        #### lr+sym random shift padding (BHI ver.) ####
        # 3:4:2 ratio for right, left, both(symmetric) padding
        elif len(data.iloc[i,seq_idx]) < seqsize :
            # pad right
            if random.random() < 0.333:
                data.iloc[i,seq_idx] = data.iloc[i,seq_idx] + vector_right[:110-len(data.iloc[i,seq_idx])]
            # pad left
            elif random.random() > 0.666:
                data.iloc[i,seq_idx] = vector_left[-(110-len(data.iloc[i,seq_idx])):] + data.iloc[i,seq_idx]
            # pad left right both, symmetrically
            else:
                data_len = len(data.iloc[i,seq_idx])
                pad_len = seqsize - data_len
                left_pad = pad_len//2
                right_pad = pad_len - left_pad    # right_pad >= left_pad

                # pad right
                if right_pad > 0:
                    data.iloc[i,seq_idx] = data.iloc[i,seq_idx] + vector_right[:right_pad]
                # pad left
                if left_pad > 0:
                    data.iloc[i,seq_idx] = vector_left[-left_pad:] + data.iloc[i,seq_idx]

    return data


def BHI_preprocess_df(path: str | Path,
                  seqsize: int,
                  plasmid_path: str |  Path):
    df = pd.read_csv(path)[["seq", "score"]]
    df = BHI_preprocess_data(df,
                         seqsize=seqsize,
                         plasmid_path=plasmid_path)
    return df


def initialize_weights(m: nn.Module, generator: Generator):
    if isinstance(m, nn.Conv1d):
        n = m.kernel_size[0] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2 / n), generator=generator)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.001, generator=generator)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)