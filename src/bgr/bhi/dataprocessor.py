from torch import Generator
from torch.utils.data import DataLoader
from pathlib import Path
from .utils import BHI_preprocess_df
from .dataset import BHISeqDatasetProb
from ..models import DataProcessor

"""
230406 BHIDataProcessor added
- Target standardization
- left, right, symmetric padding with vector sequence
- RC equivariant
"""


class BHIDataloaderWrapper:
    def __init__(self,
                 dataloader: DataLoader,
                 batch_per_epoch: int):
        self.batch_per_epoch = batch_per_epoch
        self.dataloader = dataloader
        self.iterator = iter(dataloader)

    def __len__(self):
        return self.batch_per_epoch

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)

    def __iter__(self):
        for _ in range(self.batch_per_epoch):
            try:
                yield next(self.iterator)
            except StopIteration:
                self.iterator = iter(self.dataloader)

class BHIDataProcessor(DataProcessor):
    def __init__(
            self,
            seqsize: int,
            path_to_training_data: str | Path,
            path_to_validation_data: str | Path,
            plasmid_path: str | Path,
            generator: Generator,
            train_batch_size: int = 1024,
            train_workers: int = 1,
            shuffle_train: bool = True,
            valid_batch_size: int = 4096,
            valid_workers: int = 1,
            shuffle_val: bool = False
    ):
        self.train = BHI_preprocess_df(path=path_to_training_data,
                                       seqsize=seqsize,
                                       plasmid_path=plasmid_path)

        self.valid = BHI_preprocess_df(path=path_to_validation_data,
                                       seqsize=seqsize,
                                       plasmid_path=plasmid_path)

        self.train_batch_size = train_batch_size
        self.batch_per_epoch = len(self.train) // train_batch_size + 1
        self.train_workers = train_workers
        self.shuffle_train = shuffle_train

        self.valid_batch_size = valid_batch_size
        self.valid_workers = valid_workers
        self.shuffle_val = shuffle_val
        self.batch_per_valid = len(self.valid) // valid_batch_size + 1

        self.seqsize = seqsize
        self.plasmid_path = plasmid_path
        self.generator = generator

    def prepare_train_dataloader(self):
        train_ds = BHISeqDatasetProb(
            self.train,
            seqsize=self.seqsize,
        )
        train_dl = DataLoader(
            train_ds,
            batch_size=self.train_batch_size,
            num_workers=self.train_workers,
            shuffle=self.shuffle_train,
            generator=self.generator
        )
        train_dl = BHIDataloaderWrapper(train_dl, self.batch_per_epoch)
        return train_dl

    def prepare_valid_dataloader(self):
        valid_ds = BHISeqDatasetProb(
            self.valid,
            seqsize=self.seqsize,
        )
        valid_dl = DataLoader(
            valid_ds,
            batch_size=self.valid_batch_size,
            num_workers=self.valid_workers,
            shuffle=self.shuffle_val
        )
        valid_dl = BHIDataloaderWrapper(valid_dl, self.batch_per_valid)
        return valid_dl

    def train_epoch_size(self) -> int:
        return self.batch_per_epoch

    """
    230407 BHI edited
    - 6 => 4 (only onehot encoded data)
    """

    def data_channels(self) -> int:
        return 4  # 4 - onehot, 1 - singleton, 1 - is_reverse

    def data_seqsize(self) -> int:
        return self.seqsize
