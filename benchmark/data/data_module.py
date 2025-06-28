import typing as T
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from pathlib import Path
from torch.utils.data.dataset import Subset
from torch.utils.data.dataloader import DataLoader
from benchmark.data.datasets import BenchmarkDataset


class BenchmarkDataModule(pl.LightningDataModule):
    """
    Data module for any MGF-based benchmark.
    Splits according to the 'fold' column in metadata.
    """

    def __init__(
        self,
        dataset: BenchmarkDataset,
        batch_size: int,
        num_workers: int = 0,
        persistent_workers: bool = True,
        split_pth: T.Optional[Path] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dataset = dataset
        self.split_pth = split_pth
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers if num_workers > 0 else False

    def prepare_data(self):
        # No download required; data should already be in place.
        pass

    def setup(self, stage=None):
        # Determine splits
        if self.split_pth is None:
            split = self.dataset.metadata[["identifier", "fold"]]
        else:
            split = pd.read_csv(self.split_pth, sep="\t")
            if set(split.columns) != {"identifier", "fold"}:
                raise ValueError('Split file must contain "identifier" and "fold" columns.')
            split["identifier"] = split["identifier"].astype(str)
            if set(self.dataset.metadata["identifier"]) != set(split["identifier"]):
                raise ValueError("Dataset IDs must match split file IDs.")

        split = split.set_index("identifier")["fold"]
        if not set(split) <= {"train", "val", "test"}:
            raise ValueError('Fold values must be one of "train", "val", "test".')

        mask = split.loc[self.dataset.metadata["identifier"]].values
        if stage in ("fit", None):
            train_idx = np.where(mask == "train")[0]
            val_idx = np.where(mask == "val")[0]
            self.train_dataset = Subset(self.dataset, train_idx)
            self.val_dataset = Subset(self.dataset, val_idx)
        if stage in ("test", None):
            test_idx = np.where(mask == "test")[0]
            self.test_dataset = Subset(self.dataset, test_idx)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            drop_last=False,
            collate_fn=self.dataset.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            drop_last=False,
            collate_fn=self.dataset.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            drop_last=False,
            collate_fn=self.dataset.collate_fn,
        )