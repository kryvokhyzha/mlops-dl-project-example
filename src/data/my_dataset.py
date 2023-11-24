from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class MyDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, target_col: str, feature_cols: List[str], num_classes: int) -> None:
        self.df = dataframe.fillna(0.0).reset_index(drop=True)
        self.target_col = target_col
        self.feature_cols = feature_cols
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.df.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.FloatTensor(self.df.iloc[idx][self.feature_cols].values, device="cpu")

        label = self.df.iloc[idx][self.target_col]
        if self.num_classes > 1:
            target = np.zeros(shape=(self.num_classes,))
            target[int(label)] = 1.0
            y = torch.FloatTensor(target, device="cpu")
        else:
            y = torch.FloatTensor([label], device="cpu")

        return x, y


class MyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        read_csv_path: str,
        read_csv_kwargs: Dict[str, Any],
        val_size: float,
        batch_size: int,
        dataloader_num_workers: int,
        dataloader_pin_memory: bool,
        dataloader_persistent_workers: bool,
        target_col: str,
        feature_cols: Optional[List[str]],
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.read_csv_path = read_csv_path
        self.read_csv_kwargs = read_csv_kwargs
        self.val_size = val_size
        self.dataloader_num_workers = dataloader_num_workers
        self.dataloader_pin_memory = dataloader_pin_memory
        self.dataloader_persistent_workers = dataloader_persistent_workers
        self.batch_size = batch_size
        self.target_col = target_col
        self.feature_cols = feature_cols

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    def prepare_data(self):
        """Use this to download and prepare data.

        Downloading and saving data with
        multiple processes (distributed settings) will result in corrupted data.
        Lightning ensures this method is called only within a single process, so you can
        safely add your downloading logic within.
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Use this to setup datasets and additional parameters.

        Called at the beginning of fit (train + validate), validate, test, or predict.

        This is a good hook when you need to build models dynamically or adjust something
        about them. This hook is called on every process when using DDP.

        setup is called from every process across all the nodes. Setting state here is
        recommended.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.batch_size}) is not divisible by the number of devices"
                    f" ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.batch_size // self.trainer.world_size

        df = pd.read_csv(self.read_csv_path, **self.read_csv_kwargs)
        self.feature_cols = (
            [col for col in df.columns if col != self.target_col] if self.feature_cols is None else self.feature_cols
        )
        train_df, val_df = train_test_split(df, test_size=self.val_size, random_state=42)

        num_classes = df[self.target_col].nunique()

        self.train_dataset = MyDataset(
            dataframe=train_df,
            target_col=self.target_col,
            feature_cols=self.feature_cols,
            num_classes=num_classes,
        )
        self.val_dataset = MyDataset(
            dataframe=val_df,
            target_col=self.target_col,
            feature_cols=self.feature_cols,
            num_classes=num_classes,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size_per_device,
            shuffle=True,
            num_workers=self.dataloader_num_workers,
            pin_memory=self.dataloader_pin_memory,
            persistent_workers=self.dataloader_persistent_workers,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            num_workers=self.dataloader_num_workers,
            pin_memory=self.dataloader_pin_memory,
            persistent_workers=self.dataloader_persistent_workers,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            num_workers=self.dataloader_num_workers,
            pin_memory=self.dataloader_pin_memory,
            persistent_workers=self.dataloader_persistent_workers,
        )

    def teardown(self, stage: str) -> None:
        """Teardown.

        Called at the end of fit (train + validate), validate, test, or predict.

        Args:
        ----
            stage: either ``'fit'``, ``'validate'``, ``'test'``, or ``'predict'``
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Save a checkpoint (called when saving a checkpoint.).

         Implement to generate and save the datamodule state.

        Returns
        -------
        Dict[Any, Any]
            A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Implement to reload datamodule state given the datamodule's `state_dict()` when loading a checkpoint.

        Parameters
        ----------
        state_dict: Dict[str, Any]
            The datamodule state returned by `self.state_dict()`.
        """
        pass
