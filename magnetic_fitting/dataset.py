"""Dataset utilities for magnetic parameter fitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split


class MagneticDataset(Dataset):
    """Torch dataset that exposes the CSV contents as tensors."""

    def __init__(self, csv_path: Path) -> None:
        df = pd.read_csv(csv_path)

        required_cols = [
            "Sqx",
            "Sqy",
            "Sqz",
            "qx",
            "qy",
            "qz",
            "nx",
            "ny",
            "nz",
            "E",
        ]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in {csv_path}: {missing}")

        self.sp = torch.tensor(df[["Sqx", "Sqy", "Sqz"]].values, dtype=torch.double)
        self.q = torch.tensor(df[["qx", "qy", "qz"]].values, dtype=torch.double)
        self.n = torch.tensor(df[["nx", "ny", "nz"]].values, dtype=torch.double)
        self.energy = torch.tensor(df["E"].values, dtype=torch.double)

    def __len__(self) -> int:  # pragma: no cover - simple delegation
        return self.energy.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.sp[idx], self.q[idx], self.n[idx], self.energy[idx]


@dataclass
class DataConfig:
    batch_size: int
    val_split: float
    seed: int


def create_dataloaders(
    dataset: MagneticDataset, config: DataConfig
) -> Tuple[DataLoader, DataLoader | None]:
    """Split ``dataset`` into train/validation loaders according to ``config``."""

    val_split = max(0.0, min(float(config.val_split), 0.5))
    num_val = int(len(dataset) * val_split)

    if num_val > 0:
        generator = torch.Generator().manual_seed(config.seed)
        train_set, val_set = random_split(
            dataset, [len(dataset) - num_val, num_val], generator=generator
        )
        val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False)
    else:
        train_set = dataset
        val_loader = None

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    return train_loader, val_loader
