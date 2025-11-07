"""Utilities for fitting magnetic interaction parameters."""

from .dataset import MagneticDataset, DataConfig, create_dataloaders
from .model import MagneticEnergyModel
from .train import TrainingConfig, TrainingResult, train, evaluate

__all__ = [
    "MagneticDataset",
    "DataConfig",
    "create_dataloaders",
    "MagneticEnergyModel",
    "TrainingConfig",
    "TrainingResult",
    "train",
    "evaluate",
]
