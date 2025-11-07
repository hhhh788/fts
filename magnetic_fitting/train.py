"""Training utilities for fitting magnetic interaction parameters."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import DataConfig, MagneticDataset, create_dataloaders
from .model import MagneticEnergyModel


@dataclass
class TrainingConfig:
    data_path: Path
    epochs: int = 2000
    batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    val_split: float = 0.1
    output: Path | None = None
    history_output: Path | None = None
    metrics_output: Path | None = None
    device: str = "cpu"
    seed: int = 42


@dataclass
class TrainingResult:
    parameters: Dict[str, float]
    history: List[Dict[str, float | int | None]]
    metrics: Dict[str, float]


def evaluate(model: MagneticEnergyModel, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    mae_total = 0.0
    mse_total = 0.0
    total = 0

    with torch.no_grad():
        for sp, q, n, target in loader:
            sp = sp.to(device)
            q = q.to(device)
            n = n.to(device)
            target = target.to(device)

            diff = model(sp, q, n) - target
            mae_total += torch.sum(torch.abs(diff)).item()
            mse_total += torch.sum(diff ** 2).item()
            total += target.shape[0]

    if total == 0:
        return float("nan"), float("nan")

    return mae_total / total, mse_total / total


def train(config: TrainingConfig) -> TrainingResult:
    torch.manual_seed(config.seed)

    dataset = MagneticDataset(config.data_path)
    data_cfg = DataConfig(
        batch_size=config.batch_size,
        val_split=config.val_split,
        seed=config.seed,
    )
    train_loader, val_loader = create_dataloaders(dataset, data_cfg)

    device = torch.device(config.device)

    model = MagneticEnergyModel().to(device).double()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = nn.MSELoss()

    history: List[Dict[str, float | int | None]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        total = 0
        for sp, q, n, target in train_loader:
            sp = sp.to(device)
            q = q.to(device)
            n = n.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            pred = model(sp, q, n)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

            batch_size = target.shape[0]
            epoch_loss += loss.item() * batch_size
            total += batch_size

        train_mse = epoch_loss / total
        val_mse = None
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                val_total = 0
                for sp, q, n, target in val_loader:
                    sp = sp.to(device)
                    q = q.to(device)
                    n = n.to(device)
                    target = target.to(device)
                    pred = model(sp, q, n)
                    val_loss += criterion(pred, target).item() * target.shape[0]
                    val_total += target.shape[0]
            val_mse = val_loss / val_total if val_total else float("nan")

        history.append({"epoch": epoch, "train_mse": train_mse, "val_mse": val_mse})

        if epoch == 1 or epoch % max(1, config.epochs // 10) == 0:
            if val_mse is not None:
                print(f"Epoch {epoch:5d}: train MSE={train_mse:.6e}, val MSE={val_mse:.6e}")
            else:
                print(f"Epoch {epoch:5d}: train MSE={train_mse:.6e}")

    params = model.parameter_dict()
    print("\nFitted parameters:")
    for key, value in params.items():
        print(f"  {key:>4s} = {value: .6e}")

    full_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    mae, mse = evaluate(model, full_loader, device)
    metrics = {"mae": mae, "mse": mse}
    print(f"\nDataset metrics: MAE={mae:.6e}, MSE={mse:.6e}")

    if config.output is not None:
        config.output.parent.mkdir(parents=True, exist_ok=True)
        with open(config.output, "w", encoding="utf-8") as fh:
            json.dump(params, fh, indent=2)
        print(f"Parameters saved to {config.output}")

    if config.history_output is not None:
        config.history_output.parent.mkdir(parents=True, exist_ok=True)
        with open(config.history_output, "w", encoding="utf-8") as fh:
            json.dump(history, fh, indent=2)
        print(f"Training history saved to {config.history_output}")

    if config.metrics_output is not None:
        config.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        with open(config.metrics_output, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)
        print(f"Evaluation metrics saved to {config.metrics_output}")

    return TrainingResult(parameters=params, history=history, metrics=metrics)
