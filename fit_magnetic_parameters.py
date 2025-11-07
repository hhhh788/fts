"""Command line interface for fitting magnetic interaction parameters."""

from __future__ import annotations

import argparse
from pathlib import Path

from magnetic_fitting import TrainingConfig, train


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(
        description="Fit magnetic interaction parameters using PyTorch"
    )
    parser.add_argument("--data", required=True, type=Path, help="Path to fit_data.csv")
    parser.add_argument("--epochs", type=int, default=2000, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=1024, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay for Adam")
    parser.add_argument(
        "--val-split", type=float, default=0.1, help="Fraction of data for validation"
    )
    parser.add_argument(
        "--output", type=Path, help="Optional JSON file to store fitted parameters"
    )
    parser.add_argument(
        "--history-output",
        type=Path,
        help="Optional JSON file to store per-epoch training/validation losses",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        help="Optional JSON file to store dataset-level MAE/MSE",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Computation device (cpu or cuda)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    return TrainingConfig(
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        val_split=args.val_split,
        output=args.output,
        history_output=args.history_output,
        metrics_output=args.metrics_output,
        device=args.device,
        seed=args.seed,
    )


def main() -> None:  # pragma: no cover - CLI entry point
    config = parse_args()
    train(config)


if __name__ == "__main__":  # pragma: no cover
    main()
