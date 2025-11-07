"""Magnetic interaction parameter fitting using PyTorch.

This module implements a differentiable version of the Mathematica model
described in the prompt.  It loads a CSV file containing rows of
``Sqx,Sqy,Sqz,qx,qy,qz,nx,ny,nz,E`` and optimises the interaction
parameters so that the model energy matches the provided ``E`` values.

Usage (command line)::

    python fit_magnetic_parameters.py --data fit_data.csv \
        --epochs 5000 --batch-size 1024 --lr 1e-3 --val-split 0.1

The implementation keeps all symbolic expressions differentiable so that
any first-order optimiser from ``torch.optim`` can be used.  All
computations are performed in ``float64`` for numerical stability.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, random_split


def rotation_matrix_z(theta: Tensor) -> Tensor:
    """Return a Z-axis rotation matrix for a scalar angle ``theta``."""

    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    return torch.stack(
        (
            torch.stack((cos_t, -sin_t, torch.zeros_like(theta)), dim=-1),
            torch.stack((sin_t, cos_t, torch.zeros_like(theta)), dim=-1),
            torch.stack((torch.zeros_like(theta), torch.zeros_like(theta), torch.ones_like(theta)), dim=-1),
        ),
        dim=-2,
    )


def generate_neighbor_vectors(base_vector: Tensor, num_neighbors: int) -> Tensor:
    """Generate the rotated neighbour vectors used in the model."""

    angles = torch.arange(num_neighbors, dtype=base_vector.dtype, device=base_vector.device)
    angles = angles * (math.pi / num_neighbors)
    rotations = rotation_matrix_z(angles)
    rotated = torch.matmul(rotations, base_vector.unsqueeze(-1)).squeeze(-1)
    return rotated


def lattice_transform(delta: Iterable[float], lattice: Tensor) -> Tensor:
    """Apply the lattice matrix to a fractional displacement ``delta``."""

    vector = torch.tensor(delta, dtype=lattice.dtype, device=lattice.device)
    return torch.matmul(vector.unsqueeze(0), lattice).squeeze(0)


def compute_base_vectors(lattice: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Replicate the Mathematica construction of R1nn, R2nn, R3nn."""

    r1 = lattice_transform(
        (
            -0.0000000000014,
            -0.000000000002,
            0.4999999999998,
        ),
        lattice,
    ) - lattice_transform(
        (
            -0.0000000000014,
            0.99999999999998,
            0.49999999999998,
        ),
        lattice,
    )

    r2 = lattice_transform(
        (
            0.9999999999986,
            -0.000000000002,
            0.4999999999998,
        ),
        lattice,
    ) - lattice_transform(
        (
            -0.000000000014,
            0.9999999999998,
            0.4999999999998,
        ),
        lattice,
    )

    r3 = lattice_transform(
        (
            0.000000000014,
            -1.000000000002,
            0.4999999999998,
        ),
        lattice,
    ) - lattice_transform(
        (
            -0.000000000014,
            0.9999999999998,
            0.4999999999998,
        ),
        lattice,
    )

    return r1, r2, r3


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


class MagneticEnergyModel(nn.Module):
    """Differentiable replica of the Mathematica Hamiltonian."""

    def __init__(self) -> None:
        super().__init__()

        lattice = torch.tensor(
            (
                (3.968395879315, 0.0, 0.0),
                (-1.9841979742, 3.4367317036, 0.0),
                (0.0, 0.0, 30.0342597961),
            ),
            dtype=torch.double,
        )

        r1, r2, r3 = compute_base_vectors(lattice)

        self.register_buffer("r1_vectors", generate_neighbor_vectors(r1, 6))
        self.register_buffer("r2_vectors", generate_neighbor_vectors(r2, 6))
        self.register_buffer("r3_vectors", generate_neighbor_vectors(r3, 6))

        # Coordination numbers
        self.N1b = 1.0
        self.N2b = 2.0

        # Scalar parameters
        self.J1 = nn.Parameter(torch.zeros((), dtype=torch.double))
        self.J2 = nn.Parameter(torch.zeros((), dtype=torch.double))
        self.J8 = nn.Parameter(torch.zeros((), dtype=torch.double))
        self.J9 = nn.Parameter(torch.zeros((), dtype=torch.double))
        self.J10 = nn.Parameter(torch.zeros((), dtype=torch.double))
        self.J11 = nn.Parameter(torch.zeros((), dtype=torch.double))
        self.E0 = nn.Parameter(torch.zeros((), dtype=torch.double))

        # Six-component parameter vectors for the anisotropic interactions.
        self.J3 = nn.Parameter(torch.zeros(6, dtype=torch.double))
        self.J4 = nn.Parameter(torch.zeros(6, dtype=torch.double))
        self.J5 = nn.Parameter(torch.zeros(6, dtype=torch.double))
        self.J6 = nn.Parameter(torch.zeros(6, dtype=torch.double))

    # === Helper expressions ===
    @staticmethod
    def j1(sp: Tensor) -> Tensor:
        return sp[..., 0] ** 2 + sp[..., 1] ** 2

    @staticmethod
    def j2(sp: Tensor) -> Tensor:
        return sp[..., 2] ** 2

    @staticmethod
    def j3(sp: Tensor, sq: Tensor) -> Tensor:
        return (
            sp[..., 0] * sq[..., 0]
            + 0.3464 * sp[..., 0] * sq[..., 1]
            + 0.3464 * sp[..., 1] * sq[..., 0]
            + 0.6 * sp[..., 1] * sq[..., 1]
        )

    @staticmethod
    def j4(sp: Tensor, sq: Tensor) -> Tensor:
        return sp[..., 0] * sq[..., 1] + sp[..., 1] * sq[..., 0] - 1.1547 * sp[..., 1] * sq[..., 1]

    @staticmethod
    def j5(sp: Tensor, sq: Tensor) -> Tensor:
        return (
            sp[..., 0] * sq[..., 2]
            + 0.5774 * sp[..., 1] * sq[..., 2]
            + sp[..., 2] * sq[..., 0]
            + 0.5774 * sp[..., 2] * sq[..., 1]
        )

    @staticmethod
    def j6(sp: Tensor, sq: Tensor) -> Tensor:
        return sp[..., 2] * sq[..., 2]

    @staticmethod
    def j8(sp: Tensor, sq: Tensor) -> Tensor:
        return sp[..., 0] * sq[..., 0] + sp[..., 1] * sq[..., 1] + sp[..., 2] * sq[..., 2]

    @staticmethod
    def j9(sp: Tensor, sq: Tensor) -> Tensor:
        return (
            sp[..., 0] ** 2 * sq[..., 0] ** 2
            + 0.5 * sp[..., 0] ** 2 * sq[..., 1] ** 2
            - 0.5 * sp[..., 0] ** 2
            + sp[..., 0] * sp[..., 1] * sq[..., 0] * sq[..., 1]
            + sp[..., 0] * sp[..., 2] * sq[..., 0] * sq[..., 2]
            + 0.5 * sp[..., 1] ** 2 * sq[..., 0] ** 2
            + sp[..., 1] ** 2 * sq[..., 1] ** 2
            - 0.5 * sp[..., 1] ** 2
            + sp[..., 1] * sp[..., 2] * sq[..., 1] * sq[..., 2]
            - 0.5 * sq[..., 0] ** 2
            - 0.5 * sq[..., 1] ** 2
        )

    j10 = j8
    j11 = j9

    def _compute_sq(self, sp: Tensor, q: Tensor, n: Tensor, neighbor_vectors: Tensor) -> Tensor:
        """Evaluate the Sq function for all neighbours in ``neighbor_vectors``."""

        cross_term = torch.cross(n, sp, dim=-1)

        sq_values = []
        for vector in neighbor_vectors:
            scaled_vector = vector * (2.0 * math.pi)
            phase = torch.matmul(q, scaled_vector)
            cos_term = torch.cos(phase).unsqueeze(-1)
            sin_term = torch.sin(phase).unsqueeze(-1)
            sq = cos_term * sp + sin_term * cross_term
            sq_values.append(sq)

        return torch.stack(sq_values, dim=1)

    def forward(self, sp: Tensor, q: Tensor, n: Tensor) -> Tensor:
        sp = sp.double()
        q = q.double()
        n = n.double()

        # Base contributions (single-site terms).
        j1_total = self.j1(sp)
        j2_total = self.j2(sp)

        # Neighbour dependent terms.
        sq1 = self._compute_sq(sp, q, n, self.r1_vectors)
        sq2 = self._compute_sq(sp, q, n, self.r2_vectors)
        sq3 = self._compute_sq(sp, q, n, self.r3_vectors)

        j3_vals = self.j3(sp.unsqueeze(1), sq1)
        j4_vals = self.j4(sp.unsqueeze(1), sq1)
        j5_vals = self.j5(sp.unsqueeze(1), sq1)
        j6_vals = self.j6(sp.unsqueeze(1), sq1)

        j8_vals = self.j8(sp.unsqueeze(1), sq2)
        j9_vals = self.j9(sp.unsqueeze(1), sq2)
        j10_vals = self.j10(sp.unsqueeze(1), sq3)
        j11_vals = self.j11(sp.unsqueeze(1), sq3)

        heff = (self.J1 * j1_total + self.J2 * j2_total) / self.N1b
        heff = heff + (self.J8 * j8_vals.sum(dim=1) + self.J9 * j9_vals.sum(dim=1)) / self.N2b
        heff = heff + (self.J10 * j10_vals.sum(dim=1) + self.J11 * j11_vals.sum(dim=1)) / self.N2b

        heff = heff + (torch.matmul(j3_vals, self.J3) + torch.matmul(j4_vals, self.J4)) / self.N2b
        heff = heff + (torch.matmul(j5_vals, self.J5) + torch.matmul(j6_vals, self.J6)) / self.N2b

        return heff + self.E0

    def parameter_dict(self) -> OrderedDict:
        params = OrderedDict()
        params["J1"] = float(self.J1.detach().cpu())
        params["J2"] = float(self.J2.detach().cpu())
        params["J8"] = float(self.J8.detach().cpu())
        params["J9"] = float(self.J9.detach().cpu())
        params["J10"] = float(self.J10.detach().cpu())
        params["J11"] = float(self.J11.detach().cpu())
        for idx, tensor in enumerate((self.J3, self.J4, self.J5, self.J6), start=3):
            for comp in range(6):
                params[f"J{idx}{comp + 1}"] = float(tensor[comp].detach().cpu())
        params["E0"] = float(self.E0.detach().cpu())
        return params


@dataclass
class TrainingConfig:
    data_path: Path
    epochs: int = 2000
    batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    val_split: float = 0.1
    output: Path | None = None
    device: str = "cpu"
    seed: int = 42


def create_dataloaders(
    dataset: MagneticDataset, batch_size: int, val_split: float, seed: int
) -> Tuple[DataLoader, DataLoader | None]:
    val_split = max(0.0, min(float(val_split), 0.5))
    num_val = int(len(dataset) * val_split)

    if num_val > 0:
        generator = torch.Generator().manual_seed(seed)
        train_set, val_set = random_split(
            dataset, [len(dataset) - num_val, num_val], generator=generator
        )
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    else:
        train_set = dataset
        val_loader = None

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader


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


def train(config: TrainingConfig) -> None:
    torch.manual_seed(config.seed)

    dataset = MagneticDataset(config.data_path)
    train_loader, val_loader = create_dataloaders(
        dataset, config.batch_size, config.val_split, config.seed
    )

    device = torch.device(config.device)

    model = MagneticEnergyModel().to(device).double()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.MSELoss()

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

        avg_loss = epoch_loss / total

        if epoch == 1 or epoch % max(1, config.epochs // 10) == 0:
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
                val_avg = val_loss / val_total if val_total else float("nan")
                print(f"Epoch {epoch:5d}: train MSE={avg_loss:.6e}, val MSE={val_avg:.6e}")
            else:
                print(f"Epoch {epoch:5d}: train MSE={avg_loss:.6e}")

    params = model.parameter_dict()
    print("\nFitted parameters:")
    for key, value in params.items():
        print(f"  {key:>4s} = {value: .6e}")

    full_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    mae, mse = evaluate(model, full_loader, device)
    print(f"\nDataset metrics: MAE={mae:.6e}, MSE={mse:.6e}")

    if config.output is not None:
        config.output.parent.mkdir(parents=True, exist_ok=True)
        with open(config.output, "w", encoding="utf-8") as fh:
            json.dump(params, fh, indent=2)
        print(f"Parameters saved to {config.output}")


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Fit magnetic interaction parameters using PyTorch")
    parser.add_argument("--data", required=True, type=Path, help="Path to fit_data.csv")
    parser.add_argument("--epochs", type=int, default=2000, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=1024, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay for Adam")
    parser.add_argument("--val-split", type=float, default=0.1, help="Fraction of data for validation")
    parser.add_argument("--output", type=Path, help="Optional JSON file to store fitted parameters")
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
        device=args.device,
        seed=args.seed,
    )


def main() -> None:  # pragma: no cover - CLI entry point
    config = parse_args()
    train(config)


if __name__ == "__main__":  # pragma: no cover
    main()
