"""Geometry helpers for the magnetic interaction model."""

from __future__ import annotations

import math
from typing import Iterable, Tuple

import torch
from torch import Tensor


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
