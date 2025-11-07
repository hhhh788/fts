"""Differentiable implementation of the Mathematica Hamiltonian."""

from __future__ import annotations

from collections import OrderedDict
import math

import torch
from torch import Tensor, nn

from .geometry import compute_base_vectors, generate_neighbor_vectors


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
