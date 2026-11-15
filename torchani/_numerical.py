r"""Numerical approximations of derivative quantities"""

import torch

from torch import Tensor
from torchani.arch import _ANI

__all__ = ["numerical_virial", "numerical_stress", "numerical_forces"]


# Calculate forces numerically based on the finite-difference method.
# NOTE: Based on ASE source code
def numerical_forces(
    model: _ANI,
    species: Tensor,
    crds: Tensor,
    cell: Tensor | None,
    pbc: Tensor | None,
    eps: float = 1e-3,
    first_n_atoms: int | None = None,
) -> Tensor:
    if crds.ndim != 3:
        raise ValueError("Bad value for coords, must be (mols, atoms, 3)")

    # Note: This function must evaluate forwards 3N times
    num_atoms = crds.size(1) if first_n_atoms is None else first_n_atoms
    forces = crds.new_zeros((crds.size(0), num_atoms, 3))
    for i in range(num_atoms):
        for j in range(3):
            p = crds.clone()
            p[:, i, j] = crds[:, i, j] + eps
            eplus = model((species, p), cell, pbc).energies
            p[:, i, j] = crds[:, i, j] - eps
            eminus = model((species, p), cell, pbc).energies
            forces[:, i, j] = (eminus - eplus) / (2 * eps)
    return forces


# Calculate virial numerically based on the finite-difference method.
# NOTE: Mostly copied from ASE calculators.fd code, but their codebase changes a
# lot, so reproduced herefor consistency across ASE versions
# Also this version is pure torch
def numerical_virial(
    model: _ANI,
    species: Tensor,
    crds: Tensor,
    cell: Tensor | None,
    pbc: Tensor | None,
    # A bit of a larger value for numerical stability
    eps: float = 1e-3,
) -> Tensor:
    device = crds.device
    dtype = crds.dtype

    if cell is None or pbc is None:
        raise ValueError("Cell is required for calculating numerical virial")

    virial = torch.zeros((3, 3), dtype=dtype, device=device)
    for i in range(3):
        # Diagonal terms
        x = torch.eye(3, dtype=dtype, device=device)
        x[i, i] = 1.0 + eps
        eplus = model((species, crds @ x), cell @ x, pbc).energies
        x[i, i] = 1.0 - eps
        eminus = model((species, crds @ x), cell @ x, pbc).energies
        virial[i, i] = (eplus - eminus) / (2 * eps)

        # Off diagonal terms
        x = torch.eye(3, dtype=dtype, device=device)  # Reset
        j = i - 2  # relies on python negative index wrapping
        x[i, j] = x[j, i] = +0.5 * eps
        eplus = model((species, crds @ x), cell @ x, pbc).energies
        x[i, j] = x[j, i] = -0.5 * eps
        eminus = model((species, crds @ x), cell @ x, pbc).energies
        virial[i, j] = virial[j, i] = (eplus - eminus) / (2 * eps)
    return virial


def numerical_stress(
    model: _ANI,
    species: Tensor,
    crds: Tensor,
    cell: Tensor | None,
    pbc: Tensor | None,
    # A bit of a larger value for numerical stability
    eps: float = 1e-3,
) -> Tensor:
    if cell is None or pbc is None:
        raise ValueError("Cell is required for calculating numerical virial")
    volume = torch.linalg.det(cell).abs()
    return numerical_virial(model, species, crds, cell, pbc, eps) / volume
