r"""Fetch the dispersion constants from .pkl and .csv files, and provide them as Tensor

There are 4 different kinds of constants needed for D3 dispersion:

- Precalculated C6 coefficients
    shape (Elements, Elements, Ref, Ref), where "Ref" is the number of references
    (Grimme et. al. provides 5)
    This means for each pair of elements and reference indices there is an
    associated precalc C6 coeff
- Precalculated coordination numbers
    shape (Elements, Elements, Ref, Ref, 2)
    Where the final axis indexes the coordination number of the first or second
    atom respectively.
    This means for each pair of elements and reference indices there is an
    associated coordination number for the first and second items.
"""
import h5py
import typing as tp
import math
import pickle
from pathlib import Path

import torch
from torch import Tensor

SUPPORTED_D3_ELEMENTS = 94


def _make_symmetric(x: Tensor) -> Tensor:
    assert x.ndim == 1
    size = (math.sqrt(1 + 8 * len(x)) - 1) / 2
    if not size.is_integer():
        raise ValueError(
            "Input tensor must be of size x * (x + 1) / 2 where x is an integer"
        )
    size = int(size)
    x_symmetric = torch.zeros((size, size))
    _lower_diagonal_mask = torch.tril(torch.ones((size, size), dtype=torch.bool))
    x_symmetric.masked_scatter_(_lower_diagonal_mask, x)
    for j in range(size):
        for i in range(size):
            x_symmetric[j, i] = x_symmetric[i, j]
    return x_symmetric


def get_c6_constants() -> tp.Tuple[Tensor, Tensor, Tensor]:
    with h5py.File(str(Path(__file__).resolve().parent / "c6.h5"), "r") as f:
        c6_constants = torch.from_numpy(f["all/constants"][:])
        c6_coordnums_a = torch.from_numpy(f["all/coordnums_a"][:])
        c6_coordnums_b = torch.from_numpy(f["all/coordnums_b"][:])
    return c6_constants, c6_coordnums_a, c6_coordnums_b


def get_cutoff_radii() -> Tensor:
    # cutoff radii are in angstroms
    num_cutoff_radii = SUPPORTED_D3_ELEMENTS * (SUPPORTED_D3_ELEMENTS + 1) / 2
    path = Path(__file__).parent.joinpath('cutoff_radii.pkl').resolve()
    with open(path, 'rb') as f:
        cutoff_radii = torch.tensor(pickle.load(f))
    assert len(cutoff_radii) == num_cutoff_radii
    cutoff_radii = _make_symmetric(cutoff_radii)
    cutoff_radii = torch.cat(
        (
            torch.zeros(len(cutoff_radii), dtype=cutoff_radii.dtype).unsqueeze(0),
            cutoff_radii,
        ),
        dim=0,
    )
    cutoff_radii = torch.cat(
        (
            torch.zeros(cutoff_radii.shape[0], dtype=cutoff_radii.dtype).unsqueeze(1),
            cutoff_radii,
        ),
        dim=1,
    )
    return cutoff_radii


def get_covalent_radii() -> Tensor:
    # covalent radii are in angstroms covalent radii are used for the
    # calculation of coordination numbers covalent radii in angstrom taken
    # directly from Grimme et. al. dftd3 source code, in turn taken from Pyykko
    # and Atsumi, Chem. Eur. J. 15, 2009, 188-197 values for metals decreased
    # by 10 %
    path = Path(__file__).parent.joinpath('covalent_radii.pkl').resolve()
    with open(path, 'rb') as f:
        covalent_radii = torch.tensor(pickle.load(f))
    assert len(covalent_radii) == SUPPORTED_D3_ELEMENTS
    # element 0 is a dummy element
    covalent_radii = torch.cat((torch.tensor([0.0]), covalent_radii))
    return covalent_radii


def get_sqrt_empirical_charge() -> Tensor:
    # empirical Q is in atomic units, these correspond to sqrt(0.5 * sqrt(Z) *
    # <r**2>/<r**4>) in Grimme's code these are "r2r4", and are used to
    # calculate the C8 values
    path = Path(__file__).parent.joinpath('sqrt_empirical_charge.pkl').resolve()
    with open(path, 'rb') as f:
        sqrt_empirical_charge = torch.tensor(pickle.load(f))
    assert len(sqrt_empirical_charge) == SUPPORTED_D3_ELEMENTS
    # element 0 is a dummy element
    sqrt_empirical_charge = torch.cat((torch.tensor([0.0]), sqrt_empirical_charge))
    return sqrt_empirical_charge
