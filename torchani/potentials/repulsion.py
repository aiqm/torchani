r"""
Potentials that calculate repulsive (short-range) interactions.
"""

import typing as tp

import torch
from torch import Tensor
from torch.jit import Final

from torchani.units import ANGSTROM_TO_BOHR
from torchani.neighbors import NeighborData
from torchani.cutoffs import CutoffArg
from torchani.constants import XTB_REPULSION_ALPHA, XTB_REPULSION_YEFF
from torchani.potentials.core import PairPotential


class RepulsionXTB(PairPotential):
    r"""Calculates the xTB repulsion energy terms for a given molecule

    Potential used is as in work by Grimme:
    https://pubs.acs.org/doi/10.1021/acs.jctc.8b01176

    By default `alpha`, `yeff` and `krep` parameters are taken from Grimme et. al.
    pairwise_kwargs are passed to `PairPotential`

    `krep_hydrogen` is only used for H-H repulsive interaction. All other
    interactions use `krep`
    """

    ANGSTROM_TO_BOHR: Final[float]

    def __init__(
        self,
        symbols: tp.Sequence[str],
        cutoff: float,
        krep_hydrogen: float = 1.0,
        krep: float = 1.5,
        alpha: tp.Sequence[float] = (),
        yeff: tp.Sequence[float] = (),
        cutoff_fn: CutoffArg = "smooth",
    ):
        super().__init__(symbols=symbols, cutoff=cutoff, cutoff_fn=cutoff_fn)
        num_elements = len(symbols)

        if not alpha:
            alpha = [XTB_REPULSION_ALPHA[j] for j in self.atomic_numbers]
        if not len(alpha) == num_elements:
            raise ValueError("len(alpha) should be equal to len(symbols) if provided")
        if not yeff:
            yeff = [XTB_REPULSION_YEFF[j] for j in self.atomic_numbers]
        if not len(yeff) == num_elements:
            raise ValueError("len(yeff) should be equal to len(symbols) if provided")

        k_rep_ab = torch.full((num_elements, num_elements), krep)
        if 1 in self.atomic_numbers:
            hydrogen_idx = (self.atomic_numbers == 1).nonzero().view(-1)
            k_rep_ab[hydrogen_idx, hydrogen_idx] = krep_hydrogen

        # Pre-calculate pairwise parameters for efficiency
        _yeff = torch.tensor(yeff)
        self.register_buffer("y_ab", torch.outer(_yeff, _yeff))
        _alpha = torch.tensor(alpha)
        self.register_buffer("sqrt_alpha_ab", torch.sqrt(torch.outer(_alpha, _alpha)))
        self.register_buffer("k_rep_ab", k_rep_ab)
        self.ANGSTROM_TO_BOHR = ANGSTROM_TO_BOHR

    def pair_energies(
        self,
        element_idxs: Tensor,
        neighbors: NeighborData,
    ) -> Tensor:
        # Clamp distances to prevent singularities when dividing by zero
        distances = torch.clamp(neighbors.distances, min=1e-7)

        # All internal calculations of this module are made with atomic units,
        # so distances are first converted to bohr
        distances = distances * self.ANGSTROM_TO_BOHR

        # Distances has all interaction pairs within a given cutoff, for a
        # molecule or set of molecules and atom_index12 holds all pairs of
        # indices species is of shape (C x Atoms)
        species12 = element_idxs.flatten()[neighbors.indices]

        # Find pre-computed constant multiplications for every species pair
        y_ab = self.y_ab[species12[0], species12[1]]
        sqrt_alpha_ab = self.sqrt_alpha_ab[species12[0], species12[1]]
        k_rep_ab = self.k_rep_ab[species12[0], species12[1]]

        # calculates repulsion energies using distances and constants
        return (y_ab / distances) * torch.exp(-sqrt_alpha_ab * (distances**k_rep_ab))
