r"""Potentials that calculate repulsive (short-range) interactions"""

import math
import typing as tp

import torch
from torch import Tensor

from torchani.neighbors import Neighbors
from torchani.cutoffs import CutoffArg
from torchani.constants import XTB_REPULSION_ALPHA, XTB_REPULSION_YEFF
from torchani.potentials.core import BasePairPotential


# TODO: trainable?
class RepulsionXTB(BasePairPotential):
    r"""Calculates the xTB repulsion energy terms for a given molecule

    Potential used is as in work by Grimme:
    https://pubs.acs.org/doi/10.1021/acs.jctc.8b01176

    By default ``alpha``, ``yeff`` and ``krep`` parameters are taken from Grimme et. al.

    ``krep_hydrogen`` is only used for H-H repulsive interaction. All other
    interactions use ``krep``
    """

    def __init__(
        self,
        symbols: tp.Sequence[str],
        # Potential
        krep_hydrogen: float = 1.0,
        krep: float = 1.5,
        alpha: tp.Sequence[float] = (),
        yeff: tp.Sequence[float] = (),
        *,  # Cutoff
        cutoff: float = math.inf,
        cutoff_fn: CutoffArg = "smooth",
    ):
        super().__init__(symbols, cutoff=cutoff, cutoff_fn=cutoff_fn)
        alpha = self._validate_elem_seq("alpha", alpha, XTB_REPULSION_ALPHA)
        yeff = self._validate_elem_seq("yeff", yeff, XTB_REPULSION_YEFF)

        num_elem = len(symbols)
        k_rep_ab = torch.full((num_elem, num_elem), krep)
        if 1 in self.atomic_numbers:
            hydrogen_idx = (self.atomic_numbers == 1).nonzero().view(-1)
            k_rep_ab[hydrogen_idx, hydrogen_idx] = krep_hydrogen

        # Pre-calculate pairwise constants for efficiency
        _yeff = torch.tensor(yeff)
        self.register_buffer("y_ab", torch.outer(_yeff, _yeff))
        _alpha = torch.tensor(alpha)
        self.register_buffer("sqrt_alpha_ab", torch.outer(_alpha, _alpha).sqrt())
        self.register_buffer("k_rep_ab", k_rep_ab)

    def pair_energies(
        self,
        element_idxs: Tensor,
        neighbors: Neighbors,
    ) -> Tensor:
        # Clamp distances to prevent singularities when dividing by zero
        # All internal calcs use atomic units, so convert to Bohr
        dists = self.clamp(neighbors.distances) * self.ANGSTROM_TO_BOHR

        # Distances has all interaction pairs within a given cutoff, for a
        # molecule or set of molecules and atom_index12 holds all pairs of
        # indices species is of shape (C x Atoms)
        species12 = element_idxs.view(-1)[neighbors.indices]

        # Find pre-computed constant multiplications for every species pair
        y_ab = self.y_ab[species12[0], species12[1]]
        sqrt_alpha_ab = self.sqrt_alpha_ab[species12[0], species12[1]]
        k_rep_ab = self.k_rep_ab[species12[0], species12[1]]

        # calculates repulsion energies using distances and constants
        return (y_ab / dists) * torch.exp(-sqrt_alpha_ab * (dists**k_rep_ab))
