import math
import typing as tp

import torch
from torch import Tensor

from torchani.potentials.core import BasePairPotential
from torchani.neighbors import Neighbors
from torchani.cutoffs import CutoffArg


# TODO: Trainable?
class FixedCoulomb(BasePairPotential):
    def __init__(
        self,
        symbols: tp.Sequence[str],
        dielectric: float = 1.0,
        charges: tp.Sequence[float] = (),
        *,  # Cutoff
        cutoff: float = math.inf,
        cutoff_fn: CutoffArg = "smooth",
    ):
        super().__init__(symbols, cutoff=cutoff, cutoff_fn=cutoff_fn)
        charges = self._validate_elem_seq("charges", charges)
        self._dielectric = dielectric
        self.register_buffer("_charges", torch.tensor(charges), persistent=False)

    def pair_energies(self, elem_idxs: Tensor, neighbors: Neighbors) -> Tensor:
        # Clamp distances to prevent singularities when dividing by zero
        # All internal calcs use atomic units, so convert to Bohr
        dists = self.clamp(neighbors.distances) * self.ANGSTROM_TO_BOHR
        elem_pairs = elem_idxs.view(-1)[neighbors.indices]
        charge_prod = self._charges[elem_pairs[0]] * self._charges[elem_pairs[1]]
        charge_prod /= self._dielectric
        return charge_prod / dists


# TODO: Trainable?
# TODO: Is it correct?
# In the GFN-xTB paper eta are parametrized as (1 + k[l, e[a]]) * eta[e[a]],
# where k is shell and element specific
# charges are parametrized as q[l, a], shell and element specific,
#
# Since the charges are fixed this is equivalent to using multiple FixedMNOK
class FixedMNOK(BasePairPotential):
    def __init__(
        self,
        symbols: tp.Sequence[str],
        dielectric: float = 1.0,
        charges: tp.Sequence[float] = (),
        eta: tp.Sequence[float] = (),
        *,  # Cutoff
        cutoff: float = math.inf,
        cutoff_fn: CutoffArg = "smooth",
    ):
        super().__init__(symbols, cutoff=cutoff, cutoff_fn=cutoff_fn)

        charges = self._validate_elem_seq("charges", charges)
        eta = self._validate_elem_seq("eta", eta)

        self._dielectric = dielectric
        self.register_buffer("_charges", torch.tensor(charges), persistent=False)
        self.register_buffer("_eta", torch.tensor(eta), persistent=False)

    def combine_inv_eta(self, elem_pairs: Tensor) -> Tensor:
        return 2 / (self._eta[elem_pairs[0]] + self._eta[elem_pairs[1]])

    def pair_energies(self, elem_idxs: Tensor, neighbors: Neighbors) -> Tensor:
        # No need to clamp as long as inv_eta ** 2 is nonzero
        # All internal calcs use atomic units, so convert to Bohr
        dists = neighbors.distances * self.ANGSTROM_TO_BOHR
        elem_pairs = elem_idxs.view(-1)[neighbors.indices]
        inv_eta = self.combine_inv_eta(elem_pairs)
        charge_prod = self._charges[elem_pairs[0]] * self._charges[elem_pairs[1]]
        return charge_prod / (dists**2 + inv_eta**2).sqrt()
