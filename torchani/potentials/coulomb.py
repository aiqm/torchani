import math
import typing as tp

import torch
from torch import Tensor

from torchani.potentials.core import BasePairPotential
from torchani.neighbors import Neighbors
from torchani.cutoffs import CutoffArg


class Damp(torch.nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.ones_like(x)


class ErfDamp(Damp):
    def __init__(self, alpha: float = 1.0, r0: float = 0.0) -> None:
        super().__init__()
        self._alpha = alpha
        self._r0 = r0

    def forward(self, x: Tensor) -> Tensor:
        return torch.erf(self._alpha * (x - self._r0))


class TanhDamp(Damp):
    def __init__(self, alpha: float = 1.0, r0: float = 0.0) -> None:
        super().__init__()
        self._alpha = alpha
        self._r0 = r0

    def forward(self, x: Tensor) -> Tensor:
        return torch.tanh(self._alpha * (x - self._r0))


def parse_damp(name: tp.Union[str, torch.nn.Module]) -> Damp:
    if not name:
        return Damp()
    elif name == "erf":
        return ErfDamp(0.5)
    elif name == "tanh":
        return TanhDamp(0.5)
    else:
        assert isinstance(name, Damp)
    return name


# TODO: test me
class Coulomb(BasePairPotential):
    def __init__(
        self,
        symbols: tp.Sequence[str],
        dielectric: float = 1.0,
        *,  # Cutoff
        cutoff: float = math.inf,
        cutoff_fn: CutoffArg = "smooth",
        damp_fn: tp.Union[str, Damp] = "",
    ):
        super().__init__(symbols, cutoff=cutoff, cutoff_fn=cutoff_fn)
        self._dielectric = dielectric
        self._damp_fn = parse_damp(damp_fn)

    def pair_energies(
        self,
        elem_idxs: Tensor,
        neighbors: Neighbors,
        scalars: tp.Optional[Tensor] = None,
    ) -> Tensor:
        assert scalars is not None
        assert scalars.shape == elem_idxs.shape
        # Clamp distances to prevent singularities when dividing by zero
        # All internal calcs use atomic units, so convert to Bohr
        dists = self.clamp(neighbors.distances) * self.ANGSTROM_TO_BOHR
        charges = scalars.view(-1)[neighbors.indices]
        charge_prod = charges[0] * charges[1]
        charge_prod /= self._dielectric
        return charge_prod / dists * self._damp_fn(dists)


# TODO: Trainable?
class FixedCoulomb(BasePairPotential):
    _charges: Tensor

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

    def pair_energies(
        self,
        elem_idxs: Tensor,
        neighbors: Neighbors,
        scalars: tp.Optional[Tensor] = None,
    ) -> Tensor:
        assert scalars is None
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
    _eta: Tensor
    _charges: Tensor

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

    def pair_energies(
        self,
        elem_idxs: Tensor,
        neighbors: Neighbors,
        scalars: tp.Optional[Tensor] = None,
    ) -> Tensor:
        assert scalars is None
        # No need to clamp as long as inv_eta ** 2 is nonzero
        # All internal calcs use atomic units, so convert to Bohr
        dists = neighbors.distances * self.ANGSTROM_TO_BOHR
        elem_pairs = elem_idxs.view(-1)[neighbors.indices]
        inv_eta = self.combine_inv_eta(elem_pairs)
        charge_prod = self._charges[elem_pairs[0]] * self._charges[elem_pairs[1]]
        return charge_prod / (dists**2 + inv_eta**2).sqrt()
