import typing_extensions as tpx
import math
import typing as tp

import torch
from torch import Tensor
from torch.nn import Parameter

from torchani.potentials.core import BasePairPotential
from torchani.neighbors import Neighbors
from torchani.cutoffs import CutoffArg
from torchani.units import HARTREE_TO_KCALPERMOL

# Defaults
_EPS = 0.1 / HARTREE_TO_KCALPERMOL  # Hartree
_SIGMA = 1.5  # Angstrom

# Here for convenience and sanity checks, but not meant to be exposed to users
# Ar from White J Chem Phys 1999, param set 4
# Rest from ff19SB parm
# S from ff19SB frcmod (GAFF2)
# Atom types used:
# N -> N|N3, C -> C*|C5|C4|C, H -> HC, O -> O|O2
# All are similar except HO is 0, 0 (!) and H|HS is 0.6, 0.0157

# Ne from  R. Ramírez and C. P. Herrero "Quantum path-integral study of the phase
# diagram and isotope effects of neon", Journal of Chemical Physics 129 204502 (2008)
_FF19SB_SIGMAS = {
    "H": 1.4870,
    "C": 1.9080,
    "N": 1.8240,
    "O": 1.6612,
    "F": 1.7500,
    "Ne": 2.782,
    "P": 2.1000,
    "S": 1.9825,
    "Cl": 1.948,
    "Ar": 3.346,
    "Br": 2.22,
    "I": 2.35,
}

_FF19SB_EPS = {
    "H": 0.0157,
    "C": 0.0860,
    "N": 0.1700,
    "O": 0.2100,
    "F": 0.0610,
    "Ne": 0.0711,
    "P": 0.2000,
    "S": 0.2824,
    "Cl": 0.265,
    "Ar": 0.24979,
    "Br": 0.320,
    "I": 0.40,
}
_FF19SB_EPS = {k: v / HARTREE_TO_KCALPERMOL for k, v in _FF19SB_EPS.items()}


class _LJ(BasePairPotential):
    r"""Lennard-Jones style potential

    This potential accepts ``sigma`` in Angstrom and ``eps`` in Hartree
    as parameters. The default values are sigma=1.5 and eps=0.1 for all elements.
    These are not meant to accurately represent any real system, please don't use
    them in production.
    """

    def __init__(
        self,
        symbols: tp.Sequence[str],
        eps: tp.Sequence[float] = (),
        sigma: tp.Sequence[float] = (),
        *,  # Cutoff
        trainable: tp.Sequence[str] = (),
        cutoff: float = math.inf,
        cutoff_fn: CutoffArg = "smooth",
    ):
        super().__init__(symbols, cutoff=cutoff, cutoff_fn=cutoff_fn)
        if not set(trainable).issubset(("sigma", "eps")):
            raise ValueError(f"Unsupported parameters in {trainable}")

        for k, v in (("sigma", sigma), ("eps", eps)):
            v = self._validate_elem_seq(k, v, [_SIGMA if k == "sigma" else _EPS] * 118)
            if k in trainable:
                self.register_parameter(f"_{k}", Parameter(torch.tensor(v)))
            else:
                self.register_buffer(f"_{k}", torch.tensor(v), persistent=False)

    @classmethod
    def ff19SB(
        cls,
        symbols: tp.Sequence[str],
        trainable: tp.Sequence[str] = (),
        cutoff: float = math.inf,
        cutoff_fn: CutoffArg = "smooth",
    ) -> tpx.Self:
        r"""Use different defaults. Convenience constructor for debugging

        Use defaults based on the ff19SB Amber force field.
        """
        sigma = tuple(_FF19SB_SIGMAS[s] for s in symbols)
        eps = tuple(_FF19SB_EPS[s] for s in symbols)
        return cls(
            symbols, eps, sigma, trainable=trainable, cutoff=cutoff, cutoff_fn=cutoff_fn
        )

    def combine_eps(self, elem_pairs: Tensor) -> Tensor:
        # Berthelot rule
        return torch.sqrt(self._eps[elem_pairs[0]] * self._eps[elem_pairs[1]])

    def combine_sigma(self, elem_pairs: Tensor) -> Tensor:
        # Lorentz rule
        return (self._sigma[elem_pairs[0]] + self._sigma[elem_pairs[1]]) / 2


class DispersionLJ(_LJ):
    def pair_energies(self, elem_idxs, neighbors: Neighbors):
        elem_pairs = elem_idxs.view(-1)[neighbors.indices]
        eps = self.combine_eps(elem_pairs)
        sigma = self.combine_sigma(elem_pairs)
        x = sigma / self.clamp(neighbors.distances)
        return -4 * eps * x**6


class RepulsionLJ(_LJ):
    def pair_energies(self, elem_idxs, neighbors: Neighbors):
        elem_pairs = elem_idxs.view(-1)[neighbors.indices]
        eps = self.combine_eps(elem_pairs)
        sigma = self.combine_sigma(elem_pairs)
        x = sigma / self.clamp(neighbors.distances)
        return 4 * eps * x**12


class LennardJones(_LJ):
    def pair_energies(self, elem_idxs, neighbors: Neighbors):
        elem_pairs = elem_idxs.view(-1)[neighbors.indices]
        eps = self.combine_eps(elem_pairs)
        sigma = self.combine_sigma(elem_pairs)
        x = sigma / self.clamp(neighbors.distances)
        return 4 * eps * (x**12 - x**6)
