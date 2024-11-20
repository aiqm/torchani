import typing as tp

import torch
from torch import Tensor

from torchani.constants import GSAES
from torchani.neighbors import Neighbors
from torchani.potentials.core import Potential


class SelfEnergy(Potential):
    """Adds constant atomic energies that depend only on the atom types

    Arguments:
        symbols: |symbols|
        self_energies: Ordered sequence of floats corresponding to the self energy of
            each element. Energies must be be in order: ``self_energies[i]`` corresponds
            to element ``symbols[i]``.
    """

    self_energies: Tensor

    def __init__(self, symbols: tp.Sequence[str], self_energies: tp.Sequence[float]):
        super().__init__(symbols, cutoff=0.0)
        self_energies = self._validate_elem_seq("self_energies", self_energies)
        self.register_buffer("self_energies", torch.tensor(self_energies))

    # Return a sequence of GSAES sorted by element
    # Example usage:
    # gsaes = sorted_gsaes(('H', 'C', 'S'), 'wB97X', '631Gd')
    # gives: gsaes = [-0.4993213, -37.8338334, -398.0814169]
    # Functional and basis set are case insensitive
    @staticmethod
    def _sorted_gsaes(
        symbols: tp.Sequence[str], functional: str, basis_set: str
    ) -> tp.List[float]:
        gsaes = GSAES[f"{functional.lower()}-{basis_set.lower()}"]
        return [gsaes[e] for e in symbols]

    @classmethod
    def with_gsaes(cls, symbols: tp.Sequence[str], functional: str, basis_set: str):
        r"""Instantiate SelfEnergy with ground state atomic energies"""
        return cls(symbols, cls._sorted_gsaes(symbols, functional, basis_set))

    # TODO: Neighbors(empty(0), ...) is a Hack that seems to work for now to
    # avoid having to pass that argument to SelfEnergies
    def forward(
        self,
        elem_idxs: Tensor,
        neighbors: Neighbors = Neighbors(
            torch.empty(0), torch.empty(0), torch.empty(0)
        ),
        _coords: tp.Optional[Tensor] = None,
        ghost_flags: tp.Optional[Tensor] = None,
        atomic: bool = False,
        ensemble_values: bool = False,
    ) -> Tensor:
        if not self._enabled:
            if atomic:
                return neighbors.distances.new_zeros(elem_idxs.shape)
            return neighbors.distances.new_zeros(elem_idxs.shape[0])
        return self.compute(
            elem_idxs, neighbors, _coords, ghost_flags, atomic, ensemble_values
        )

    def compute(
        self,
        elem_idxs: Tensor,
        neighbors: Neighbors,
        _coords: tp.Optional[Tensor] = None,
        ghost_flags: tp.Optional[Tensor] = None,
        atomic: bool = False,
        ensemble_values: bool = False,
    ) -> Tensor:
        # Compute atomic self energies for a set of species.
        self_atomic_energies = self.self_energies[elem_idxs]
        self_atomic_energies = self_atomic_energies.masked_fill(elem_idxs == -1, 0.0)
        if atomic:
            return self_atomic_energies
        return self_atomic_energies.sum(dim=-1)
