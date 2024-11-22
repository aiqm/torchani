r"""Simple module that calculates a "1-body atomic potential"

You can use `torchani.sae.SelfEnergy` to add a values to all atoms according to their
atomic numbers (typically these values will be ground state atomic energies, or GSAEs.
"""

import typing as tp

import torch
from torch import Tensor

from torchani.constants import GSAES
from torchani._core import _ChemModule


class SelfEnergy(_ChemModule):
    """Adds constant atomic energies that depend only on the atom types

    Arguments:
        symbols: |symbols|
        self_energies: Ordered sequence of floats corresponding to the self energy of
            each element. Energies must be be in order: ``self_energies[i]`` corresponds
            to element ``symbols[i]``.
    """

    self_energies: Tensor
    _enabled: bool

    def __init__(self, symbols: tp.Sequence[str], self_energies: tp.Sequence[float]):
        super().__init__(symbols)
        self_energies = self._validate_elem_seq("self_energies", self_energies)
        self.register_buffer("self_energies", torch.tensor(self_energies))
        self._enabled = True

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
        atomic: bool = False,
    ) -> Tensor:
        # Compute atomic self energies for a set of species.
        self_atomic_energies = self.self_energies[elem_idxs]
        self_atomic_energies = self_atomic_energies.masked_fill(elem_idxs == -1, 0.0)
        if atomic:
            return self_atomic_energies
        return self_atomic_energies.sum(dim=-1)
