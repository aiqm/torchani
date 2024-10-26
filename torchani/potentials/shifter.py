import typing as tp

import torch
from torch import Tensor

from torchani.constants import GSAES
from torchani.neighbors import NeighborData
from torchani.potentials.core import Potential
from torchani.potentials.wrapper import PotentialWrapper


class EnergyAdder(Potential):
    """Adds constant atomic energies that depend only on the atom types

    Arguments:
        symbols: (:class:``list[str]``): Sequence of symbols corresponding to the
            supported elements.
        self_energies (:class:`list[float]`): Ordered sequence of floats
            corresponding to the self energy of each element. Energies must be
            be in order: ``self_energies[i]`` corresponds to element ``symbols[i]``.
    """

    self_energies: Tensor

    def __init__(self, symbols: tp.Sequence[str], self_energies: tp.Sequence[float]):
        super().__init__(symbols=symbols, cutoff=0.0, is_trainable=False)
        if not len(symbols) == len(self_energies):
            raise ValueError(
                "Chemical symbols and self energies do not match in length"
            )
        self.register_buffer(
            "self_energies", torch.tensor(self_energies, dtype=torch.float)
        )

    # Return a sequence of GSAES sorted by element
    # Example usage:
    # gsaes = sorted_gsaes(('H', 'C', 'S'), 'wB97X', '631Gd')
    # gives: gsaes = [-0.4993213, -37.8338334, -398.0814169]
    # Functional and basis set are case insensitive
    @staticmethod
    def _sorted_gsaes(
        elements: tp.Sequence[str], functional: str, basis_set: str
    ) -> tp.List[float]:
        gsaes = GSAES[f"{functional.lower()}-{basis_set.lower()}"]
        return [gsaes[e] for e in elements]

    @classmethod
    def with_gsaes(cls, elements: tp.Sequence[str], functional: str, basis_set: str):
        r"""Instantiate an EnergyAdder with ground state atomic energies"""
        obj = cls(elements, cls._sorted_gsaes(elements, functional, basis_set))
        return obj

    @torch.jit.export
    def atomic_energies(
        self,
        element_idxs: Tensor,
        neighbors: NeighborData = NeighborData(
            torch.empty(0), torch.empty(0), torch.empty(0)
        ),
        _coordinates: tp.Optional[Tensor] = None,
        ghost_flags: tp.Optional[Tensor] = None,
        ensemble_average: bool = False,
    ) -> Tensor:
        # Compute atomic self energies for a set of species.
        self_atomic_energies = self.self_energies[element_idxs]
        self_atomic_energies = self_atomic_energies.masked_fill(element_idxs == -1, 0.0)
        if ensemble_average:
            return self_atomic_energies
        return self_atomic_energies.unsqueeze(0)

    def forward(
        self,
        element_idxs: Tensor,
        neighbors: NeighborData = NeighborData(
            torch.empty(0), torch.empty(0), torch.empty(0)
        ),
        _coordinates: tp.Optional[Tensor] = None,
        ghost_flags: tp.Optional[Tensor] = None,
    ) -> Tensor:
        return self.atomic_energies(
            element_idxs,
            neighbors,
            _coordinates=_coordinates,
            ghost_flags=ghost_flags,
            ensemble_average=True,
        ).sum(dim=-1)


def StandaloneEnergyAdder(
    symbols: tp.Sequence[str],
    self_energies: tp.Sequence[float],
    periodic_table_index: bool = True,
) -> PotentialWrapper:
    module = EnergyAdder(
        symbols=symbols,
        self_energies=self_energies,
    )
    return PotentialWrapper(
        module,
        periodic_table_index,
    )
