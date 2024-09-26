import typing as tp

import torch
from torch import Tensor

from torchani.sae import sorted_gsaes
from torchani.neighbors import NeighborData
from torchani.potentials.core import Potential
from torchani.potentials.wrapper import PotentialWrapper


class EnergyAdder(Potential):
    """Adds constant atomic energies that depend only on the atom types

    Arguments:
        symbols: (:class:``list[str]``): Sequence of symbols corresponding to the
            supported elements
        self_energies (:class:`collections.abc.Sequence`): Sequence of floats
            corresponding to self energy of each atom type. The numbers should
            be in order, i.e. ``self_energies[i]`` should be atom type ``i``.
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

    @classmethod
    def with_gsaes(cls, elements: tp.Sequence[str], functional: str, basis_set: str):
        r"""Instantiate an EnergyAdder with ground state atomic energies"""
        obj = cls(elements, sorted_gsaes(elements, functional, basis_set))
        return obj

    @torch.jit.export
    def atomic_energies(
        self,
        element_idxs: Tensor,
        neighbors: NeighborData = NeighborData(
            torch.empty(0), torch.empty(0), torch.empty(0)
        ),
        ghost_flags: tp.Optional[Tensor] = None,
        average: bool = False,
    ) -> Tensor:
        # Compute atomic self energies for a set of species.
        self_atomic_energies = self.self_energies[element_idxs]
        self_atomic_energies = self_atomic_energies.masked_fill(element_idxs == -1, 0.0)
        if not average:
            return self_atomic_energies.unsqueeze(0)
        return self_atomic_energies

    def forward(
        self,
        element_idxs: Tensor,
        neighbors: NeighborData = NeighborData(
            torch.empty(0), torch.empty(0), torch.empty(0)
        ),
        ghost_flags: tp.Optional[Tensor] = None,
    ) -> Tensor:
        return self.atomic_energies(
            element_idxs, neighbors, ghost_flags, average=True
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
