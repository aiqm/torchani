import typing as tp

import torch
from torch import Tensor

from torchani.wrappers import StandaloneElementWrapper
from torchani.utils import PERIODIC_TABLE, ATOMIC_NUMBERS, sorted_gsaes


class EnergyAdder(torch.nn.Module):
    """Adds atomic energies that depend only on the atom types

    Arguments:
        symbols: (:class:``list[str]``): Sequence of symbols corresponding to the
            supported elements
        self_energies (:class:`collections.abc.Sequence`): Sequence of floats
            corresponding to self energy of each atom type. The numbers should
            be in order, i.e. ``self_energies[i]`` should be atom type ``i``.
    """
    self_energies: Tensor
    atomic_numbers: Tensor

    def __init__(self, symbols: tp.Sequence[str], self_energies: tp.Sequence[float]):
        super().__init__()
        if not len(symbols) == len(self_energies):
            raise ValueError(
                "Chemical symbols and self energies do not match in length"
            )
        numbers = torch.tensor([ATOMIC_NUMBERS[e] for e in symbols], dtype=torch.long)
        self.register_buffer('atomic_numbers', numbers)
        self.register_buffer('self_energies', torch.tensor(self_energies, dtype=torch.float))

    @classmethod
    def with_gsaes(cls, elements: tp.Sequence[str], functional: str, basis_set: str):
        r"""Instantiate an EnergyAdder with ground state atomic energies"""
        obj = cls(elements, sorted_gsaes(elements, functional, basis_set))
        return obj

    @torch.jit.unused
    def get_chemical_symbols(self) -> tp.Tuple[str, ...]:
        return tuple(PERIODIC_TABLE[z] for z in self.atomic_numbers)

    @torch.jit.export
    def _atomic_saes(self, element_idxs: Tensor) -> Tensor:
        # Compute atomic self energies for a set of species.
        self_atomic_energies = self.self_energies[element_idxs]
        self_atomic_energies = self_atomic_energies.masked_fill(element_idxs == -1, 0.0)
        return self_atomic_energies

    def forward(self, element_idxs: Tensor) -> Tensor:
        return self._atomic_saes(element_idxs).sum(dim=1)


def StandaloneEnergyAdder(
    symbols: tp.Sequence[str],
    self_energies: tp.Sequence[float],
    periodic_table_index: bool = True,
) -> StandaloneElementWrapper:
    module = EnergyAdder(
        symbols=symbols,
        self_energies=self_energies,
    )
    return StandaloneElementWrapper(
        module,
        periodic_table_index,
    )
