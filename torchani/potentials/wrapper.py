r"""
This class wraps potentials so that they can function directly with an
input of (species_coordinates, cell, pbc). This is useful for testing purposes
and for some special cases.
"""

import typing as tp

import torch
from torch import Tensor

from torchani.nn import SpeciesConverter
from torchani.tuples import SpeciesEnergies
from torchani.neighbors import _parse_neighborlist, NeighborlistArg, NeighborData
from torchani.potentials.core import Potential


class PotentialWrapper(torch.nn.Module):
    def __init__(
        self,
        potential: Potential,
        periodic_table_index: bool = True,
        neighborlist: NeighborlistArg = "full_pairwise",
    ):
        super().__init__()
        self.periodic_table_index = periodic_table_index
        self.potential = potential
        self.neighborlist = _parse_neighborlist(neighborlist)
        self.znumbers_to_idxs = SpeciesConverter(self.potential.get_chemical_symbols())

    def _validate_inputs(
        self,
        input_: tp.Tuple[Tensor, Tensor],
    ) -> None:
        element_idxs, coordinates = input_
        num_molecules = element_idxs.shape[0]
        num_atoms = element_idxs.shape[1]
        assert element_idxs.shape == (num_molecules, num_atoms), "Bad input shape"
        assert coordinates.shape == (num_molecules, num_atoms, 3), "Bad input shape"

    def forward(
        self,
        species_coordinates: tp.Tuple[Tensor, Tensor],
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
    ) -> SpeciesEnergies:

        self._validate_inputs(species_coordinates)
        species, coordinates = species_coordinates

        if self.periodic_table_index:
            element_idxs = self.znumbers_to_idxs(species_coordinates).species
        else:
            element_idxs = species
        if self.potential.cutoff > 0.0:
            neighbors = self.neighborlist(element_idxs, coordinates, self.potential.cutoff, cell, pbc)
        else:
            neighbors = NeighborData(torch.empty(0), torch.empty(0), torch.empty(0))
        return SpeciesEnergies(species, self.potential(element_idxs, neighbors))
