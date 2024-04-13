r"""
These classs wrap modules so that they can function directly with an
input of species_coordinates, cell, pbc. This is useful for testing purposes
and for some special cases, it is especially useful for pairwise potentials,
the "repulsion" and "dispersion" computers, and the EnergyAdder
"""

import typing as tp

import torch
from torch import Tensor
from torch.jit import Final

from torchani.nn import SpeciesConverter
from torchani.tuples import SpeciesEnergies
from torchani.neighbors import FullPairwise, BaseNeighborlist


class Wrapper(torch.nn.Module):
    r"""
    Base class for wrappers that take a species_coordinates tuple (and
    optionally a periodic cell and pbc specification), and output a
    SpeciesEnergies namedtuple
    """

    module: torch.nn.Module

    def __init__(self, *args: tp.Any, **kwargs: tp.Any) -> None:
        super().__init__()

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
        raise NotImplementedError("Must be implemented by subclasses")


class StandaloneWrapper(Wrapper):
    r"""
    Wrapper for modules that need neighbor-data and elements
    """
    periodic_table_index: Final[bool]

    def __init__(
        self,
        module: torch.nn.Module,
        periodic_table_index: bool = True,
        neighborlist: tp.Type[BaseNeighborlist] = FullPairwise,
        neighborlist_cutoff: float = 5.2,
    ):
        super().__init__()
        self.periodic_table_index = periodic_table_index
        self.module = module
        self.neighborlist = neighborlist(neighborlist_cutoff)
        self.znumbers_to_idxs = SpeciesConverter(self.module.get_chemical_symbols())  # type: ignore
        # module must implement:
        # def forward(
        #     self,
        #     element_idxs: Tensor,
        #     neighbors: NeighborData,
        # ) -> Tensor:
        # def get_chemical_symbols(self) -> Sequence[str]

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
        neighbors = self.neighborlist(element_idxs, coordinates, cell, pbc)
        return SpeciesEnergies(species, self.module(element_idxs, neighbors))


class StandaloneElementWrapper(Wrapper):
    r"""
    Wrapper for element only modules that don't need a neighborlist
    """
    periodic_table_index: Final[bool]

    def __init__(
        self,
        module: torch.nn.Module,
        periodic_table_index: bool = True,
    ):
        super().__init__()
        self.periodic_table_index = periodic_table_index
        self.module = module
        self.znumbers_to_idxs = SpeciesConverter(self.module.get_chemical_symbols())  # type: ignore
        # module must implement:
        # def forward(
        #     self,
        #     element_idxs: Tensor,
        # ) -> Tensor:
        # def get_chemical_symbols(self) -> Sequence[str]

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
        return SpeciesEnergies(species, self.module(element_idxs))
