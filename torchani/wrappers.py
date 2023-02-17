from typing import Tuple, Optional, Type

import torch
from torch import Tensor
from torch.nn import Module

from .compat import Final
from .nn import SpeciesConverter
from .structs import SpeciesEnergies
from .aev import FullPairwise, BaseNeighborlist


# This helper class wraps modules so that they can function directly with
# an input of species_coordinates, cell, pbc. This is useful for testing
# purposes and for some special cases, it is especially useful for the
# "repulsion" and "dispersion" computers
# IMPORTANT: This should be inherited from FIRST (leftmost in inheritance list)
# for the scheme to work properly
class StandaloneWrapper(Module):
    periodic_table_index: Final[bool]
    needs_neighbor_data: Final[bool]

    def __init__(
        self,
        module: torch.nn.Module,
        periodic_table_index: bool = True,
        needs_neighbor_data: bool = True,
        neighborlist: Type[BaseNeighborlist] = FullPairwise,
        neighborlist_cutoff: float = 5.2,
    ):
        super().__init__()
        self.periodic_table_index = periodic_table_index
        self.needs_neighbor_data = needs_neighbor_data
        self.module = module
        # neighborlist uses radial cutoff only
        self.neighborlist = neighborlist(neighborlist_cutoff)
        self.znumbers_to_idxs = SpeciesConverter(self.module.get_chemical_symbols())  # type: ignore
        # module must implement
        # def _calculate_energy(
        #     self,
        #     element_idxs: Tensor,
        #     coordinates: Tensor,
        #     neighbor_idxs: Tensor,
        #     distances: Tensor,
        #     diff_vectors: Optional[Tensor] = None,
        # ) -> SpeciesEnergies:
        # module must also implement get_chemical_symbols() -> Sequence[str]

    def _validate_inputs(
        self,
        species_coordinates: Tuple[Tensor, Tensor],
    ) -> None:
        species, coordinates = species_coordinates
        # check shapes for correctness
        assert species.dim() == 2
        assert coordinates.dim() == 3
        assert (species.shape == coordinates.shape[:2]) and (coordinates.shape[2] == 3)

    def forward(
        self,
        species_coordinates: Tuple[Tensor, Tensor],
        cell: Optional[Tensor] = None,
        pbc: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:

        self._validate_inputs(species_coordinates)
        species, coordinates = species_coordinates

        if self.periodic_table_index:
            element_idxs = self.znumbers_to_idxs(species_coordinates).species
        else:
            element_idxs = species

        # the coordinates that are input into the neighborlist are **not**
        # assumed to be mapped into the central cell for pbc calculations, and
        # **in general are not**
        if self.needs_neighbor_data:
            neighbor_data = self.neighborlist(element_idxs, coordinates, cell, pbc)
        else:
            neighbor_data = self.neighborlist.dummy()

        energy = self.module._calculate_energy(  # type: ignore
            element_idxs,
            neighbor_idxs=neighbor_data.indices,
            distances=neighbor_data.distances,
            diff_vectors=neighbor_data.diff_vectors,
        )
        return SpeciesEnergies(species, torch.zeros(species.shape[0], dtype=energy.dtype, device=energy.device) + energy)


# This is used for sequential calls AEVComputer -> NN -> Potential -> Potential -> ...
class SequentialWrapper(Module):
    def __init__(self, module: Module):
        super().__init__()
        self.module = module
        # module must implement
        # _calculate_sequential((element_idxs, Tensor)) -> (element_idxs, Tensor)

    def forward(
        self,
        element_idxs_tensor: Tuple[Tensor, Tensor],
        cell: Optional[Tensor] = None,
        pbc: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        return self.module._calculate_sequential(element_idxs_tensor)  # type: ignore
