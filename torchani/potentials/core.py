import typing as tp

import torch
from torch import Tensor

from torchani.constants import ATOMIC_NUMBER, PERIODIC_TABLE
from torchani.cutoffs import parse_cutoff_fn, CutoffArg
from torchani.neighbors import (
    NeighborData,
    _call_global_cell_list,
    _call_global_all_pairs,
)


# TODO: The "_coordinates" input is only required due to a quirk of the
# implementation of the cuAEV
class Potential(torch.nn.Module):
    r"""Base class for all atomic potentials

    Potentials may be many-body potentials or 2-body (pair) potentials
    By default, the base class returns a Tensor of zeros. Subclasses
    must override forward.
    """

    cutoff: float
    atomic_numbers: Tensor

    def __init__(
        self,
        symbols: tp.Sequence[str],
        cutoff: float,
    ):
        super().__init__()
        self.atomic_numbers = torch.tensor(
            [ATOMIC_NUMBER[e] for e in symbols], dtype=torch.long
        )
        self.cutoff = cutoff
        # First element is extra-pair and last element will always be -1
        conv_tensor = -torch.ones(118 + 2, dtype=torch.long)
        for i, znum in enumerate(self.atomic_numbers):
            conv_tensor[znum] = i
        self._conv_tensor = conv_tensor

    @property
    @torch.jit.unused
    def symbols(self) -> tp.Tuple[str, ...]:
        return tuple(PERIODIC_TABLE[z] for z in self.atomic_numbers)

    @torch.jit.unused
    def calc(
        self,
        species: Tensor,
        coordinates: Tensor,
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
        neighborlist: str = "all_pairs",
        periodic_table_index: bool = True,
        atomic: bool = False,
    ) -> Tensor:
        r"""
        Outputs energy, as calculated by the potential

        Output shape depends on the value of ``atomic``, it is either
        ``(molecs, atoms)`` or ``(molecs,)``
        """
        if periodic_table_index:
            elem_idxs = self._conv_tensor.to(species.device)[species]
        else:
            elem_idxs = species
        # Check inputs
        assert elem_idxs.dim() == 2
        assert coordinates.shape == (elem_idxs.shape[0], elem_idxs.shape[1], 3)

        if self.cutoff > 0.0:
            if neighborlist == "cell_list":
                neighbors = _call_global_cell_list(
                    elem_idxs, coordinates, self.cutoff, cell, pbc
                )
            elif neighborlist == "all_pairs":
                neighbors = _call_global_all_pairs(
                    elem_idxs, coordinates, self.cutoff, cell, pbc
                )
        else:
            neighbors = NeighborData(torch.empty(0), torch.empty(0), torch.empty(0))
        return self(
            elem_idxs,
            neighbors,
            atomic=atomic,
        )

    # Forward should be modified by subclasses of Potential, but be careful =S
    def forward(
        self,
        elem_idxs: Tensor,
        neighbors: NeighborData,
        _coordinates: tp.Optional[Tensor] = None,
        ghost_flags: tp.Optional[Tensor] = None,
        atomic: bool = False,
    ) -> Tensor:
        if atomic:
            return neighbors.distances.new_zeros(elem_idxs.shape)
        return neighbors.distances.new_zeros(elem_idxs.shape[0])


class PairPotential(Potential):
    r"""Base class for all pairwise potentials

    Subclasses must override pair_energies
    """

    def __init__(
        self,
        symbols: tp.Sequence[str],
        cutoff: float,
        cutoff_fn: CutoffArg = "dummy",
    ):
        super().__init__(cutoff=cutoff, symbols=symbols)
        self.cutoff_fn = parse_cutoff_fn(cutoff_fn)

    def pair_energies(
        self,
        elem_idxs: Tensor,
        neighbors: NeighborData,
    ) -> Tensor:
        r"""
        Return energy of all pairs of neighbors, disregarding the cutoff fn envelope

        Returns a Tensor of energies, of shape ('pairs',) where 'pairs' is
        the number of neighbor pairs.
        """
        return torch.zeros_like(neighbors.distances)

    # Modulate pair_energies by wrapping with the cutoff fn envelope,
    # and scale ghost pair energies by 0.5
    def _pair_energies_wrapper(
        self,
        elem_idxs: Tensor,
        neighbors: NeighborData,
        ghost_flags: tp.Optional[Tensor] = None,
    ) -> Tensor:
        # Input validation
        assert elem_idxs.ndim == 2, "species should be 2 dimensional"
        assert neighbors.distances.ndim == 1, "distances should be 1 dimensional"
        assert neighbors.indices.ndim == 2, "atom_index12 should be 2 dimensional"
        assert neighbors.distances.shape[0] == neighbors.indices.shape[1]

        pair_energies = self.pair_energies(elem_idxs, neighbors)
        pair_energies *= self.cutoff_fn(neighbors.distances, self.cutoff)

        if ghost_flags is not None:
            if not ghost_flags.numel() == elem_idxs.numel():
                raise ValueError(
                    "ghost_flags and species should have the same number of elements"
                )
            ghost12 = ghost_flags.flatten()[neighbors.indices]
            ghost_mask = torch.logical_or(ghost12[0], ghost12[1])
            pair_energies = torch.where(ghost_mask, pair_energies * 0.5, pair_energies)
        return pair_energies

    # Forward should not be modified by subclasses of PairPotential
    def forward(
        self,
        elem_idxs: Tensor,
        neighbors: NeighborData,
        _coordinates: tp.Optional[Tensor] = None,
        ghost_flags: tp.Optional[Tensor] = None,
        atomic: bool = False,
    ) -> Tensor:
        pair_energies = self._pair_energies_wrapper(elem_idxs, neighbors, ghost_flags)
        molecs_num, atoms_num = elem_idxs.shape
        if atomic:
            energies = neighbors.distances.new_zeros(molecs_num * atoms_num)
            energies.index_add_(0, neighbors.indices[0], pair_energies / 2)
            energies.index_add_(0, neighbors.indices[1], pair_energies / 2)
            energies = energies.view(molecs_num, atoms_num)
        else:
            energies = neighbors.distances.new_zeros(molecs_num)
            molecs_idxs = torch.div(
                neighbors.indices[0], elem_idxs.shape[1], rounding_mode="floor"
            )
            energies.index_add_(0, molecs_idxs, pair_energies)
        return energies
