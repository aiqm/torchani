import typing as tp

import torch
from torch import Tensor

from torchani.tuples import EnergiesAtomicCharges
from torchani.constants import ATOMIC_NUMBER, PERIODIC_TABLE
from torchani.cutoffs import parse_cutoff_fn, CutoffArg
from torchani.neighbors import NeighborData


class Potential(torch.nn.Module):
    r"""Base class for all atomic potentials

    Potentials may be many-body potentials or pairwise potentials
    Subclasses must override 'forward' and 'atomic_energies'

    Optionally, if the potential also assigns atomic charges to atoms,
    subclasses may override energies_and_atomic_charges, by default this
    function makes all atomic charges zero.
    """

    cutoff: float
    is_trainable: bool
    atomic_numbers: Tensor

    def __init__(
        self,
        symbols: tp.Sequence[str],
        cutoff: float,
        is_trainable: bool = False,
    ):
        super().__init__()
        self.atomic_numbers = torch.tensor(
            [ATOMIC_NUMBER[e] for e in symbols], dtype=torch.long
        )
        self.cutoff = cutoff
        self.is_trainable = is_trainable

    @torch.jit.unused
    def get_chemical_symbols(self) -> tp.Tuple[str, ...]:
        return tuple(PERIODIC_TABLE[z] for z in self.atomic_numbers)

    def forward(
        self,
        element_idxs: Tensor,
        neighbors: NeighborData,
        ghost_flags: tp.Optional[Tensor] = None,
    ) -> Tensor:
        r"""
        Outputs "energy", with shape (N,)

        All distances are assumed to lie inside self.cutoff (which may be infinite).
        """
        return torch.zeros(
            element_idxs.shape[0],
            dtype=neighbors.distances.dtype,
            device=element_idxs.device,
        )

    @torch.jit.export
    def energies_and_atomic_charges(
        self,
        element_idxs: Tensor,
        neighbors: NeighborData,
        ghost_flags: tp.Optional[Tensor] = None,
        total_charge: float = 0.0,
    ) -> EnergiesAtomicCharges:
        energies = self(element_idxs, neighbors, ghost_flags)
        atomic_charges = torch.zeros_like(element_idxs, dtype=energies.dtype)
        return EnergiesAtomicCharges(energies, atomic_charges)

    @torch.jit.export
    def atomic_energies(
        self,
        element_idxs: Tensor,
        neighbors: NeighborData,
        ghost_flags: tp.Optional[Tensor] = None,
        ensemble_average: bool = True,
    ) -> Tensor:
        r"""Outputs "atomic_energies"

        All distances are assumed to lie inside self.cutoff (which may be infinite)

        'ensemble_average' controls whether the atomic energies are averaged over
        the ensemble models.

        Shape is (M, N, A) if not averaged over the models,
        or (N, A) if averaged over models

        Potentials that don't have an ensemble of models output shape (1, N, A)
        if ensemble_average=False.
        """
        return torch.zeros_like(element_idxs, dtype=neighbors.distances.dtype)


class PairPotential(Potential):
    r"""Base class for all pairwise potentials

    Subclasses must override pair_energies
    """

    def __init__(
        self,
        cutoff: float,
        symbols: tp.Sequence[str],
        is_trainable: bool = False,
        cutoff_fn: CutoffArg = "dummy",
    ):
        super().__init__(cutoff=cutoff, is_trainable=is_trainable, symbols=symbols)
        self.cutoff_fn = parse_cutoff_fn(cutoff_fn)

    def pair_energies(
        self,
        element_idxs: Tensor,
        neighbors: NeighborData,
    ) -> Tensor:
        r"""Calculate the raw (non-smoothed) energy of all pairs of neighbors

        This function must be overriden by subclasses
        If there are P pairs of neighbors, then
        this must return a tensor of shape (P,)"""
        return torch.zeros_like(neighbors.distances)

    # This function wraps calculate_pair_energies
    # It potentially smooths out the energies using a cutoff function,
    # and it scales pair energies of ghost atoms by 1/2
    def _calculate_pair_energies_wrapper(
        self,
        element_idxs: Tensor,
        neighbors: NeighborData,
        ghost_flags: tp.Optional[Tensor] = None,
    ) -> Tensor:
        # Validation
        assert element_idxs.ndim == 2, "species should be 2 dimensional"
        assert neighbors.distances.ndim == 1, "distances should be 1 dimensional"
        assert neighbors.indices.ndim == 2, "atom_index12 should be 2 dimensional"
        assert neighbors.distances.shape[0] == neighbors.indices.shape[1]

        pair_energies = self.pair_energies(element_idxs, neighbors)

        pair_energies *= self.cutoff_fn(neighbors.distances, self.cutoff)

        if ghost_flags is not None:
            assert (
                ghost_flags.numel() == element_idxs.numel()
            ), "ghost_flags and species should have the same number of elements"
            ghost12 = ghost_flags.flatten()[neighbors.indices]
            ghost_mask = torch.logical_or(ghost12[0], ghost12[1])
            pair_energies = torch.where(ghost_mask, pair_energies * 0.5, pair_energies)
        return pair_energies

    def forward(
        self,
        element_idxs: Tensor,
        neighbors: NeighborData,
        ghost_flags: tp.Optional[Tensor] = None,
    ) -> Tensor:
        pair_energies = self._calculate_pair_energies_wrapper(
            element_idxs,
            neighbors,
            ghost_flags,
        )
        energies = torch.zeros(
            element_idxs.shape[0],
            dtype=pair_energies.dtype,
            device=pair_energies.device,
        )
        molecule_indices = torch.div(
            neighbors.indices[0], element_idxs.shape[1], rounding_mode="floor"
        )
        energies.index_add_(0, molecule_indices, pair_energies)
        return energies

    @torch.jit.export
    def atomic_energies(
        self,
        element_idxs: Tensor,
        neighbors: NeighborData,
        ghost_flags: tp.Optional[Tensor] = None,
        ensemble_average: bool = True,
    ) -> Tensor:
        pair_energies = self._calculate_pair_energies_wrapper(
            element_idxs,
            neighbors,
            ghost_flags,
        )
        molecules_num = element_idxs.shape[0]
        atoms_num = element_idxs.shape[1]

        atomic_energies = torch.zeros(
            molecules_num * atoms_num,
            dtype=pair_energies.dtype,
            device=pair_energies.device,
        )
        atomic_energies.index_add_(0, neighbors.indices[0], pair_energies / 2)
        atomic_energies.index_add_(0, neighbors.indices[1], pair_energies / 2)
        atomic_energies = atomic_energies.view(molecules_num, atoms_num)
        if not ensemble_average:
            return atomic_energies.unsqueeze(0)
        return atomic_energies
