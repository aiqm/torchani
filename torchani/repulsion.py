from typing import Sequence, Union, Optional, Tuple

import torch
from torch import Tensor

from . import units
from .utils import ATOMIC_NUMBERS, PERIODIC_TABLE
from .wrappers import StandaloneWrapper
from .parse_repulsion_constants import alpha_constants, y_eff_constants
from .aev.cutoffs import _parse_cutoff_fn
from .compat import Final


class RepulsionXTB(torch.nn.Module):
    r"""Calculates the xTB repulsion energy terms for a given molecule as seen
    in work by Grimme: https://pubs.acs.org/doi/10.1021/acs.jctc.8b01176"""

    cutoff: Final[float]
    ANGSTROM_TO_BOHR: Final[float]
    y_ab: Tensor
    sqrt_alpha_ab: Tensor
    k_rep_ab: Tensor
    atomic_numbers: Tensor

    def __init__(self,
                 cutoff: float = 5.2,
                 alpha: Sequence[float] = None,
                 y_eff: Sequence[float] = None,
                 k_rep_ab: Optional[Tensor] = None,
                 symbols: Sequence[str] = ('H', 'C', 'N', 'O'),
                 cutoff_fn: Union[str, torch.nn.Module] = 'smooth'):
        super().__init__()
        atomic_numbers = torch.tensor([ATOMIC_NUMBERS[e] for e in symbols], dtype=torch.long)

        # by default alpha, y_eff and krep parameters are taken from Grimme et. al.
        if alpha is None:
            _alpha = torch.tensor(alpha_constants)[atomic_numbers]
        if y_eff is None:
            _y_eff = torch.tensor(y_eff_constants)[atomic_numbers]
        if k_rep_ab is None:
            k_rep_ab = torch.full((len(ATOMIC_NUMBERS) + 1, len(ATOMIC_NUMBERS) + 1), 1.5)
            k_rep_ab[1, 1] = 1.0
            k_rep_ab = k_rep_ab[atomic_numbers, :][:, atomic_numbers]
        assert k_rep_ab is not None
        assert k_rep_ab.shape[0] == len(symbols)
        assert k_rep_ab.shape[1] == len(symbols)
        assert len(_y_eff) == len(symbols)
        assert len(_alpha) == len(symbols)

        self.cutoff_function = _parse_cutoff_fn(cutoff_fn)
        self.cutoff = cutoff
        # pre-calculate pairwise parameters for efficiency
        self.register_buffer("atomic_numbers", atomic_numbers)
        self.register_buffer('y_ab', torch.outer(_y_eff, _y_eff))
        self.register_buffer('sqrt_alpha_ab', torch.sqrt(torch.outer(_alpha, _alpha)))
        self.register_buffer('k_rep_ab', k_rep_ab)
        self.ANGSTROM_TO_BOHR = units.ANGSTROM_TO_BOHR

    @torch.jit.unused
    def get_chemical_symbols(self) -> Tuple[str, ...]:
        return tuple(PERIODIC_TABLE[z] for z in self.atomic_numbers)

    def _calculate_energy(self,
                          element_idxs: Tensor,
                          neighbor_idxs: Tensor,
                          distances: Tensor,
                          diff_vectors: Optional[Tensor] = None,
                          ghost_flags: Optional[Tensor] = None) -> Tensor:

        # clamp distances to prevent singularities when dividing by zero
        distances = torch.clamp(distances, min=1e-7)

        # all internal calculations of this module are made with atomic units,
        # so distances are first converted to bohr
        distances = distances * self.ANGSTROM_TO_BOHR

        assert distances.ndim == 1, "distances should be 1 dimensional"
        assert element_idxs.ndim == 2, "species should be 2 dimensional"
        assert neighbor_idxs.ndim == 2, "atom_index12 should be 2 dimensional"
        assert len(distances) == neighbor_idxs.shape[1]

        # Distances has all interaction pairs within a given cutoff, for a
        # molecule or set of molecules and atom_index12 holds all pairs of
        # indices species is of shape (C x Atoms)
        num_atoms = element_idxs.shape[1]
        species12 = element_idxs.flatten()[neighbor_idxs]

        # find pre-computed constant multiplications for every species pair
        y_ab = self.y_ab[species12[0], species12[1]]
        sqrt_alpha_ab = self.sqrt_alpha_ab[species12[0], species12[1]]
        k_rep_ab = self.k_rep_ab[species12[0], species12[1]]

        # calculates repulsion energies using distances and constants
        prefactor = (y_ab / distances)
        rep_energies = prefactor * torch.exp(-sqrt_alpha_ab * (distances ** k_rep_ab))

        if self.cutoff_function is not None:
            rep_energies *= self.cutoff_function(
                distances,
                self.cutoff * self.ANGSTROM_TO_BOHR
            )

        if ghost_flags is not None:
            assert ghost_flags.numel() == element_idxs.numel(), "ghost_flags and species should have the same number of elements"
            ghost12 = ghost_flags.flatten()[neighbor_idxs]
            ghost_mask = torch.logical_or(ghost12[0], ghost12[1])
            rep_energies = torch.where(ghost_mask, rep_energies * 0.5, rep_energies)

        energies = torch.zeros(
            element_idxs.shape[0],
            dtype=rep_energies.dtype,
            device=rep_energies.device
        )
        molecule_indices = torch.div(neighbor_idxs[0], num_atoms, rounding_mode='floor')
        energies.index_add_(0, molecule_indices, rep_energies)
        return energies

    def forward(
        self,
        element_idxs: Tensor,
        neighbor_idxs: Tensor,
        distances: Tensor,
        diff_vectors: Optional[Tensor] = None,
        ghost_flags: Optional[Tensor] = None
    ) -> Tensor:
        # diff_vector is unused in 2-body potentials
        return self._calculate_energy(
            element_idxs=element_idxs,
            neighbor_idxs=neighbor_idxs,
            distances=distances,
            diff_vectors=diff_vectors,
            ghost_flags=ghost_flags
        )


def StandaloneRepulsionXTB(
    cutoff: float = 5.2,
    alpha: Sequence[float] = None,
    y_eff: Sequence[float] = None,
    k_rep_ab: Optional[Tensor] = None,
    symbols: Sequence[str] = ('H', 'C', 'N', 'O'),
    cutoff_fn: Union[str, torch.nn.Module] = 'smooth',
    **standalone_kwargs,
) -> StandaloneWrapper:
    module = RepulsionXTB(cutoff, alpha, y_eff, k_rep_ab, symbols, cutoff_fn)
    return StandaloneWrapper(module, **standalone_kwargs)
