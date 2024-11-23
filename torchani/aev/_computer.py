import warnings
import os
import typing as tp

import torch
from torch import Tensor
from torch.jit import Final
import typing_extensions as tpx

from torchani.tuples import SpeciesAEV
from torchani.neighbors import (
    _parse_neighborlist,
    neighbors_to_triples,
    discard_outside_cutoff,
    NeighborlistArg,
    Neighbors,
)
from torchani.cutoffs import CutoffArg
from torchani.aev._terms import (
    _parse_angular_term,
    _parse_radial_term,
    ANIAngular,
    ANIRadial,
    RadialArg,
    AngularArg,
)
from torchani.csrc import CUAEV_IS_INSTALLED

# Envvar meant to be used by developers to debug AEV branch
_PRINT_AEV_BRANCH = os.getenv("TORCHANI_PRINT_AEV_BRANCH") == "1"


def jit_unused_if_no_cuaev():
    def decorator(func):
        if CUAEV_IS_INSTALLED:
            return torch.jit.export(func)
        return torch.jit.unused(func)

    return decorator


class AEVComputer(torch.nn.Module):
    r"""Base class for modules that compute AEVs

    Can be used to compute local atomic features (atomic environment vectors or AEVs),
    given a batch of molecules.

    Args:
        radial: The module used to compute the radial part of the AEVs
        angular: The module used to compute the angular part of the AEVs
        num_species: The number of elements this module supports
        strategy: The AEV computation strategy. Valid values are:
            - 'cuaev': Use the hand-coded CUDA cuAEV extension
            - 'pyaev': Use pytorch exclusively
            - 'auto': Choose the most performant available option
        cutoff_fn: Cutoff fn to use for the radial and angular terms
        neighborlist: Kind of neighborlist for internal computation.
        Valid values are: 'all_pairs', 'cell_list', 'adaptive'
    """

    num_species: Final[int]
    num_species_pairs: Final[int]

    angular_len: Final[int]
    radial_len: Final[int]
    out_dim: Final[int]

    triu_index: Tensor
    _strategy: str
    _cuaev_is_avail: bool
    _cuaev_computer_is_init: bool

    def __init__(
        self,
        radial: RadialArg,
        angular: AngularArg,
        num_species: int,
        strategy: str = "pyaev",
        cutoff_fn: tp.Optional[CutoffArg] = None,
        neighborlist: NeighborlistArg = "all_pairs",
    ):
        super().__init__()
        self._print_aev_branch = _PRINT_AEV_BRANCH
        self.num_species = num_species
        self.num_species_pairs = num_species * (num_species + 1) // 2
        self.register_buffer("triu_index", self._calculate_triu_index(num_species))

        # Terms
        self.radial = _parse_radial_term(radial)
        self.angular = _parse_angular_term(angular)
        if not (self.angular.cutoff_fn.is_same(self.radial.cutoff_fn)):
            raise ValueError("Cutoff fn must be the same for angular and radial terms")
        if self.angular.cutoff > self.radial.cutoff:
            raise ValueError(
                f"Angular cutoff {self.angular.cutoff}"
                f" should be smaller than radial cutoff {self.radial.cutoff}"
            )
        self._cuaev_cutoff_fn = self.angular.cutoff_fn._cuaev_name

        # Neighborlist
        self.neighborlist = _parse_neighborlist(neighborlist)

        # Lenghts
        self.radial_len = self.radial.num_feats * self.num_species
        self.angular_len = self.angular.num_feats * self.num_species_pairs
        self.out_dim = self.radial_len + self.angular_len

        # Perform init and checks of cuAEV
        # Check if the cuaev and the cuaev fused are available for use
        self._cuaev_is_avail = self._check_cuaev_avail()

        # Do cuAEV dummy init ('registration'), even if not used. Required by JIT
        if CUAEV_IS_INSTALLED:
            self._register_cuaev_computer()

        # Delay cuAEV true init to fwd, to put tensors in correct device
        self._cuaev_computer_is_init = False

        # Check the requested strategy is available
        if strategy == "auto":
            strategy = "cuaev" if self._cuaev_is_avail else "pyaev"

        if strategy == "pyaev":
            pass
        elif strategy in ["cuaev", "cuaev-fused", "cuaev-interface"]:
            self._check_cuaev_avail(raise_exc=True)
        else:
            raise ValueError(f"Unsupported strategy {strategy}")
        self._strategy = strategy

    @property
    def strategy(self) -> str:
        return self._strategy

    @torch.jit.export
    def set_strategy(
        self,
        strat: str,
    ) -> None:
        if strat == "auto":
            strat = "cuaev" if self._cuaev_is_avail else "pyaev"
        if strat == "pyaev":
            pass
        elif strat == "cuaev" or strat == "cuaev-fused" or strat == "cuaev-interface":
            if not self._cuaev_is_avail:
                raise ValueError(f"{strat} strategy is not available")
        else:
            raise ValueError("Unknown compute strategy")
        self._strategy = strat

    def _check_cuaev_avail(self, raise_exc: bool = False) -> bool:
        # The fused cuaev calculates using all pairs, and bypasses the neighborlist
        if not CUAEV_IS_INSTALLED:
            if raise_exc:
                raise ValueError("cuAEV is not installed")
            return False
        if (
            not self._cuaev_cutoff_fn
            or type(self.angular) is not ANIAngular
            or type(self.radial) is not ANIRadial
        ):
            if raise_exc:
                raise ValueError(
                    "cuAEV only supports ANIAngular and ANIAngular terms, "
                    "and CosineCutoff or SmoothCutoff functions (with default args)"
                )
            return False
        return True

    def extra_repr(self) -> str:
        r""":meta private:"""
        radial_perc = f"{self.radial_len / self.out_dim * 100:.2f}% of feats"
        angular_perc = f"{self.angular_len / self.out_dim * 100:.2f}% of feats"
        parts = [
            r"#  " f"out_dim={self.out_dim}",
            r"#  " f"radial_len={self.radial_len} ({radial_perc})",
            r"#  " f"angular_len={self.angular_len} ({angular_perc})",
            f"num_species={self.num_species},",
            f"strategy={self._strategy},",
        ]
        return " \n".join(parts)

    @staticmethod
    def _calculate_triu_index(num_species: int) -> Tensor:
        # Helper method for initialization
        species1, species2 = torch.triu_indices(num_species, num_species).unbind(0)
        pair_index = torch.arange(species1.shape[0], dtype=torch.long)
        ret = torch.zeros(num_species, num_species, dtype=torch.long)
        ret[species1, species2] = pair_index
        ret[species2, species1] = pair_index
        return ret

    def forward(
        self,
        elem_idxs: Tensor,
        coords: Tensor,
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
    ) -> Tensor:
        r"""Compute AEVs for a batch of molecules

        Arguments:
            elem_idxs: |elem_idxs|
            coords: |coords|
            cell: |cell|
            pbc: |pbc|

        Returns:
            |aevs|
        """
        if not (torch.jit.is_scripting() or torch.compiler.is_compiling()):
            if isinstance(elem_idxs, tuple):
                warnings.warn(
                    "You seem to be attempting to call "
                    "`_, aevs = aev_computer((species, coords), cell, pbc)`. "
                    "This signature was modified in TorchANI 3, and will be removed"
                    "Use `aevs = aev_computer(species, coords, cell, pbc)` instead."
                )
                cell = coords
                pbc = cell
                _elem_idxs, coords = elem_idxs
                return SpeciesAEV(_elem_idxs, self(_elem_idxs, coords, cell, pbc))
            if pbc is not None and not pbc.any():
                raise ValueError(
                    "pbc = torch.tensor([False, False, False]) is not supported anymore"
                    " please use pbc = None"
                )
        # Check input shape correctness and validate cutoffs
        assert elem_idxs.dim() == 2
        assert coords.shape == (elem_idxs.shape[0], elem_idxs.shape[1], 3)
        assert self.angular.cutoff < self.radial.cutoff

        if self._strategy == "cuaev-fused":
            if pbc is not None:
                raise RuntimeError("cuAEV-fused doesn't support PBC")
            return self._cuaev_fused(elem_idxs, coords)
        if self._strategy == "cuaev" and pbc is None:
            return self._cuaev_fused(elem_idxs, coords)

        # IMPORTANT: If a neighborlist is used, the coords that input to neighborlist
        # are **not required** to be mapped to the central cell for pbc calculations.
        neighbors = self.neighborlist(self.radial.cutoff, elem_idxs, coords, cell, pbc)
        if self._strategy == "cuaev" or self._strategy == "cuaev-interface":
            return self._cuaev_compute_from_neighbors(elem_idxs, coords, neighbors)
        if self._strategy == "pyaev":
            return self._pyaev_compute_from_neighbors(elem_idxs, coords, neighbors)
        raise RuntimeError(f"Invalid strategy {self._strategy}")

    def compute_from_neighbors(
        self,
        elem_idxs: Tensor,
        coords: Tensor,
        neighbors: Neighbors,
    ) -> Tensor:
        r"""Compute the AEVs from the result of a neighborlist calculation

        Args:
            elem_idxs: |elem_idxs|
            neighbors: |neighbors|

        Returns:
            |aevs|
        """
        if self._strategy == "pyaev":
            return self._pyaev_compute_from_neighbors(elem_idxs, coords, neighbors)
        if self._strategy == "cuaev" or self._strategy == "cuaev-interface":
            return self._cuaev_compute_from_neighbors(elem_idxs, coords, neighbors)
        if self._strategy == "cuaev-fused":
            raise RuntimeError("cuAEV-fused doesn't support `compute_from_neighbors`")
        raise RuntimeError(f"Invalid strategy {self._strategy}")

    def _pyaev_compute_from_neighbors(
        self, elem_idxs: Tensor, coords: Tensor, neighbors: Neighbors
    ) -> Tensor:
        if self._print_aev_branch:
            print("Executing branch: pyAEV")
        terms = self.radial(neighbors.distances)  # (pairs, rad)
        # (molecs, atoms, species * rad)
        radial_aev = self._collect_radial(elem_idxs, neighbors.indices, terms)

        # Discard neighbors outside the (smaller) angular cutoff to improve performance
        neighbors = discard_outside_cutoff(neighbors, self.angular.cutoff)
        triples = neighbors_to_triples(neighbors)

        terms = self.angular(triples.distances, triples.diff_vectors)  # (triples, ang)
        # (molecs, atoms, species-pairs * ang)
        angular_aev = self._collect_angular(
            elem_idxs,
            neighbors.indices,
            triples.central_idxs,
            triples.side_idxs,
            triples.diff_signs,
            terms,
        )
        # Shape (molecs, atoms, num-species-pairs * ang + num-species * rad)
        return torch.cat([radial_aev, angular_aev], dim=-1)

    # Input shapes: (triples,), (2, triples), (2, triples), (2, ang)
    # Output shape: (molecs, atoms, num-species-pairs * ang)
    def _collect_angular(
        self,
        elem_idxs: Tensor,
        neighbor_idxs: Tensor,
        central_idx: Tensor,
        side_idxs: Tensor,
        sign12: Tensor,
        terms: Tensor,
    ) -> Tensor:
        num_molecs, num_atoms = elem_idxs.shape
        neighbor_elem_idxs = elem_idxs.view(-1)[neighbor_idxs]  # shape (2, pairs)

        # shape (2, 2, T)
        species12_small = neighbor_elem_idxs[:, side_idxs]
        # shape (2, T)
        triple_element_side_idxs = torch.where(
            sign12 == 1,
            species12_small[1],
            species12_small[0],
        )
        # shape (CxAxSp, Z)
        angular_aev = terms.new_zeros(
            (num_molecs * num_atoms * self.num_species_pairs, self.angular.num_feats)
        )
        # shape (T,)
        # NOTE: Casting is necessary in C++ due to a LibTorch bug
        index = central_idx * self.num_species_pairs + self.triu_index[
            triple_element_side_idxs[0], triple_element_side_idxs[1]
        ].to(torch.long)
        angular_aev.index_add_(0, index, terms)
        # shape (C, A, SpxZ)
        return angular_aev.reshape(num_molecs, num_atoms, self.angular_len)

    # Input shapes: (molecs, atoms), (2, pairs), (pairs, rad)
    # Output shape (molecs, atoms, num_species * rad-feat)
    def _collect_radial(
        self, elem_idxs: Tensor, neighbor_idxs: Tensor, terms: Tensor
    ) -> Tensor:
        num_molecs, num_atoms = elem_idxs.shape
        neighbor_elem_idxs = elem_idxs.view(-1)[neighbor_idxs]  # shape (2, pairs)
        # shape (CxAxS, R)
        radial_aev = terms.new_zeros(
            (num_molecs * num_atoms * self.num_species, self.radial.num_feats)
        )
        # shape (2, P)
        index12 = neighbor_idxs * self.num_species + neighbor_elem_idxs.flip(0)
        radial_aev.index_add_(0, index12[0], terms)
        radial_aev.index_add_(0, index12[1], terms)
        return radial_aev.reshape(num_molecs, num_atoms, self.radial_len)

    @jit_unused_if_no_cuaev()
    def _register_cuaev_computer(self) -> None:
        # cuaev_computer is needed only when using a cuAEV compute strategy. However jit
        # always needs to register the CuaevComputer type This 'registration' is only a
        # kind of "dummy" initialization, it is always necessary to reinitialize in
        # forward at least once, since some tensors may be on CPU at this point, but on
        # GPU when forward is called.
        empty = torch.empty(0)
        self.cuaev_computer = torch.classes.cuaev.CuaevComputer(
            0.0, 0.0, empty, empty, empty, empty, empty, empty, 1, True
        )

    @jit_unused_if_no_cuaev()
    def _init_cuaev_computer(self) -> None:
        # If we reach this part of the code then the radial and
        # angular terms must be Standard*, so these tensors will exist
        self.cuaev_computer = torch.classes.cuaev.CuaevComputer(
            self.radial.cutoff,
            self.angular.cutoff,
            self.radial.eta,
            self.radial.shifts,
            self.angular.eta,
            self.angular.zeta,
            self.angular.shifts,
            self.angular.sections,
            self.num_species,
            (self._cuaev_cutoff_fn == "cosine"),
        )
        self._cuaev_computer_is_init = True

    @jit_unused_if_no_cuaev()
    def _cuaev_fused(self, species: Tensor, coordinates: Tensor) -> Tensor:
        self._prepare_cuaev_execution(species, "fused")
        return torch.ops.cuaev.run(
            coordinates, species.to(torch.int32), self.cuaev_computer
        )

    @jit_unused_if_no_cuaev()
    def _cuaev_compute_from_neighbors(
        self,
        species: Tensor,
        coordinates: Tensor,
        neighbors: Neighbors,
    ) -> Tensor:
        self._prepare_cuaev_execution(species, "half-neighborlist")
        # The coordinates will not be used in forward calculation, but it's
        # gradient (force) will still be calculated in cuaev kernel, so it's
        # important to have coordinates passed as an argument.
        return torch.ops.cuaev.run_with_half_nbrlist(
            coordinates,
            species.to(torch.int32),
            neighbors.indices.to(torch.int32),
            neighbors.diff_vectors,
            neighbors.distances,
            self.cuaev_computer,
        )

    # NOTE: This function is for testing purposes or for usage with the LAMMPS interface
    # Computing aev with full nbrlist that is from
    # 1. Lammps interface
    # 2. For testing purpose, the full nbrlist converted from half nbrlist
    # The full neighbor list format needs the following three tensors:
    # - `ilist_unique`: This is a 1D tensor containing all local atom indices.
    # - `jlist`: A 1D tensor containing all the neighbors for all atoms.
    #    The neighbors for atom `i` can be inferred from the numneigh tensor
    # - `numneigh`: This is a 1D tensor that specifies the number of
    # neighbors for each atom i.
    @jit_unused_if_no_cuaev()
    def _compute_cuaev_with_full_nbrlist(
        self,
        species: Tensor,
        coordinates: Tensor,
        ilist_unique: Tensor,
        jlist: Tensor,
        numneigh: Tensor,
    ) -> Tensor:
        self._prepare_cuaev_execution(species, "full-neighborlist")
        if coordinates.shape[0] != 1:
            raise ValueError("cuAEV with full neighborlist doesn't support batches")
        return torch.ops.cuaev.run_with_full_nbrlist(
            coordinates,
            species.to(torch.int32),
            ilist_unique.to(torch.int32),
            jlist.to(torch.int32),
            numneigh.to(torch.int32),
            self.cuaev_computer,
        )

    @jit_unused_if_no_cuaev()
    def _prepare_cuaev_execution(self, species: Tensor, branch: str) -> None:
        if self._print_aev_branch:
            print(f"Executing branch: cuAEV {branch}")
        if species.device.type != "cuda" and species.shape[1] != 0:
            raise ValueError(
                "cuAEV requires inputs in a CUDA device if there is at least 1 atom"
            )
        if not self._cuaev_computer_is_init:
            self._init_cuaev_computer()

    # Converet half nbrlist to full nbrlist.
    @jit_unused_if_no_cuaev()
    @staticmethod
    def _half_to_full_nbrlist(atom_index12: Tensor) -> tp.Tuple[Tensor, Tensor, Tensor]:
        ilist = atom_index12.view(-1)
        jlist = atom_index12.flip(0).view(-1)
        ilist_sorted, indices = ilist.sort(stable=True)
        jlist = jlist[indices]
        ilist_unique, numneigh = torch.unique_consecutive(
            ilist_sorted, return_counts=True
        )
        return ilist_unique, jlist, numneigh

    @jit_unused_if_no_cuaev()
    @staticmethod
    def _full_to_half_nbrlist(
        ilist_unique: Tensor,
        jlist: Tensor,
        numneigh: Tensor,
        species: Tensor,
        fullnbr_diff_vector: Tensor,
    ) -> tp.Tuple[Tensor, Tensor, Tensor]:
        # NOTE: This function has the following limitations: itonly works for
        # lammps-type pbc neighborlists (with local and ghost atoms). TorchANI
        # neighborlists only have 1 set of atoms and do mapping with local and image
        # atoms, which will not work here.
        ilist_unique = ilist_unique.long()
        jlist = jlist.long()
        ilist = torch.repeat_interleave(ilist_unique, numneigh)
        atom_index12 = torch.cat(
            [ilist.unsqueeze(0), jlist.unsqueeze(0)], 0
        )  # [2, num_pairs]

        # sort by atom i
        sort_indices = atom_index12[0].sort().indices
        atom_index12 = atom_index12[:, sort_indices]
        diff_vector = fullnbr_diff_vector[sort_indices]

        # select half nbr by choose atom i < atom j
        half_mask = atom_index12[0] < atom_index12[1]
        atom_index12 = atom_index12[:, half_mask]
        diff_vector = diff_vector[half_mask]

        distances = diff_vector.norm(2, -1)
        return atom_index12, diff_vector, distances

    # Constructors:
    @classmethod
    def like_1x(
        cls,
        num_species: int = 4,
        strategy: str = "pyaev",
        cutoff_fn: CutoffArg = "cosine",
        neighborlist: NeighborlistArg = "all_pairs",
        # Radial args
        radial_start: float = 0.9,
        radial_cutoff: float = 5.2,
        radial_eta: float = 16.0,
        radial_num_shifts: int = 16,
        # Angular args
        angular_start: float = 0.9,
        angular_cutoff: float = 3.5,
        angular_eta: float = 8.0,
        angular_zeta: float = 32.0,
        angular_num_shifts: int = 4,
        angular_num_sections: int = 8,
    ) -> tpx.Self:
        r"""Build an AEVComputer with standard radial and angular terms

        Uses the same defaults as those in the `torchani.models.ANI1x` model.

        Args:
            cutoff_fn: The cutoff function used for the calculation.
            neighborlist: The neighborlist usied for the calculation.
        Returns:
            The constructed `AEVComputer`, ready for use.
        """
        return cls(
            radial=ANIRadial.cover_linearly(
                start=radial_start,
                cutoff=radial_cutoff,
                eta=radial_eta,
                num_shifts=radial_num_shifts,
                cutoff_fn=cutoff_fn,
            ),
            angular=ANIAngular.cover_linearly(
                start=angular_start,
                cutoff=angular_cutoff,
                eta=angular_eta,
                zeta=angular_zeta,
                num_shifts=angular_num_shifts,
                num_sections=angular_num_sections,
                cutoff_fn=cutoff_fn,
            ),
            num_species=num_species,
            strategy=strategy,
            neighborlist=neighborlist,
        )

    @classmethod
    def like_2x(
        cls,
        num_species: int = 7,
        strategy: str = "pyaev",
        cutoff_fn: CutoffArg = "cosine",
        neighborlist: NeighborlistArg = "all_pairs",
        # Radial args
        radial_start: float = 0.8,
        radial_cutoff: float = 5.1,
        radial_eta: float = 19.7,
        radial_num_shifts: int = 16,
        # Angular args
        angular_start: float = 0.8,
        angular_cutoff: float = 3.5,
        angular_eta: float = 12.5,
        angular_zeta: float = 14.1,
        angular_num_shifts: int = 8,
        angular_num_sections: int = 4,
    ) -> tpx.Self:
        r"""Build an AEVComputer with standard radial and angular terms

        Uses the same defaults as those in the `torchani.models.ANI2x` model.

        Args:
            cutoff_fn: The cutoff function used for the calculation.
            neighborlist: The neighborlist usied for the calculation.
        Returns:
            The constructed `AEVComputer`, ready for use.
        """
        return cls(
            radial=ANIRadial.cover_linearly(
                start=radial_start,
                cutoff=radial_cutoff,
                eta=radial_eta,
                num_shifts=radial_num_shifts,
                cutoff_fn=cutoff_fn,
            ),
            angular=ANIAngular.cover_linearly(
                start=angular_start,
                cutoff=angular_cutoff,
                eta=angular_eta,
                zeta=angular_zeta,
                num_shifts=angular_num_shifts,
                num_sections=angular_num_sections,
                cutoff_fn=cutoff_fn,
            ),
            num_species=num_species,
            strategy=strategy,
            neighborlist=neighborlist,
        )

    # Legacy API
    @classmethod
    def from_constants(
        cls,
        radial_cutoff: float,
        angular_cutoff: float,
        radial_eta: float,
        radial_shifts: tp.Sequence[float],
        angular_eta: float,
        angular_zeta: float,
        angular_shifts: tp.Sequence[float],
        sections: tp.Sequence[float],
        num_species: int,
        strategy: str = "pyaev",
        cutoff_fn: CutoffArg = "cosine",
        neighborlist: NeighborlistArg = "all_pairs",
    ) -> tpx.Self:
        r"""Build an AEVComputer with standard radial and angular terms, from constants

        For more detail consult the equations in the original `ANI article`_.

        Note:
            This constructor is not recommended, it is kept for backward compatibility.
            Consider using either the primary `AEVComputer` constructor,
            `AEVComputer.like_1x`, or `AEVComputer.like_2x` instead.

        Args:
            radial_cutoff: :math:`R_C` in eq. (2) when used at eq. (3)
            angular_cutoff: :math:`R_C` in eq. (2) when used at eq. (4)
            radial_eta: The 1D tensor of :math:`\eta` in eq. (3)
            radial_shifts: The 1D tensor of :math:`R_s` in eq. (3)
            angluar_eta: The 1D tensor of :math:`\eta` in eq. (4)
            angular_zeta: The 1D tensor of :math:`\zeta` in eq. (4)
            angular_shifts: The 1D tensor of :math:`R_s` in eq. (4)
            sections: The 1D tensor of :math:`\theta_s` in eq. (4)
            num_species: Number of supported atom types.
            strategy: Compute strategy to use.
            cutoff_fn: The cutoff function used for the calculation.
            neighborlist: The neighborlist usied for the calculation.

        Returns:
            The constructed `AEVComputer`, ready for use.

        .. _ANI article:
            http://pubs.rsc.org/en/Content/ArticleLanding/2017/SC/C6SC05720A#!divAbstract
        """
        return cls(
            radial=ANIRadial(
                radial_eta,
                radial_shifts,
                radial_cutoff,
                cutoff_fn=cutoff_fn,
            ),
            angular=ANIAngular(
                angular_eta,
                angular_zeta,
                angular_shifts,
                sections,
                angular_cutoff,
                cutoff_fn=cutoff_fn,
            ),
            num_species=num_species,
            strategy=strategy,
            neighborlist=neighborlist,
        )

    # Needed for bw compatibility
    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs) -> None:
        for oldk in list(state_dict.keys()):
            k = oldk.replace("radial_terms", "radial").replace(
                "angular_terms", "angular"
            )
            state_dict[k] = state_dict.pop(oldk)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
