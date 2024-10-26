import os
import typing as tp

import torch
from torch import Tensor
from torch.jit import Final
import typing_extensions as tpx

from torchani.tuples import SpeciesAEV, NeighborData
from torchani.utils import cumsum_from_zero
from torchani.neighbors import parse_neighborlist, NeighborlistArg, FullPairwise
from torchani.cutoffs import CutoffArg
from torchani.aev.terms import (
    parse_angular_term,
    parse_radial_term,
    StandardAngular,
    StandardRadial,
    RadialTermArg,
    AngularTermArg,
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
    num_species: Final[int]
    num_species_pairs: Final[int]

    angular_length: Final[int]
    angular_sublength: Final[int]
    radial_length: Final[int]
    radial_sublength: Final[int]
    aev_length: Final[int]

    triu_index: Tensor
    _compute_strategy: str
    _cuaev_fused_strat_is_avail: bool
    _cuaev_strat_is_avail: bool
    _cuaev_computer_is_init: bool

    def __init__(
        self,
        radial_terms: RadialTermArg,
        angular_terms: AngularTermArg,
        num_species: int,
        compute_strategy: str = "pyaev",
        cutoff_fn: tp.Optional[CutoffArg] = None,
        neighborlist: NeighborlistArg = "full_pairwise",
    ):
        super().__init__()
        self._print_aev_branch = _PRINT_AEV_BRANCH
        self.num_species = num_species
        self.num_species_pairs = num_species * (num_species + 1) // 2
        self.register_buffer("triu_index", self._calculate_triu_index(num_species))

        # Terms
        self.radial_terms = parse_radial_term(radial_terms)
        self.angular_terms = parse_angular_term(angular_terms)
        if not (self.angular_terms.cutoff_fn.is_same(self.radial_terms.cutoff_fn)):
            raise ValueError("Cutoff fn must be the same for angular and radial terms")
        if self.angular_terms.cutoff > self.radial_terms.cutoff:
            raise ValueError(
                f"Angular cutoff {self.angular_terms.cutoff}"
                f" should be smaller than radial cutoff {self.radial_terms.cutoff}"
            )
        self._cuaev_cutoff_fn = self.angular_terms.cutoff_fn._cuaev_name

        # Neighborlist
        self.neighborlist = parse_neighborlist(neighborlist)

        # Lenghts
        self.radial_sublength = self.radial_terms.sublength
        self.angular_sublength = self.angular_terms.sublength
        self.radial_length = self.radial_sublength * self.num_species
        self.angular_length = self.angular_sublength * self.num_species_pairs
        self.aev_length = self.radial_length + self.angular_length

        # The following corresponds to initialization and checks for the cuAEV:

        # Check if the cuaev and the cuaev fused are available for used
        self._cuaev_fused_strat_is_avail = self._check_cuaev_fused_strat_avail()
        self._cuaev_strat_is_avail = self._check_cuaev_strat_avail()

        # cuAEV dummy initialization ('registration') happens here, as long as
        # cuAEV is installed, even if it is not used. This is required by JIT
        if CUAEV_IS_INSTALLED:
            self._register_cuaev_computer()

        # cuAEV init is delayed until fwd, to ensure tensors are in the correct device
        self._cuaev_computer_is_init = False

        # Check that the requested strategy is available
        if compute_strategy == "pyaev":
            pass
        elif compute_strategy == "cuaev":
            self._check_cuaev_strat_avail(raise_exc=True)
        elif compute_strategy == "cuaev-fused":
            self._check_cuaev_fused_strat_avail(raise_exc=True)
        else:
            raise ValueError(f"Unsupported strategy {compute_strategy}")
        self._compute_strategy = compute_strategy

    @property
    def compute_strategy(self) -> str:
        return self._compute_strategy

    @torch.jit.export
    def set_compute_strategy(
        self,
        strat: str = "pyaev",
    ) -> None:
        if strat == "pyaev":
            pass
        elif strat == "cuaev-fused":
            if not self._cuaev_fused_strat_is_avail:
                raise ValueError("cuAEV-fused strategy is not available")
        elif strat == "cuaev":
            if not self._cuaev_strat_is_avail:
                raise ValueError("cuAEV strategy is not available")
        else:
            raise ValueError("Unknown compute strategy")
        self._compute_strategy = strat

    def _check_cuaev_strat_avail(self, raise_exc: bool = False) -> bool:
        if not CUAEV_IS_INSTALLED:
            if raise_exc:
                raise ValueError("cuAEV is not installed")
            return False
        if (
            not self._cuaev_cutoff_fn
            or type(self.angular_terms) is not StandardAngular
            or type(self.radial_terms) is not StandardRadial
        ):
            if raise_exc:
                raise ValueError(
                    "cuAEV only supports StandardAngular and StandardAngular terms, "
                    "and CosineCutoff or SmoothCutoff functions (with default args)"
                )
            return False
        return True

    def _check_cuaev_fused_strat_avail(self, raise_exc: bool = False) -> bool:
        avail = self._check_cuaev_strat_avail(raise_exc=raise_exc)
        if not avail:
            return False
        if not isinstance(self.neighborlist, FullPairwise):
            if raise_exc:
                raise ValueError("cuAEV-fused only supports the default neighborlist")
            return False
        return True

    def extra_repr(self) -> str:
        radial_perc = f"{self.radial_length / self.aev_length * 100:.2f}% of features"
        angular_perc = f"{self.angular_length / self.aev_length * 100:.2f}% of features"
        parts = [
            r"#  " f"aev_length={self.aev_length}",
            r"#  " f"radial_length={self.radial_length} ({radial_perc})",
            r"#  " f"angular_length={self.angular_length} ({angular_perc})",
            f"num_species={self.num_species},",
            f"compute_strategy={self._compute_strategy},",
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
        input_: tp.Tuple[Tensor, Tensor],
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
    ) -> SpeciesAEV:
        """Compute AEVs

        Arguments:
            input_ (tuple): Can be one of the following two cases:

                If you don't care about periodic boundary conditions at all,
                then input can be a tuple of two tensors: species, coordinates.
                species must have shape ``(N, A)``, coordinates must have shape
                ``(N, A, 3)`` where ``N`` is the number of molecules in a batch,
                and ``A`` is the number of atoms.

                .. warning::

                    The species must be indexed in 0, 1, 2, 3, ..., not the element
                    index in periodic table. Check :class:`torchani.SpeciesConverter`
                    if you want periodic table indexing.

                .. note:: The coordinates, and cell are in Angstrom.

                If you want to apply periodic boundary conditions, then the input
                would be a tuple of two tensors (species, coordinates) and two keyword
                arguments `cell=...` , and `pbc=...` where species and coordinates are
                the same as described above, cell is a tensor of shape (3, 3) of the
                three vectors defining unit cell:

                .. code-block:: python

                    tensor([[x1, y1, z1],
                            [x2, y2, z2],
                            [x3, y3, z3]])

                and pbc is boolean vector of size 3 storing if pbc is enabled
                for that direction.

        Returns:
            NamedTuple: Species and AEVs. species are the species from the input
            unchanged, and AEVs is a tensor of shape ``(N, A, self.aev_length)``
        """
        species, coords = input_
        # Check input shape correctness and validate cutoffs
        assert species.dim() == 2
        assert coords.dim() == 3
        assert (species.shape == coords.shape[:2]) and (coords.shape[2] == 3)
        assert self.angular_terms.cutoff < self.radial_terms.cutoff
        # WARNING: If a neighborlist is used, the coords that are input
        # into the neighborlist do **not** need to be mapped into the
        # central cell for pbc calculations.

        cutoff = self.radial_terms.cutoff
        if self._compute_strategy == "pyaev":
            neighbors = self.neighborlist(species, coords, cutoff, cell, pbc)
            aev = self._compute_aev(element_idxs=species, neighbors=neighbors)
        elif self._compute_strategy == "cuaev":
            neighbors = self.neighborlist(species, coords, cutoff, cell, pbc)
            aev = self._compute_cuaev_with_half_nbrlist(species, coords, neighbors)
        elif self._compute_strategy == "cuaev-fused":
            if pbc is not None:
                if pbc.any():
                    raise ValueError("cuAEV-fused doesn't support PBC")
            aev = self._compute_cuaev(species, coords)
        else:
            raise RuntimeError(f"Invalid compute strategy {self._compute_strategy}")
        return SpeciesAEV(species, aev)

    def _compute_aev(
        self,
        element_idxs: Tensor,  # shape (C, A)
        neighbors: NeighborData,
    ) -> Tensor:
        if self._print_aev_branch:
            print("Executing branch: pyAEV")
        neighbor_idxs = neighbors.indices  # shape (2, P)
        distances = neighbors.distances  # shape (P,)
        diff_vectors = neighbors.diff_vectors  # shape (P, 3)
        num_molecules, num_atoms = element_idxs.shape
        # shape (2, P)
        neighbor_element_idxs = element_idxs.view(-1)[neighbor_idxs]
        # shape (P, R)
        terms = self.radial_terms(distances)
        # shape (C, A, SxZ)
        radial_aev = self._collect_radial_terms(
            num_molecules,
            num_atoms,
            neighbor_element_idxs,
            neighbor_idxs=neighbor_idxs,
            terms=terms,
        )

        # Angular cutoff is smaller than radial. Here we discard neighbors
        # outside the angular cutoff to improve performance
        # New shape: (P')
        closer_indices = (distances <= self.angular_terms.cutoff).nonzero().view(-1)
        # new shapes: (2, P') (2, P')  (P', 3)
        neighbor_idxs = neighbor_idxs.index_select(1, closer_indices)
        neighbor_element_idxs = neighbor_element_idxs.index_select(1, closer_indices)
        diff_vectors = diff_vectors.index_select(0, closer_indices)
        distances = distances.index_select(0, closer_indices)

        # shapes: (T,) (2, T) (2, T)
        central_idx, side_idxs, sign12 = self._triple_idxs_from_neighbors(neighbor_idxs)
        # shape (2, T, 3)
        triple_vectors = diff_vectors.index_select(0, side_idxs.view(-1)).view(2, -1, 3)
        triple_vectors = triple_vectors * sign12.view(2, -1, 1)
        triple_distances = distances.index_select(0, side_idxs.view(-1)).view(2, -1)

        # shape (T, Z)
        terms = self.angular_terms(triple_vectors, triple_distances)
        # shape (C, A, SpxZ)
        angular_aev = self._collect_angular_terms(
            num_molecules,
            num_atoms,
            neighbor_element_idxs,
            central_idx,
            side_idxs,
            sign12,
            terms,
        )
        # shape (C, A, SxR + SpxZ)
        return torch.cat([radial_aev, angular_aev], dim=-1)

    def _collect_angular_terms(
        self,
        num_molecules: int,
        num_atoms: int,
        neighbor_element_idxs: Tensor,  # shape (2, P')
        central_idx: Tensor,  # shape (T,)
        side_idxs: Tensor,  # shape (2, T)
        sign12: Tensor,  # shape (2, T)
        terms: Tensor,  # shape (T, Z)
    ) -> Tensor:
        # shape (2, 2, T)
        species12_small = neighbor_element_idxs[:, side_idxs]
        # shape (2, T)
        triple_element_side_idxs = torch.where(
            sign12 == 1,
            species12_small[1],
            species12_small[0],
        )
        # shape (CxAxSp, Z)
        angular_aev = terms.new_zeros(
            (num_molecules * num_atoms * self.num_species_pairs, self.angular_sublength)
        )
        # shape (T,)
        # TODO: This gets cast to double in TorchScript,
        # the issue should be investigated
        index = central_idx * self.num_species_pairs + self.triu_index[
            triple_element_side_idxs[0], triple_element_side_idxs[1]
        ].to(torch.long)
        angular_aev.index_add_(0, index, terms)
        # shape (C, A, SpxZ)
        return angular_aev.reshape(num_molecules, num_atoms, self.angular_length)

    def _collect_radial_terms(
        self,
        num_molecules: int,
        num_atoms: int,
        neighbor_element_idxs: Tensor,  # shape (2, P)
        neighbor_idxs: Tensor,  # shape (2, P)
        terms: Tensor,  # shape (P, R)
    ) -> Tensor:
        # shape (CxAxS, R)
        radial_aev = terms.new_zeros(
            (num_molecules * num_atoms * self.num_species, self.radial_sublength)
        )
        # shape (2, P)
        index12 = neighbor_idxs * self.num_species + neighbor_element_idxs.flip(0)
        radial_aev.index_add_(0, index12[0], terms)
        radial_aev.index_add_(0, index12[1], terms)
        # shape (C, A, SxR)
        return radial_aev.reshape(num_molecules, num_atoms, self.radial_length)

    def _triple_idxs_from_neighbors(
        self, neighbor_idxs: Tensor
    ) -> tp.Tuple[Tensor, Tensor, Tensor]:
        """Input: indices for pairs of atoms that are close to each other.
        each pair only appear once, i.e. only one of the pairs (1, 2) and
        (2, 1) exists.

        Output: indices for all central atoms and it pairs of neighbors. For
        example, if input has pair (0, 1), (0, 2), (0, 3), (0, 4), (1, 2),
        (1, 3), (1, 4), (2, 3), (2, 4), (3, 4), then the output would have
        central atom 0, 1, 2, 3, 4 and for cental atom 0, its pairs of neighbors
        are (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)
        """
        # convert representation from pair to central-others and sort
        sorted_flat_neighbor_idxs, rev_idxs = neighbor_idxs.view(-1).sort()

        # sort compute unique key
        uniqued_central_atom_idx, counts = torch.unique_consecutive(
            sorted_flat_neighbor_idxs, return_inverse=False, return_counts=True
        )

        # compute central_atom_idx
        pair_sizes = (counts * (counts - 1)).div(2, rounding_mode="floor")
        pair_indices = torch.repeat_interleave(pair_sizes)
        central_atom_idx = uniqued_central_atom_idx.index_select(0, pair_indices)

        # do local combinations within unique key, assuming sorted
        m = counts.max().item() if counts.numel() > 0 else 0
        n = pair_sizes.shape[0]
        intra_pair_indices = (
            torch.tril_indices(m, m, -1, device=neighbor_idxs.device)
            .unsqueeze(1)
            .expand(-1, n, -1)
        )
        mask = (
            torch.arange(intra_pair_indices.shape[2], device=neighbor_idxs.device)
            < pair_sizes.unsqueeze(1)
        ).view(-1)
        sorted_local_idx12 = intra_pair_indices.flatten(1, 2)[:, mask]
        sorted_local_idx12 += cumsum_from_zero(counts).index_select(0, pair_indices)

        # unsort result from last part
        local_idx12 = rev_idxs[sorted_local_idx12]

        # compute mapping between representation of central-other to pair
        n = neighbor_idxs.shape[1]
        sign12 = ((local_idx12 < n).to(torch.int8) * 2) - 1
        return central_atom_idx, local_idx12 % n, sign12

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
            self.radial_terms.cutoff,
            self.angular_terms.cutoff,
            self.radial_terms.eta,
            self.radial_terms.shifts,
            self.angular_terms.eta,
            self.angular_terms.zeta,
            self.angular_terms.shifts,
            self.angular_terms.angle_sections,
            self.num_species,
            (self._cuaev_cutoff_fn == "cosine"),
        )
        self._cuaev_computer_is_init = True

    @jit_unused_if_no_cuaev()
    def _compute_cuaev(self, species: Tensor, coordinates: Tensor) -> Tensor:
        self._prepare_cuaev_execution(species, "fused")
        return torch.ops.cuaev.run(
            coordinates, species.to(torch.int32), self.cuaev_computer
        )

    @jit_unused_if_no_cuaev()
    def _compute_cuaev_with_half_nbrlist(
        self,
        species: Tensor,
        coordinates: Tensor,
        neighbors: NeighborData,
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

    @jit_unused_if_no_cuaev()
    def _compute_cuaev_with_full_nbrlist(
        self,
        species: Tensor,
        coordinates: Tensor,
        ilist_unique: Tensor,
        jlist: Tensor,
        numneigh: Tensor,
    ) -> Tensor:
        """
        Computing aev with full nbrlist that is from
            1. Lammps interface
            2. For testting purpose, the full nbrlist converted from half nbrlist

        The full neighbor list format needs the following three tensors:
            - `ilist_unique`: This is a 1D tensor containing all local atom indices.
            - `jlist`: A 1D tensor containing all the neighbors for all atoms.
                  The neighbors for atom `i` could be inferred from the numneigh
                  tensor.
            - `numneigh`: This is a 1D tensor that specifies the number of
                neighbors for each atom i.
        """
        self._prepare_cuaev_execution(species, "full-neighborlist")
        if coordinates.shape[0] != 1:
            raise ValueError("cuAEV with full neighborlist only supports single molecs")
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

    @jit_unused_if_no_cuaev()
    @staticmethod
    def _half_to_full_nbrlist(atom_index12: Tensor) -> tp.Tuple[Tensor, Tensor, Tensor]:
        """
        Convereting half nbrlist to full nbrlist.
        """
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
        """
        Limitations: only works for lammps-type pbc neighborlists (with local
        and ghost atoms). TorchANI neighborlists only have 1 set of atoms and
        do mapping with local and image atoms, which will not work here.
        """
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
    def from_constants(
        cls,
        radial_cutoff: float,
        angular_cutoff: float,
        radial_eta: float,
        radial_shifts: tp.Sequence[float],
        angular_eta: float,
        angular_zeta: float,
        angular_shifts: tp.Sequence[float],
        angle_sections: tp.Sequence[float],
        num_species: int,
        compute_strategy: str = "pyaev",
        cutoff_fn: CutoffArg = "cosine",
        neighborlist: NeighborlistArg = "full_pairwise",
    ) -> tpx.Self:
        r"""Initialize the AEV computer from the following constants:

        Arguments:
            radial_cutoff (float): :math:`R_C` in equation (2) when used at equation (3)
                in the `ANI paper`_.
            Rca (float): :math:`R_C` in equation (2) when used at equation (4)
                in the `ANI paper`_.
            radial_eta (:class:`torch.Tensor`): The 1D tensor of :math:`\eta` in
                equation (3) in the `ANI paper`_.
            radial_shifts (:class:`torch.Tensor`): The 1D tensor of :math:`R_s` in
                equation (3) in the `ANI paper`_.
            angluar_eta (:class:`torch.Tensor`): The 1D tensor of :math:`\eta` in
                equation (4) in the `ANI paper`_.
            angular_zeta (:class:`torch.Tensor`): The 1D tensor of :math:`\zeta` in
                equation (4) in the `ANI paper`_.
            angular_shifts (:class:`torch.Tensor`): The 1D tensor of :math:`R_s` in
                equation (4) in the `ANI paper`_.
            angle_sections (:class:`torch.Tensor`): The 1D tensor of :math:`\theta_s` in
                equation (4) in the `ANI paper`_.
            num_species (int): Number of supported atom types.
            compute_strategy (str): Compute strategy to use, one of 'pyaev', 'cuaev',
                'cuaev-fused'

        .. _ANI paper:
            http://pubs.rsc.org/en/Content/ArticleLanding/2017/SC/C6SC05720A#!divAbstract
        """
        return cls(
            radial_terms=StandardRadial(
                radial_eta,
                radial_shifts,
                radial_cutoff,
                cutoff_fn=cutoff_fn,
            ),
            angular_terms=StandardAngular(
                angular_eta,
                angular_zeta,
                angular_shifts,
                angle_sections,
                angular_cutoff,
                cutoff_fn=cutoff_fn,
            ),
            num_species=num_species,
            compute_strategy=compute_strategy,
            neighborlist=neighborlist,
        )

    @classmethod
    def like_1x(
        cls,
        num_species: int = 4,
        compute_strategy: str = "pyaev",
        cutoff_fn: CutoffArg = "cosine",
        neighborlist: NeighborlistArg = "full_pairwise",
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
        angular_num_angle_sections: int = 8,
    ) -> tpx.Self:
        return cls(
            radial_terms=StandardRadial.cover_linearly(
                start=radial_start,
                cutoff=radial_cutoff,
                eta=radial_eta,
                num_shifts=radial_num_shifts,
                cutoff_fn=cutoff_fn,
            ),
            angular_terms=StandardAngular.cover_linearly(
                start=angular_start,
                cutoff=angular_cutoff,
                eta=angular_eta,
                zeta=angular_zeta,
                num_shifts=angular_num_shifts,
                num_angle_sections=angular_num_angle_sections,
                cutoff_fn=cutoff_fn,
            ),
            num_species=num_species,
            compute_strategy=compute_strategy,
            neighborlist=neighborlist,
        )

    @classmethod
    def like_2x(
        cls,
        num_species: int = 7,
        compute_strategy: str = "pyaev",
        cutoff_fn: CutoffArg = "cosine",
        neighborlist: NeighborlistArg = "full_pairwise",
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
        angular_num_angle_sections: int = 4,
    ) -> tpx.Self:
        return cls(
            radial_terms=StandardRadial.cover_linearly(
                start=radial_start,
                cutoff=radial_cutoff,
                eta=radial_eta,
                num_shifts=radial_num_shifts,
                cutoff_fn=cutoff_fn,
            ),
            angular_terms=StandardAngular.cover_linearly(
                start=angular_start,
                cutoff=angular_cutoff,
                eta=angular_eta,
                zeta=angular_zeta,
                num_shifts=angular_num_shifts,
                num_angle_sections=angular_num_angle_sections,
                cutoff_fn=cutoff_fn,
            ),
            num_species=num_species,
            compute_strategy=compute_strategy,
            neighborlist=neighborlist,
        )
