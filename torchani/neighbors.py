import typing as tp
import math

import torch
from torch import Tensor
from torch.jit import Final

from torchani.utils import map_to_central, cumsum_from_zero
from torchani.tuples import NeighborData


def rescreen(
    cutoff: float,
    neighbors: NeighborData,
) -> NeighborData:
    closer_indices = (neighbors.distances <= cutoff).nonzero().flatten()
    return NeighborData(
        indices=neighbors.indices.index_select(1, closer_indices),
        distances=neighbors.distances.index_select(0, closer_indices),
        diff_vectors=neighbors.diff_vectors.index_select(0, closer_indices),
    )


class Neighborlist(torch.nn.Module):
    default_pbc: Tensor
    default_cell: Tensor

    def __init__(self):
        """Compute pairs of atoms that are neighbors, uses pbc depending on
        weather pbc.any() is True or not

        Arguments:
            coordinates (:class:`torch.Tensor`): tensor of shape
                (molecules, atoms, 3) for atom coordinates.
        """
        super().__init__()
        self.register_buffer(
            "default_cell", torch.eye(3, dtype=torch.float), persistent=False
        )
        self.register_buffer(
            "default_pbc", torch.zeros(3, dtype=torch.bool), persistent=False
        )
        self.diff_vectors = torch.empty(0)

    @torch.jit.export
    def _compute_bounding_cell(
        self, coordinates: Tensor, eps: float
    ) -> tp.Tuple[Tensor, Tensor]:
        # this works but its not needed for this naive implementation
        # This should return a bounding cell
        # for the molecule, in all cases, also it displaces coordinates a fixed
        # value, so that they fit inside the cell completely. This should have
        # no effects on forces or energies

        # add an epsilon to pad due to floating point precision
        min_ = torch.min(coordinates.view(-1, 3), dim=0).values - eps
        max_ = torch.max(coordinates.view(-1, 3), dim=0).values + eps
        largest_dist = max_ - min_
        coordinates = coordinates - min_
        cell = self.default_cell * largest_dist
        assert (coordinates > 0.0).all()
        assert (coordinates < torch.norm(cell, dim=1)).all()
        return coordinates, cell

    def _screen_with_cutoff(
        self,
        cutoff: float,
        coordinates: Tensor,
        input_neighbor_indices: Tensor,
        shift_values: tp.Optional[Tensor] = None,
        mask: tp.Optional[Tensor] = None,
    ) -> NeighborData:
        # passing an infinite cutoff will only work for non pbc conditions
        # (shift values must be None)
        #
        # Screen a given neighborlist using a cutoff and return a neighborlist with
        # atoms that are within that cutoff, for all molecules in a coordinate set.
        #
        # If the initial coordinates have more than one molecule in the batch
        # dimension then this function expects an input neighborlist that
        # correctly indexes flattened coordinates obtained via
        # coordinates.view(-1, 3).  If the initial coordinates have only one
        # molecule then the output neighborlist will index non flattened
        # coordinates correctly

        # First we check if there are any dummy atoms in species, if there are
        # we get rid of those pairs to prevent wasting resources in calculation
        # of dummy distances
        if mask is not None:
            if mask.any():
                mask = mask.view(-1)[input_neighbor_indices.view(-1)].view(2, -1)
                non_dummy_pairs = (~torch.any(mask, dim=0)).nonzero().flatten()
                input_neighbor_indices = input_neighbor_indices.index_select(
                    1, non_dummy_pairs
                )
                # shift_values can be None when there are no pbc conditions to prevent
                # torch from launching kernels with only zeros
                if shift_values is not None:
                    shift_values = shift_values.index_select(0, non_dummy_pairs)

        coordinates = coordinates.view(-1, 3)
        # Difference vector and distances could be obtained for free when
        # screening, unfortunately distances have to be recalculated twice each
        # time they are screened, since otherwise torch prepares to calculate
        # derivatives of multiple distances that will later be disregarded
        if cutoff != math.inf:
            coordinates_ = coordinates.detach()
            # detached calculation #
            coords0 = coordinates_.index_select(0, input_neighbor_indices[0])
            coords1 = coordinates_.index_select(0, input_neighbor_indices[1])
            diff_vectors = coords0 - coords1
            if shift_values is not None:
                diff_vectors += shift_values
            distances = diff_vectors.norm(2, -1)
            in_cutoff = (distances <= cutoff).nonzero().flatten()
            # ------------------- #

            screened_neighbor_indices = input_neighbor_indices.index_select(
                1, in_cutoff
            )
            if shift_values is not None:
                shift_values = shift_values.index_select(0, in_cutoff)
        else:
            assert (
                shift_values is None
            ), "PBC can't be implemented with an infinite cutoff"
            screened_neighbor_indices = input_neighbor_indices

        coords0 = coordinates.index_select(0, screened_neighbor_indices[0])
        coords1 = coordinates.index_select(0, screened_neighbor_indices[1])
        screened_diff_vectors = coords0 - coords1
        if shift_values is not None:
            screened_diff_vectors += shift_values

        # This is the very first `diff_vectors` that are used to calculate
        # various potentials: 2-body (radial), 3-body (angular), repulsion,
        # dispersion and etc. To enable stress calculation using partial_fdotr
        # approach, `diff_vectors` requires the `requires_grad` flag to be set
        # and needs to be saved for future differentiation.
        screened_diff_vectors.requires_grad_()
        self.diff_vectors = screened_diff_vectors

        screened_distances = screened_diff_vectors.norm(2, -1)
        return NeighborData(
            indices=screened_neighbor_indices,
            distances=screened_distances,
            diff_vectors=screened_diff_vectors,
        )

    def get_diff_vectors(self):
        return self.diff_vectors

    def dummy(self) -> NeighborData:
        # return dummy neighbor data
        device = self.default_cell.device
        dtype = self.default_cell.dtype
        indices = torch.tensor([[0], [1]], dtype=torch.long, device=device)
        distances = torch.tensor([1.0], dtype=dtype, device=device)
        diff_vectors = torch.tensor([[1.0, 0.0, 0.0]], dtype=dtype, device=device)
        return NeighborData(
            indices=indices,
            distances=distances,
            diff_vectors=diff_vectors,
        )

    @torch.jit.export
    def _recast_long_buffers(self) -> None:
        pass


class FullPairwise(Neighborlist):
    default_shift_values: Tensor

    def __init__(self):
        """Compute pairs of atoms that are neighbors, uses pbc depending on
        weather pbc.any() is True or not
        """
        super().__init__()
        self.register_buffer(
            "default_shift_values", torch.tensor(0.0), persistent=False
        )

    def forward(
        self,
        species: Tensor,
        coordinates: Tensor,
        cutoff: float,
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
    ) -> NeighborData:
        """
        Arguments:
            coordinates (:class:`torch.Tensor`): tensor of shape
                (molecules, atoms, 3) for atom coordinates.
            cell (:class:`torch.Tensor`): tensor of shape (3, 3) of the three vectors
                defining unit cell: tensor([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]])
            cutoff (float): the cutoff inside which atoms are considered pairs
            pbc (:class:`torch.Tensor`): boolean tensor of shape (3,) storing
            wheather pbc is required
        """
        assert (cell is not None and pbc is not None) or (cell is None and pbc is None)
        cell = cell if cell is not None else self.default_cell
        pbc = pbc if pbc is not None else self.default_pbc

        mask = species == -1
        if pbc.any():
            atom_index12, shift_indices = self._full_pairwise_pbc(
                species, cutoff, cell, pbc
            )
            shift_values = shift_indices.to(cell.dtype) @ cell
            # before being screened the coordinates have to be mapped to the
            # central cell in case they are not inside it, this is not necessary
            # if there is no pbc
            coordinates = map_to_central(coordinates, cell, pbc)
            return self._screen_with_cutoff(
                cutoff, coordinates, atom_index12, shift_values, mask
            )
        else:
            num_molecules = species.shape[0]
            num_atoms = species.shape[1]
            # Create a pairwise neighborlist for all molecules and all atoms,
            # assuming that there are no atoms at all. Dummy species will be
            # screened later
            atom_index12 = torch.triu_indices(
                num_atoms, num_atoms, 1, device=species.device
            )
            if num_molecules > 1:
                atom_index12 = atom_index12.unsqueeze(1).repeat(1, num_molecules, 1)
                atom_index12 += num_atoms * torch.arange(
                    num_molecules, device=mask.device
                ).view(1, -1, 1)
                atom_index12 = atom_index12.view(-1).view(2, -1)
            return self._screen_with_cutoff(
                cutoff, coordinates, atom_index12, shift_values=None, mask=mask
            )

    def _full_pairwise_pbc(
        self,
        species: Tensor,
        cutoff: float,
        cell: Tensor,
        pbc: Tensor,
    ) -> tp.Tuple[Tensor, Tensor]:
        cell = cell.detach()
        shifts = self._compute_shifts(cutoff, cell, pbc)
        num_atoms = species.shape[1]
        all_atoms = torch.arange(num_atoms, device=cell.device)

        # Step 2: center cell
        p12_center = torch.triu_indices(num_atoms, num_atoms, 1, device=cell.device)
        shifts_center = shifts.new_zeros((p12_center.shape[1], 3))

        # Step 3: cells with shifts
        # shape convention (shift index, molecule index, atom index, 3)
        num_shifts = shifts.shape[0]
        all_shifts = torch.arange(num_shifts, device=cell.device)
        prod = torch.cartesian_prod(all_shifts, all_atoms, all_atoms).t()
        shift_index = prod[0]
        p12 = prod[1:]
        shifts_outside = shifts.index_select(0, shift_index)

        # Step 4: combine results for all cells
        shifts_all = torch.cat([shifts_center, shifts_outside])
        all_atom_pairs = torch.cat([p12_center, p12], dim=1)
        return all_atom_pairs, shifts_all

    def _compute_shifts(self, cutoff: float, cell: Tensor, pbc: Tensor) -> Tensor:
        """Compute the shifts of unit cell along the given cell vectors to make it
        large enough to contain all pairs of neighbor atoms with PBC under
        consideration

        Arguments:
            cell (:class:`torch.Tensor`): tensor of shape (3, 3) of the three
            vectors defining unit cell:
                tensor([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]])
            pbc (:class:`torch.Tensor`): boolean vector of size 3 storing
                if pbc is enabled for that direction.

        Returns:
            :class:`torch.Tensor`: long tensor of shifts. the center cell and
                symmetric cells are not included.
        """
        reciprocal_cell = cell.inverse().t()
        inv_distances = reciprocal_cell.norm(2, -1)
        num_repeats = torch.ceil(cutoff * inv_distances).to(torch.long)
        num_repeats = torch.where(pbc, num_repeats, num_repeats.new_zeros(()))
        r1 = torch.arange(1, num_repeats[0].item() + 1, device=cell.device)
        r2 = torch.arange(1, num_repeats[1].item() + 1, device=cell.device)
        r3 = torch.arange(1, num_repeats[2].item() + 1, device=cell.device)
        o = torch.zeros(1, dtype=torch.long, device=cell.device)
        return torch.cat(
            [
                torch.cartesian_prod(r1, r2, r3),
                torch.cartesian_prod(r1, r2, o),
                torch.cartesian_prod(r1, r2, -r3),
                torch.cartesian_prod(r1, o, r3),
                torch.cartesian_prod(r1, o, o),
                torch.cartesian_prod(r1, o, -r3),
                torch.cartesian_prod(r1, -r2, r3),
                torch.cartesian_prod(r1, -r2, o),
                torch.cartesian_prod(r1, -r2, -r3),
                torch.cartesian_prod(o, r2, r3),
                torch.cartesian_prod(o, r2, o),
                torch.cartesian_prod(o, r2, -r3),
                torch.cartesian_prod(o, o, r3),
            ]
        )


class CellList(Neighborlist):
    verlet: Final[bool]
    constant_volume: Final[bool]

    grid_numel: int
    skin: float
    cell_diagonal: Tensor
    grid_shape: Tensor
    vector_idx_to_flat: Tensor
    translation_cases: Tensor
    vector_idx_displacement: Tensor
    translation_displacement_indices: Tensor
    bucket_length_lower_bound: Tensor
    spherical_factor: Tensor

    def __init__(
        self,
        buckets_per_cutoff: int = 1,
        verlet: bool = False,
        skin: tp.Optional[float] = None,
        constant_volume: bool = False,
    ):
        super().__init__()

        # right now I will only support this, and the extra neighbors are
        # hardcoded, but full support for arbitrary buckets per cutoff is possible
        assert (
            buckets_per_cutoff == 1
        ), "Cell list currently only supports one bucket per cutoff"
        assert not verlet, "Verlet cell list has issues and should not be used"
        self.constant_volume = constant_volume
        self.verlet = verlet
        self.grid_numel: int = 0
        self.register_buffer(
            "spherical_factor", torch.full(size=(3,), fill_value=1.0), persistent=False
        )
        self.register_buffer("cell_diagonal", torch.zeros(1), persistent=False)
        self.register_buffer(
            "grid_shape", torch.zeros(3, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "vector_idx_to_flat", torch.zeros(1, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "translation_cases", torch.zeros(1, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "vector_idx_displacement",
            torch.zeros(1, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "translation_displacement_indices",
            torch.zeros(1, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "bucket_length_lower_bound", torch.zeros(1), persistent=False
        )
        if skin is None:
            self.skin = 1.0 if verlet else 0.0
        else:
            self.skin = skin

        # only used for verlet option
        self.register_buffer("old_cell_diagonal", torch.zeros(1), persistent=False)
        self.register_buffer(
            "old_shift_indices", torch.zeros(1, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "old_atom_pairs", torch.zeros(1, dtype=torch.long), persistent=False
        )
        self.register_buffer("old_coordinates", torch.zeros(1), persistent=False)

        # buckets_per_cutoff is also the number of buckets that is scanned in
        # each direction. It determines how fine grained the grid is, with
        # respect to the cutoff. This is 2 for amber, but 1 is useful for debug
        self.buckets_per_cutoff = buckets_per_cutoff
        # Here I get the vector index displacements for the neighbors of an
        # arbitrary vector index I think these are enough (this is different
        # from pmemd)
        # I choose all the displacements except for the zero
        # displacement that does nothing, which is the last one
        # hand written order ("right-to-left, top-to-bottom")
        # The selected grid elements in the planes are:
        # ("x" selected elements, "-" non-selected and "o" reference element)
        # top,   same,  bottom,
        # |---|  |---|  |xxx|
        # |---|  |xo-|  |xxx|
        # |---|  |xxx|  |xxx|

        # NOTE: "0" corresponds to [0, 0, 0], but I don't really need that for
        # vector indices only for translation displacements
        grid_idx3_offsets = [
            # Neighbor grid elements in the same plane (gz = 0)
            [-1, 0, 0],  # 1
            [-1, -1, 0],  # 2
            [0, -1, 0],  # 3
            [1, -1, 0],  # 4
            # Neighbor grid elements in bottom plane (gz = -1)
            [-1, 1, -1],  # 5
            [0, 1, -1],  # 6
            [1, 1, -1],  # 7
            [-1, 0, -1],  # 8
            [0, 0, -1],  # 9
            [1, 0, -1],  # 10
            [-1, -1, -1],  # 11
            [0, -1, -1],  # 12
            [1, -1, -1],  # 13
        ]
        self.vector_idx_displacement = torch.tensor(grid_idx3_offsets, dtype=torch.long)
        # These are the translation displacement indices, used to displace the
        # image atoms
        assert self.vector_idx_displacement.shape == (13, 3)

        # The translation displacements need all possible displacements in the
        # same plane, so I add the missing ones here (these ones don't exist
        # inside the grid elements)
        grid_idx3_offsets.insert(0, [0, 0, 0])  # 0
        grid_idx3_offsets.extend(
            [
                [-1, 1, 0],  # 14
                [0, 1, 0],  # 15
                [1, 1, 0],  # 16
                [1, 0, 0],  # 17
            ],
        )
        self.translation_displacement_indices = torch.tensor(
            grid_idx3_offsets, dtype=torch.long
        )
        assert self.translation_displacement_indices.shape == (18, 3)
        # This is 26 for 2 buckets and 18 for 1 bucket
        # This is necessary for the image - atom map and atom - image map
        self.num_neighbors = len(self.vector_idx_displacement)

        # variables are not set until we have received a cell at least once
        self.last_cutoff = -1.0
        self.cell_variables_are_set = False
        self.old_values_are_cached = False

    @torch.jit.export
    def _recast_long_buffers(self) -> None:
        # for cell list
        self.grid_shape = self.grid_shape.to(dtype=torch.long)
        self.vector_idx_to_flat = self.vector_idx_to_flat.to(dtype=torch.long)
        self.translation_cases = self.translation_cases.to(dtype=torch.long)
        self.vector_idx_displacement = self.vector_idx_displacement.to(dtype=torch.long)
        self.translation_displacement_indices = (
            self.translation_displacement_indices.to(dtype=torch.long)
        )

    def forward(
        self,
        species: Tensor,
        coordinates: Tensor,
        cutoff: float,
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
    ) -> NeighborData:
        assert cutoff >= 0.0, "Cutoff must be a positive float"
        assert coordinates.shape[0] == 1, "Cell list doesn't support batches"
        if cell is None:
            assert pbc is None or not pbc.any()
        # if cell is None then a bounding cell for the molecule is obtained
        # from the coordinates, in this case the coordinates are assumed to be
        # mapped to the central cell, since anything else would be meaningless
        pbc = pbc if pbc is not None else self.default_pbc
        assert pbc.all() or (not pbc.any()), "CellList supports PBC in all or no dirs"

        if cell is None:
            # Displaced coordinates only used for computation if pbc is not required
            coordinates_displaced, cell = self._compute_bounding_cell(
                coordinates.detach(), eps=1e-3
            )
        else:
            coordinates_displaced = coordinates.detach()

        if (
            (not self.constant_volume)
            or (not self.cell_variables_are_set)
            or (cutoff != self.last_cutoff)
        ):
            # Cell parameters need to be set only once for constant V
            # simulations, and every time for variable V  simulations If the
            # neighborlist cutoff is changed, the variables have to be reset
            # too
            self._setup_variables(cell.detach(), cutoff)

        if (
            self.verlet
            and self.old_values_are_cached
            and (not self._need_new_list(coordinates_displaced.detach()))
        ):
            # If a new cell list is not needed use the old cached values
            # IMPORTANT: here cached values should NOT be updated, moving cache
            # to the new step is incorrect
            atom_pairs = self.old_atom_pairs
            shift_indices: tp.Optional[Tensor] = self.old_shift_indices
        else:
            # The cell list is calculated with a skin here. Since coordinates are
            # fractionalized before cell calculation, it is not needed for them to
            # be imaged to the central cell, they can lie outside the cell.
            atom_pairs, shift_indices = self._calculate_cell_list(
                coordinates_displaced.detach(),
                cell,
                pbc,
            )
            # 'Verlet' prevent unnecessary rebuilds of the neighborlist
            if self.verlet:
                self._cache_values(
                    atom_pairs, shift_indices, coordinates_displaced.detach()
                )

        if pbc.any():
            assert shift_indices is not None
            shift_values = shift_indices.to(cell.dtype) @ cell
            # Before the screening step we map the coordinates to the central cell,
            # same as with a full pairwise calculation
            coordinates = map_to_central(coordinates, cell.detach(), pbc)
            # The final screening does not use the skin, the skin is only used
            # internally to prevent neighborlist recalculation.  We must screen
            # even if the list is not rebuilt, two atoms may have moved a long
            # enough distance that they are not neighbors anymore, but a short
            # enough distance that the neighborlist is not rebuilt. Rebuilds
            # happen only if it can't be guaranteed that the cached
            # neighborlist holds at least all atom pairs, but it may hold more.
            return self._screen_with_cutoff(
                cutoff, coordinates, atom_pairs, shift_values, (species == -1)
            )
        else:
            return self._screen_with_cutoff(
                cutoff, coordinates, atom_pairs, shift_values=None, mask=(species == -1)
            )

    def _calculate_cell_list(
        self,
        coordinates: Tensor,  # shape (C, A, 3)
        cell: Tensor,  # shape (3, 3)
        pbc: Tensor,  # shape (3,)
    ) -> tp.Tuple[Tensor, tp.Optional[Tensor]]:
        # 1) Calculate the location of each atom i the 3D grid that spans the
        # cell. This location is given by a by a grid_idx3 "g3", and by a
        # single flat grid_idx "g"
        # shapes (C, A, 3) and (C, A) for g3[a] and g[a]
        atom_grid_idx3 = coords_to_grid_idx3(coordinates, cell, self.grid_shape)
        atom_grid_idx = flatten_grid_idx3(atom_grid_idx3, self.grid_shape)

        # FIRST WE WANT "WITHIN" IMAGE PAIRS
        # 1) Calculate:
        # - The num of atoms in each grid element c[g]
        # - The max num atoms in any grid element c*
        # - The cumulative num of atoms BEFORE each grid element cc[g]
        grid_count, grid_cumcount = count_atoms_in_grid(atom_grid_idx, self.grid_numel)
        grid_count_max: int = int(grid_count.max())

        # 2) These indices represent pairs WITHIN each element of the central grid
        within_image_pairs = image_pairs_within_grid_elements(
            grid_count, grid_cumcount, grid_count_max
        )

        # NOW WE WANT "BETWEEN" IMAGE PAIRS
        # 1) Get the vector indices of all (pure) neighbors of each atom
        # this gives g3[a, n] and g[a, n]
        # shapes (C, A, N, 3) and (C, A, N)
        neighbor_grid_idx3, neighbor_grid_idx = self._get_neighbor_indices(
            atom_grid_idx3
        )

        # 2) Upper and lower part of the external pairlist this is the
        # correct "unpadded" upper
        # part of the pairlist it repeats each image
        # idx a number of times equal to the number of atoms on the
        # neighborhood of each atom

        # neighborhood count is A{n} (a), the number of atoms on the
        # neighborhood (all the neighbor buckets) of each atom,
        # A{n} (a) has shape 1 x A
        # neighbor_translation_types
        # has the type of shift for T(a, n), atom a,
        # neighbor bucket n
        neighbor_count = grid_count[neighbor_grid_idx]
        neighbor_cumcount = grid_cumcount[neighbor_grid_idx]
        neighbor_translation_types = self._get_neighbor_translation_types(
            neighbor_grid_idx3
        )
        lower, between_pairs_translation_types = self._lower_between_image_pairs(
            neighbor_count,
            neighbor_cumcount,
            neighbor_translation_types,
            grid_count_max,
        )

        # NOTE: watch out, since sorting is not stable this may scramble the atoms
        # in the same box, so that the atidx you get after applying
        # atidx_from_imidx[something] will not be the correct order
        # since what we want is the pairs this is fine, pairs are agnostic to
        # species.
        # shapes (A,) and (A,) for i[a] and a[i]
        atom_to_image, image_to_atom = atom_image_converters(atom_grid_idx)

        neighborhood_count = neighbor_count.sum(-1).squeeze()
        upper = torch.repeat_interleave(atom_to_image, neighborhood_count)
        assert lower.shape == upper.shape
        between_image_pairs = torch.stack((upper, lower), dim=0)

        if not pbc.any():
            # select only the pairs that don't need any translation
            non_pbc_pairs = (between_pairs_translation_types == 0).nonzero().flatten()
            between_image_pairs = between_image_pairs.index_select(1, non_pbc_pairs)
            shift_indices = None
        else:
            between_pairs_shift_indices = (
                self.translation_displacement_indices.index_select(
                    0, between_pairs_translation_types
                )
            )
            assert between_pairs_shift_indices.shape[-1] == 3
            within_pairs_shift_indices = torch.zeros(
                len(within_image_pairs[0]),
                3,
                device=between_pairs_shift_indices.device,
                dtype=torch.long,
            )
            # -1 is necessary to ensure correct shifts
            shift_indices = -torch.cat(
                (between_pairs_shift_indices, within_pairs_shift_indices), dim=0
            )

        # concatenate within and between
        image_pairs = torch.cat((between_image_pairs, within_image_pairs), dim=1)
        atom_pairs = image_to_atom[image_pairs]

        return atom_pairs, shift_indices

    def _setup_variables(self, cell: Tensor, cutoff: float, extra_space: float = 1e-5):
        device = cell.device
        # Get the shape (GX, GY, GZ) of the grid. Some extra space is used as slack
        # (consistent with SANDER neighborlist by default)
        #
        # The spherical factor is different from 1 in the case of nonorthogonal
        # boxes and accounts for the "spherical protrusion", which is related
        # to the fact that the sphere of radius "cutoff" around an atom needs
        # some extra space in nonorthogonal boxes.
        #
        # NOTE: This is not actually the bucket length used in the grid,
        # it is only a lower bound used to calculate the grid size
        spherical_factor = self.spherical_factor
        bucket_length_lower_bound = (
            spherical_factor * cutoff / self.buckets_per_cutoff
        ) + extra_space

        # 1) Update the cell diagonal and translation displacements
        # sizes of each side are given by norm of each basis vector of the unit cell
        self.cell_diagonal = torch.linalg.norm(cell, dim=0)

        # 2) Get max bucket index (Gx, Gy, Gz)
        # which give the size of the grid of buckets that fully covers the
        # whole volume of the unit cell U, given by "cell", and the number of
        # flat buckets (G,) (equal to the total number of buckets, F )
        #
        # Gx, Gy, Gz is 1 + maximum index for vector g. Flat bucket indices are
        # indices for the buckets written in row major order (or equivalently
        # dictionary order), the number G = GX * GY * GZ

        # bucket_length_lower_bound = B, unit cell U_mu = B * 3 - epsilon this
        # means I can cover it with 3 buckets plus some extra space that is
        # less than a bucket, so I just stretch the buckets a little bit. In
        # this particular case grid_shape = (3, 3, 3)
        self.grid_shape = torch.div(
            self.cell_diagonal, bucket_length_lower_bound, rounding_mode="floor"
        ).to(torch.long)

        self.grid_numel = int(self.grid_shape.prod())
        if self.grid_numel == 0:
            raise RuntimeError("Cell is too small to perform pbc calculations")

        # 4) create the vector_index -> flat_index conversion tensor
        # it is not really necessary to perform circular padding,
        # since we can index the array using negative indices!
        vector_idx_to_flat = torch.arange(0, self.grid_numel, device=device)
        vector_idx_to_flat = vector_idx_to_flat.view(
            int(self.grid_shape[0]),
            int(self.grid_shape[1]),
            int(self.grid_shape[2]),
        )
        self.vector_idx_to_flat = self._pad_circular(vector_idx_to_flat)

        # 5) I now create a tensor that when indexed with vector indices
        # gives the shifting case for that atom/neighbor bucket
        self.translation_cases = torch.zeros_like(self.vector_idx_to_flat)
        # now I need to  fill the vector
        # in some smart way
        # this should fill the tensor in a smart way

        self.translation_cases[0, 1:-1, 1:-1] = 1
        self.translation_cases[0, 0, 1:-1] = 2
        self.translation_cases[1:-1, 0, 1:-1] = 3
        self.translation_cases[-1, 0, 1:-1] = 4
        self.translation_cases[0, -1, 0] = 5
        self.translation_cases[1:-1, -1, 0] = 6
        self.translation_cases[-1, -1, 0] = 7
        self.translation_cases[0, 1:-1, 0] = 8
        self.translation_cases[1:-1, 1:-1, 0] = 9
        self.translation_cases[-1, 1:-1, 0] = 10
        self.translation_cases[0, 0, 0] = 11
        self.translation_cases[1:-1, 0, 0] = 12
        self.translation_cases[-1, 0, 0] = 13
        # extra
        self.translation_cases[0, -1, 1:-1] = 14
        self.translation_cases[1:-1, -1, 1:-1] = 15
        self.translation_cases[-1, -1, 1:-1] = 16
        self.translation_cases[-1, 1:-1, 1:-1] = 17

        self.cell_variables_are_set = True

    @staticmethod
    def _pad_circular(x: Tensor) -> Tensor:
        x = x.unsqueeze(0).unsqueeze(0)
        x = torch.nn.functional.pad(x, (1, 1, 1, 1, 1, 1), mode="circular")
        return x.squeeze()

    def _lower_between_image_pairs(
        self,
        neighbor_count: Tensor,
        neighbor_cumcount: Tensor,
        neighbor_translation_types: Tensor,
        grid_count_max: int,
    ) -> tp.Tuple[Tensor, Tensor]:
        # neighbor_translation_types has shape (1 x A x N)
        # 3) now I need the LOWER part
        # this gives, for each atom, for each neighbor bucket, all the
        # unpadded, unshifted atom neighbors
        # this is basically broadcasted to the shape of fna
        # shape is (C, A, N, c*)
        atoms = neighbor_count.shape[1]
        padded_atom_neighbors = torch.arange(
            0, grid_count_max, device=neighbor_count.device
        )
        padded_atom_neighbors = padded_atom_neighbors.view(1, 1, 1, -1)
        # repeat is needed instead of expand here due to += neighbor_cumcount
        padded_atom_neighbors = padded_atom_neighbors.repeat(
            1, atoms, self.num_neighbors, 1
        )

        # repeat the neighbor translation types to account for all neighboring atoms
        # repeat is needed instead of expand due to reshaping later
        neighbor_translation_types = neighbor_translation_types.unsqueeze(-1).repeat(
            1, 1, 1, padded_atom_neighbors.shape[-1]
        )

        # now I need to add A(f' < fna) shift the padded atom neighbors to get
        # image indices I need to check here that the cumcount is correct since
        # it was technically done with imidx so I need to check correctnes of
        # both counting schemes, but first I create the mask to unpad
        # and then I shift to the correct indices
        mask = padded_atom_neighbors < neighbor_count.unsqueeze(-1)
        padded_atom_neighbors.add_(neighbor_cumcount.unsqueeze(-1))
        # the mask should have the same shape as padded_atom_neighbors, and
        # now all that is left is to apply the mask in order to unpad
        assert padded_atom_neighbors.shape == mask.shape
        assert neighbor_translation_types.shape == mask.shape
        # x.view(-1).index_select(0, mask.view(-1).nonzero().view(-1)) is equivalent to:
        # torch.masked_select(x, mask) but FASTER
        # view(-1)...view(-1) is used to avoid reshape (not sure if that is faster)
        lower = padded_atom_neighbors.view(-1).index_select(
            0, mask.view(-1).nonzero().view(-1)
        )
        between_pairs_translation_types = neighbor_translation_types.view(
            -1
        ).index_select(0, mask.view(-1).nonzero().view(-1))
        return lower, between_pairs_translation_types

    def _get_neighbor_indices(self, atom_grid_idx3: Tensor) -> tp.Tuple[Tensor, Tensor]:
        mols, atoms, _ = atom_grid_idx3.shape
        neighbors, _ = self.vector_idx_displacement.shape

        # This is actually pure neighbors, so it doesn't have
        # "the bucket itself"
        # These are
        # - g(a, n),  shape 1 x A x N x 3
        # - f(a, n),  shape 1 x A x N
        # These give, for each atom, the flat index or the vector index of its
        # neighbor buckets (neighbor buckets indexed by n).
        # these vector indices have the information that says whether to shift
        # each pair and what amount to shift it
        neighbor_grid_idx3 = (
            atom_grid_idx3.view(mols, atoms, 1, 3)
            + self.vector_idx_displacement.view(mols, 1, neighbors, 3)
        ) + 1
        neighbor_grid_idx3 = neighbor_grid_idx3.view(-1, 3)

        # NOTE: This is needed instead of unbind due to torchscript bug
        neighbor_grid_idx = self.vector_idx_to_flat[
            neighbor_grid_idx3[:, 0],
            neighbor_grid_idx3[:, 1],
            neighbor_grid_idx3[:, 2],
        ]
        return (
            neighbor_grid_idx3.view(mols, atoms, neighbors, 3),
            neighbor_grid_idx.view(mols, atoms, neighbors),
        )

    def _get_neighbor_translation_types(
        self, neighbor_vector_indices: Tensor
    ) -> Tensor:
        atoms = neighbor_vector_indices.shape[1]
        neighbors = neighbor_vector_indices.shape[2]
        neighbor_vector_indices = neighbor_vector_indices.view(-1, 3)
        neighbor_translation_types = self.translation_cases[
            neighbor_vector_indices[:, 0],
            neighbor_vector_indices[:, 1],
            neighbor_vector_indices[:, 2],
        ]
        neighbor_translation_types = neighbor_translation_types.view(
            1, atoms, neighbors
        )
        return neighbor_translation_types

    def _cache_values(
        self,
        atom_pairs: Tensor,
        shift_indices: tp.Optional[Tensor],
        coordinates: Tensor,
    ):
        self.old_atom_pairs = atom_pairs.detach()
        if shift_indices is not None:
            self.old_shift_indices = shift_indices.detach()
        self.old_coordinates = coordinates.detach()
        self.old_cell_diagonal = self.cell_diagonal.detach()
        self.old_values_are_cached = True

    def reset_cached_values(self) -> None:
        float_dtype = self.cell_diagonal.dtype
        device = self.cell_diagonal.device
        self._cache_values(
            torch.zeros(1, dtype=torch.long, device=device),
            torch.zeros(1, dtype=torch.long, device=device),
            torch.zeros(1, dtype=float_dtype, device=device),
        )
        self.old_values_are_cached = False

    def _need_new_list(self, coordinates: Tensor) -> bool:
        if not self.verlet:
            return True
        # Check if any coordinate exceedes half the skin depth,
        # if a coordinate exceedes this then the cell list has to be rebuilt
        box_scaling = self.cell_diagonal / self.old_cell_diagonal
        delta = coordinates - self.old_coordinates * box_scaling
        dist_squared = delta.pow(2).sum(-1)
        need_new_list = (dist_squared > (self.skin / 2) ** 2).any().item()
        return bool(need_new_list)


def coords_to_grid_idx3(
    coordinates: Tensor,
    cell: Tensor,
    grid_shape: Tensor,
) -> Tensor:
    # Transforms a tensor of coordinates (shape (C, A, 3))
    # into a tensor of grid_idx3 (same shape, (C, A, 3))
    #
    # 1) Fractionalize coordinates. All coordinates will be relative to the
    # cell lengths after this step, which means they lie in the range [0., 1.)
    fractionals = fractionalize_coords(coordinates, cell)  # shape (C, A, 3)
    # 2) assign to each fractional the corresponding grid_idx3
    grid_idx3 = torch.floor(fractionals * grid_shape.view(1, 1, -1)).to(torch.long)
    return grid_idx3


def fractionalize_coords(coordinates: Tensor, cell: Tensor) -> Tensor:
    # Scale coordinates to box size
    #
    # Make all coordinates relative to the box size. This means for
    # instance that if the coordinate is 3.15 times the cell length, it is
    # turned into 3.15; if it is 0.15 times the cell length, it is turned
    # into 0.15, etc
    fractional_coords = torch.matmul(coordinates, cell.inverse())
    # this is done to account for possible coordinates outside the box,
    # which amber does, in order to calculate diffusion coefficients, etc
    fractional_coords -= fractional_coords.floor()
    # fractional_coordinates should be in the range [0, 1.0)
    fractional_coords[fractional_coords >= 1.0] += -1.0
    fractional_coords[fractional_coords < 0.0] += 1.0
    return fractional_coords


def flatten_grid_idx3(grid_idx3: Tensor, grid_shape: Tensor) -> Tensor:
    # Converts a tensor that holds vector bucket indices to one that holds
    # flat bucket indices (last dimension is removed).
    # for row major this is (Gy * Gz, Gz, 1)
    grid_idx3 = grid_idx3.clone()
    grid_idx3[:, :, 0] *= grid_shape[1] * grid_shape[2]
    grid_idx3[:, :, 1] *= grid_shape[2]
    return grid_idx3.sum(-1)


def atom_image_converters(grid_idx: Tensor) -> tp.Tuple[Tensor, Tensor]:
    # this are the "image indices", indices that sort atoms in the order of
    # the flattened bucket index.  Only occupied buckets are considered, so
    # if a bucket is unoccupied the index is not taken into account.  for
    # example if the atoms are distributed as:
    # / 1 9 8 / - / 3 2 4 / 7 /
    # where the bars delimit flat buckets, then the assoc. image indices
    # are:
    # / 0 1 2 / - / 3 4 5 / 6 /
    # atom indices can be reconstructed from the image indices, so the
    # pairlist can be built with image indices and then at the end calling
    # atom_indices_from_image_indices[pairlist] you convert to atom_indices

    # atom_to_image returns tensors that convert image indices into atom
    # indices and viceversa
    # move to device necessary? not sure
    grid_idx = grid_idx.view(-1)  # shape (C, A) -> (A,), get rid of C
    image_to_atom = torch.argsort(grid_idx)
    atom_to_image = torch.argsort(image_to_atom)
    # output shapes are (A,) (A,)
    return atom_to_image, image_to_atom


def count_atoms_in_grid(
    atom_grid_idx: Tensor,  # shape (C, A)
    grid_numel: int,
) -> tp.Tuple[Tensor, Tensor]:
    # NOTE: count in flat bucket: 3 0 0 0 ... 2 0 0 0 ... 1 0 1 0 ...,
    # shape is total grid elements G. grid_cumcount has the number of
    # atoms BEFORE a given bucket cumulative buckets count: 0 3 3 3 ... 3 5
    # 5 5 ... 5 6 6 7 ...
    atom_grid_idx = atom_grid_idx.view(-1)  # shape (A,), get rid of C
    # G = the total number of grid elements
    grid_count = torch.bincount(atom_grid_idx, minlength=grid_numel)  # shape (G,)
    grid_cumcount = cumsum_from_zero(grid_count)  # shape (G,)
    return grid_count, grid_cumcount


NeighborlistArg = tp.Union[
    tp.Literal[
        "full_pairwise",
        "cell_list",
        "verlet_cell_list",
        "base",
    ],
    Neighborlist,
]


def image_pairs_within_grid_elements(
    grid_count: Tensor,  # shape (G,)
    grid_cumcount: Tensor,  # shape (G,)
    grid_count_max: int,
) -> Tensor:
    device = grid_count.device
    # note: in this function wpairs == "with_pairs"
    # max_in_bucket = maximum number of atoms contained in any bucket

    # get all indices g that have pairs inside
    # these are A(w) and Ac(w), and wpairs_flat_index is actually f(w)
    # shapes are (G',) where G' is the number of grid elements with pairs
    grid_wpairs_idx = (grid_count > 1).nonzero().view(-1)
    grid_wpairs_count = grid_count.index_select(0, grid_wpairs_idx)
    grid_wpairs_cumcount = grid_cumcount.index_select(0, grid_wpairs_idx)

    # Get the image_neighbor_idxs "within" for the grid el with the max num atoms
    padded_pairs = torch.triu_indices(  # shape (2, pairs*)
        grid_count_max,
        grid_count_max,
        offset=1,
        device=device,
    )
    # sort along first row
    padded_pairs = padded_pairs.index_select(1, torch.argsort(padded_pairs[1]))
    # shape (2, pairs) + shape (wpairs, 1, 1) = shape (wpairs, 2, pairs)

    # basically this repeats the padded pairs "wpairs" times and adds to
    # all of them the cumulative counts, then we unravel all pairs, which
    # remain in the correct order in the second row (the order within same
    # numbers in the first row is actually not essential)
    padded_pairs = padded_pairs.view(2, 1, -1) + grid_wpairs_cumcount.view(1, -1, 1)
    padded_pairs = padded_pairs.view(2, -1)

    # NOTE this code is very confusing, it could probably use some comments
    # / simplification
    grid_pairs_max = grid_count_max * (grid_count_max - 1) // 2
    wpairs_count_pairs = torch.div(
        grid_wpairs_count * torch.sub(grid_wpairs_count, 1),
        2,
        rounding_mode="floor",
    )

    mask = torch.arange(0, grid_pairs_max, device=device)
    mask = mask.expand(grid_wpairs_idx.numel(), -1)
    mask = (mask < wpairs_count_pairs.view(-1, 1)).view(-1)
    within_image_pairs = padded_pairs.index_select(1, mask.nonzero().squeeze())
    return within_image_pairs


def parse_neighborlist(neighborlist: NeighborlistArg = "base") -> Neighborlist:
    if neighborlist == "full_pairwise":
        neighborlist = FullPairwise()
    elif neighborlist == "cell_list":
        neighborlist = CellList()
    elif neighborlist == "verlet_cell_list":
        neighborlist = CellList(verlet=True)
    elif neighborlist == "base":
        neighborlist = Neighborlist()
    elif not isinstance(neighborlist, Neighborlist):
        raise ValueError(f"Unsupported neighborlist: {neighborlist}")
    return tp.cast(Neighborlist, neighborlist)
