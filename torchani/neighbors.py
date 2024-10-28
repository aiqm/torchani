r"""
Modular neighborlists to improve scaling for large systems.
"""

import typing as tp
import math

import torch
from torch import Tensor

from torchani.utils import map_to_central, cumsum_from_zero, fast_masked_select
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
    diff_vectors: Tensor

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
    def compute_bounding_cell(
        self,
        coordinates: Tensor,
        eps: float = 1e-3,
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
        mask: Tensor,
        shift_values: tp.Optional[Tensor] = None,
        return_shift_values: bool = False,
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
            if shift_values is not None:
                raise ValueError("PBC can't use an infinite cutoff")
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
            shift_values=shift_values if return_shift_values else None,
        )

    @torch.jit.export
    def process_external_input(
        self,
        species: Tensor,
        coordinates: Tensor,
        neighbor_idxs: Tensor,
        shift_values: Tensor,
        cutoff: float = 0.0,
        input_needs_screening: bool = True,
    ) -> NeighborData:
        # Check shapes
        num_pairs = neighbor_idxs.shape[1]
        assert neighbor_idxs.shape == (2, num_pairs)
        assert shift_values.shape == (num_pairs, 3)

        if input_needs_screening:
            # First screen the input neighbors in case some of the
            # values are at distances larger than the radial cutoff, or some of
            # the values are masked with dummy atoms. The first may happen if
            # the neighborlist uses some sort of skin value to rebuild itself
            # (as in Loup Verlet lists), which is common in MD programs.
            return self._screen_with_cutoff(
                cutoff,
                coordinates,
                neighbor_idxs,
                mask=(species == -1),
                shift_values=shift_values,
            )
        # If the input neighbor idxs are pre screened then
        # directly calculate the distances and diff_vectors from them
        coordinates = coordinates.view(-1, 3)
        coords0 = coordinates.index_select(0, neighbor_idxs[0])
        coords1 = coordinates.index_select(0, neighbor_idxs[1])
        diff_vectors = coords0 - coords1 + shift_values
        distances = diff_vectors.norm(2, -1)

        # Store diff vectors internally for calculation of stress
        self.diff_vectors = diff_vectors

        return NeighborData(
            indices=neighbor_idxs, distances=distances, diff_vectors=diff_vectors
        )

    def get_diff_vectors(self):
        return self.diff_vectors


class FullPairwise(Neighborlist):
    """Compute pairs of atoms that are neighbors, uses pbc depending on
    wether pbc.any() is True or not"""

    def forward(
        self,
        species: Tensor,
        coordinates: Tensor,
        cutoff: float,
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
        return_shift_values: bool = False,
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
                cutoff,
                coordinates,
                atom_index12,
                mask=mask,
                shift_values=shift_values,
                return_shift_values=return_shift_values,
            )
        else:
            num_molecules, num_atoms = species.shape
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
                cutoff,
                coordinates,
                atom_index12,
                mask=mask,
                shift_values=None,
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
    _offset_idx3: Tensor

    def __init__(self):
        super().__init__()
        # Get the grid_idx3 offsets for the surrounding buckets of an
        # arbitrary bucket (I think this is different from SANDER, not sure why)
        # NOTE: This assumes "1-bucket-per-cutoff"
        #
        # In order to avoid double counting, consider only half of the
        # surrounding buckets.
        #
        # Choose all buckets in the bottom plane, and the lower half of the
        # buckets in the same plane, not including the "self" bucket (maybe
        # other choices are possible).
        #
        # Order is reading order: "left-to-right, top-to-bottom"
        #
        # The selected buckets in the planes are:
        # ("x" selected elements, "-" non-selected and "o" reference element)
        # top,   same,  bottom,
        # |---|  |---|  |xxx|
        # |---|  |xo-|  |xxx|
        # |---|  |xxx|  |xxx|
        #
        # shape (surrounding=13, 3)
        self.register_buffer(
            "_offset_idx3",
            torch.tensor(
                [
                    # Surrounding buckets in the same plane (gz-offset = 0)
                    [-1, 0, 0],
                    [-1, -1, 0],
                    [0, -1, 0],
                    [1, -1, 0],
                    # Surrounding buckets in bottom plane (gz-offset = -1)
                    [-1, 1, -1],
                    [0, 1, -1],
                    [1, 1, -1],
                    [-1, 0, -1],
                    [0, 0, -1],
                    [1, 0, -1],
                    [-1, -1, -1],
                    [0, -1, -1],
                    [1, -1, -1],
                ],
                dtype=torch.long,
            ),
            persistent=False,
        )

    def forward(
        self,
        species: Tensor,
        coordinates: Tensor,
        cutoff: float,
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
        return_shift_values: bool = False,
    ) -> NeighborData:
        assert cutoff >= 0.0, "Cutoff must be a positive float"
        assert coordinates.shape[0] == 1, "Cell list doesn't support batches"
        # Coordinates are displaced only if (pbc=False, cell=None), in which
        # case the displaced coordinates lie all inside the created cell.
        # otherwise they are the same as the input coordinates, but "detached"
        displaced_coordinates, cell, pbc = self._validate_and_parse_cell_and_pbc(
            coordinates, cell, pbc
        )

        # The cell is spanned by a 3D grid of "buckets" or "grid elements",
        # which has grid_shape=(GX, GY, GZ) and grid_numel=G=(GX * GY * GZ)
        #
        # It is cheap to set up the grid each step, and avoids keeping track of state
        grid_shape = setup_grid(
            cell.detach(),
            cutoff,
        )

        # Since coords will be fractionalized they may lie outside the cell before this
        atom_pairs, shift_indices = self._compute_cell_list(
            displaced_coordinates.detach(),
            grid_shape,
            cell,
            pbc,
        )

        if pbc.any():
            assert shift_indices is not None
            shift_values = shift_indices.to(cell.dtype) @ cell
            # Before the screening step we map the coordinates to the central cell,
            # same as with a full pairwise calculation
            coordinates = map_to_central(coordinates, cell.detach(), pbc)
            return self._screen_with_cutoff(
                cutoff,
                coordinates,
                atom_pairs,
                mask=(species == -1),
                shift_values=shift_values,
                return_shift_values=return_shift_values,
            )
        return self._screen_with_cutoff(
            cutoff,
            coordinates,
            atom_pairs,
            mask=(species == -1),
            shift_values=None,
        )

    def _validate_and_parse_cell_and_pbc(
        self,
        coordinates: Tensor,
        cell: tp.Optional[Tensor],
        pbc: tp.Optional[Tensor],
    ) -> tp.Tuple[Tensor, Tensor, Tensor]:
        device = coordinates.device
        if pbc is None:
            pbc = torch.zeros(3, dtype=torch.bool, device=device)
        assert pbc is not None

        if not ((~pbc).all() or pbc.all()):
            raise ValueError("Cell list only supports PBC in all or no directions")

        if cell is None:
            if pbc.any():
                raise ValueError("Cell must be provided if PBC is required")
            displaced_coordinates, cell = self.compute_bounding_cell(
                coordinates.detach(),
                eps=1e-3,
            )
            return displaced_coordinates, cell, pbc

        assert cell is not None
        return coordinates.detach(), cell, pbc

    def _compute_cell_list(
        self,
        coordinates: Tensor,  # shape (C, A, 3)
        grid_shape: Tensor,  # shape (3,)
        cell: Tensor,  # shape (3, 3)
        pbc: Tensor,  # shape (3,)
    ) -> tp.Tuple[Tensor, tp.Optional[Tensor]]:
        # 1) Get location of each atom in the grid, given by a "grid_idx3"
        # or by a single flat "grid_idx" (g).
        # Shapes (C, A, 3) and (C, A)
        atom_grid_idx3 = coords_to_grid_idx3(coordinates, cell, grid_shape)
        atom_grid_idx = flatten_idx3(atom_grid_idx3, grid_shape)

        # 2) Get image pairs of atoms WITHIN atoms inside a bucket
        # To do this, first calculate:
        # - Num atoms in each bucket "count_in_grid[g]", and the max, (c-max)
        # - Cumulative num atoms *before* each bucket "comcount_in_grid[g]"
        # Both shapes (G,)
        count_in_grid, cumcount_in_grid = count_atoms_in_buckets(
            atom_grid_idx, grid_shape
        )
        count_in_grid_max: int = int(count_in_grid.max())
        # Shape (2, W)
        _image_pairs_within = image_pairs_within(
            count_in_grid,
            cumcount_in_grid,
            count_in_grid_max,
        )

        # 3) Get image pairs BETWEEN atoms inside a bucket and its surrounding buckets
        #
        # First calc the grid_idx3 associated with the buckets
        # surrounding a given atom. "self bucket" not included.
        #
        # The surrounding buckets will either lie in the central cell or wrap
        # around one or more dimensions due to PBC. In the latter case the
        # atom_surround_idx3 may be negative, or larger than the corrseponding
        # grid_shape dim, which can be used to identify them. The required shift
        # is given by the number and value of negative and overflowing
        # dimensions.
        # surround=N=13 for 1-bucket-per-cutoff
        # shape (C, A, N=13, 3) contains negative and overflowing idxs
        # NOTE: Casting is necessary for long buffers in C++ due to a LibTorch bug
        atom_surround_idx3 = atom_grid_idx3.unsqueeze(-2) + self._offset_idx3.view(
            1, 1, -1, 3
        ).to(dtype=torch.long)
        # Modulo grid_shape is used to get rid of the negative and overflowing idxs
        # shape (C, A, surround=13) contains only positive idxs
        atom_surround_idx = flatten_idx3(atom_surround_idx3 % grid_shape, grid_shape)

        # 4) Calc upper and lower part of the image_pairs_between.
        # The "unpadded" upper part of the pairlist repeats each image idx a
        # number of times equal to the number of atoms on the surroundings of
        # each atom

        # Both shapes (C, A, N=13)
        count_in_atom_surround = count_in_grid[atom_surround_idx]
        cumcount_in_atom_surround = cumcount_in_grid[atom_surround_idx]
        # shape (C, A, N=13, 3), note that the -1 is needed here
        shift_idxs_between = -torch.div(
            atom_surround_idx3, grid_shape, rounding_mode="floor"
        )

        # Shapes (B,) and (B, 3)
        lower_between, shift_idxs_between = lower_image_pairs_between(
            count_in_atom_surround,
            cumcount_in_atom_surround,
            shift_idxs_between,
            count_in_grid_max,
        )

        # Total count of all atoms in buckets surrounding a given atom.
        # shape (C, A, N=13) -> (C, A) -> (C*A,) (get rid of C with view)
        total_count_in_atom_surround = count_in_atom_surround.sum(-1).view(-1)

        # Both shapes (C*A,)
        atom_to_image, image_to_atom = atom_image_converters(atom_grid_idx)

        # For each atom we have one image_pair_between associated with each of
        # the atoms in its surrounding buckets, so we repeat the image-idx of each
        # atom that many times.
        # shape (C*A,), (C*A) -> (B,)
        upper_between = torch.repeat_interleave(
            atom_to_image,
            total_count_in_atom_surround,
        )
        # shape (2, B)
        _image_pairs_between = torch.stack((upper_between, lower_between), dim=0)

        # 5) Get the necessary shifts. If no PBC is needed also get rid of the
        # image_pairs_between that need wrapping
        if not pbc.any():
            _image_pairs_between = fast_masked_select(
                _image_pairs_between,
                (shift_idxs_between == 0).all(dim=-1),
                1,
            )
            shift_idxs = None
        else:
            shift_idxs_within = torch.zeros(
                _image_pairs_within.shape[1],
                3,
                device=grid_shape.device,
                dtype=torch.long,
            )
            shift_idxs = torch.cat((shift_idxs_between, shift_idxs_within), dim=0)

        # 6) Concatenate all image pairs, and convert to atom pairs
        image_pairs = torch.cat((_image_pairs_between, _image_pairs_within), dim=1)
        atom_pairs = image_to_atom[image_pairs]
        return atom_pairs, shift_idxs


def coords_to_grid_idx3(
    coordinates: Tensor,  # shape (C, A, 3)
    cell: Tensor,  # shape (3, 3)
    grid_shape: Tensor,  # shape (3,)
) -> Tensor:
    # 1) Fractionalize coordinates. After this ll coordinates lie in [0., 1.)
    fractionals = coords_to_fractional(coordinates, cell)  # shape (C, A, 3)
    # 2) Assign to each fractional its corresponding grid_idx3
    return torch.floor(fractionals * grid_shape).to(torch.long)


def coords_to_fractional(coordinates: Tensor, cell: Tensor) -> Tensor:
    # Transform and wrap all coordinates to be relative to the cell vectors
    #
    # Input to this function may have coordinates outside the box. If the
    # coordinate is 0.16 or 3.15 times the cell length, it is turned into 0.16
    # or 0.15 respectively.
    # All output coords are in the range [0.0, 1.0)
    fractional_coords = torch.matmul(coordinates, cell.inverse())
    fractional_coords -= fractional_coords.floor()
    fractional_coords[fractional_coords >= 1.0] += -1.0
    fractional_coords[fractional_coords < 0.0] += 1.0
    return fractional_coords


def flatten_idx3(
    idx3: Tensor,
    grid_shape: Tensor,
) -> Tensor:
    # Convert a tensor that holds idx3 (all of which lie inside the central
    # grid) to one that holds flat idxs (last dimension is removed). For
    # row-major flattening the factors needed are: (GY * GZ, GZ, 1)
    grid_factors = grid_shape.clone()
    grid_factors[0] = grid_shape[1] * grid_shape[2]
    grid_factors[1] = grid_shape[2]
    grid_factors[2] = 1
    return (idx3 * grid_factors).sum(-1)


def atom_image_converters(grid_idx: Tensor) -> tp.Tuple[Tensor, Tensor]:
    # NOTE: Since sorting is not stable this may scramble the atoms,
    # this is not important for the neighborlist, only the pairs are important,
    # and non-stable sorting is marginally faster.

    # For the "image indices", (indices that sort atoms in the order of
    # the flattened bucket index) only occupied buckets are considered, so
    # if a bucket is unoccupied the index is not taken into account.  for
    # example if the atoms are distributed as:
    # |1 9 8|, |empty|, |3 2 4|, |7|
    # where the bars delimit flat buckets, then the image idxs will be:
    # |0 1 2|, |empty|, |3 4 5|, |6|
    #
    # "atom_indices" can be reconstructed from the "image_indices". The
    # pairlist can be built with image indices, and
    # image_to_atom[image_neighborlist] = atom_neighborlist

    grid_idx = grid_idx.view(-1)  # shape (C, A) -> (A,), get rid of C
    image_to_atom = torch.argsort(grid_idx)
    atom_to_image = torch.argsort(image_to_atom)
    # Both shapes (A,)
    return atom_to_image, image_to_atom


def count_atoms_in_buckets(
    atom_grid_idx: Tensor,  # shape (C, A)
    grid_shape: Tensor,
) -> tp.Tuple[Tensor, Tensor]:
    # Return number of atoms in each bucket, and cumulative number of
    # atoms before each bucket, as indexed with the flat grid_idx
    atom_grid_idx = atom_grid_idx.view(-1)  # shape (A,), get rid of C
    count_in_grid = torch.bincount(atom_grid_idx, minlength=int(grid_shape.prod()))
    # Both shape (G,), the total number of "grid elements"|"buckets"
    return count_in_grid, cumsum_from_zero(count_in_grid)


def setup_grid(
    cell: Tensor,
    cutoff: float,
    buckets_per_cutoff: int = 1,
    extra_space: float = 1e-5,
) -> Tensor:
    # Get the shape (GX, GY, GZ) of the grid. Some extra space is used as slack.
    #
    # NOTE: "buckets_per_cutoff" determines how fine grained the 3D grid is,
    # with respect to the distance cutoff, and is currently hardcoded to 1 in
    # CellList and VerletCellList, but support for 2 may be possible. This may
    # be 2 for SANDER, not sure. If this is changed then the surround_offsets
    # must also be changed
    #
    # NOTE: extra_space is currently hardcoded to be consistent with SANDER
    #
    # The spherical factor is different from 1 in the case of nonorthogonal
    # boxes and accounts for the "spherical protrusion", which is related
    # to the fact that the sphere of radius "cutoff" around an atom needs
    # some more room to fit in nonorthogonal boxes.

    # To get the shape of the grid (number of "buckets" or "grid elements"
    # in each direction) calculate first a lower bound, and afterwards
    # perform floor division.
    #
    # NOTE: This is not actually the bucket length used in the grid,
    # it is only a lower bound used to calculate the grid size, it is the minimum
    # size that spawns a new bucket in the grid.
    spherical_factor = torch.tensor(
        [1.0, 1.0, 1.0], dtype=torch.float, device=cell.device
    )  # TODO: calculate correctly
    bucket_length_lower_bound = (
        spherical_factor * cutoff / buckets_per_cutoff
    ) + extra_space

    # Lengths of each cell edge are given by norm of each cell basis vector
    cell_lengths = torch.linalg.norm(cell, dim=0)

    # For example, if a cell length is "3 * bucket_length_lower_bound + eps" it
    # can be covered with 3 buckets if they are stretched to be slightly larger
    # than the lower bound.
    grid_shape = torch.div(
        cell_lengths,
        bucket_length_lower_bound,
        rounding_mode="floor",
    ).to(torch.long)

    if (grid_shape == 0).any():
        raise RuntimeError("Cell is too small to perform pbc calculations")
    return grid_shape


def image_pairs_within(
    count_in_grid: Tensor,  # shape (G,)
    cumcount_in_grid: Tensor,  # shape (G,)
    count_in_grid_max: int,  # max number of atoms in any bucket
) -> Tensor:
    device = count_in_grid.device
    # Calc all possible image-idx-pairs within each central bucket ("W" in total)
    # Output is shape (2, W)
    #
    # NOTE: Inside each central bucket there are grid_count[g] num atoms.
    # These atoms are indexed with an "image idx", "i", different from the "atom idx"
    # which indexes the atoms in the coords
    # For instance:
    # - central bucket g=0 has "image" atoms 0...grid_count[0]
    # - central bucket g=1 has "image" atoms grid_count[0]...grid_count[1], etc

    # 1) Get idxs "g" that have atom pairs inside ("H" in total).
    # Index them with 'h' using g[h], from this, get count_in_haspairs[h] and
    # cumcount_in_haspairs[h].
    # shapes are (H,)
    haspairs_idx_to_grid_idx = (count_in_grid > 1).nonzero().view(-1)
    count_in_haspairs = count_in_grid.index_select(0, haspairs_idx_to_grid_idx)
    cumcount_in_haspairs = cumcount_in_grid.index_select(0, haspairs_idx_to_grid_idx)

    # 2) Get image pairs pairs assuming every bucket (with pairs) has
    # the same num atoms as the fullest one. To do this:
    # - Get the image-idx-pairs for the fullest bucket
    # - Repeat (view) the image pairs in the fullest bucket H-times,
    # - Add to each repeat the cumcount of atoms in all previous buckets.
    #
    # After this step:
    # - There are more pairs than needed
    # - Some of the extra pairs may have out-of-bounds idxs
    # Screen the incorrect, unneeded pairs in the next step.
    # shapes are (2, cp-max) and (2, H*cp-max)
    image_pairs_in_fullest_bucket = torch.tril_indices(
        count_in_grid_max,
        count_in_grid_max,
        offset=-1,
        device=device,
    )
    _image_pairs_within = (
        image_pairs_in_fullest_bucket.view(2, 1, -1)
        + cumcount_in_haspairs.view(1, -1, 1)
    ).view(2, -1)

    # 3) Get actual number of pairs in each bucket (with pairs), and a
    # mask that selects those from the unscreened pairs
    # shapes (H,) (cp-max,), (H, cp-max)
    paircount_in_haspairs = torch.div(
        count_in_haspairs * torch.sub(count_in_haspairs, 1),
        2,
        rounding_mode="floor",
    )
    mask = torch.arange(0, image_pairs_in_fullest_bucket.shape[1], device=device)
    mask = mask.view(1, -1) < paircount_in_haspairs.view(-1, 1)

    # 4) Screen the incorrect, unneeded pairs.
    # shape (2, H*cp-max) -> (2, W)
    return fast_masked_select(_image_pairs_within, mask, 1)


def lower_image_pairs_between(
    count_in_atom_surround: Tensor,  # shape (C, A, N=13)
    cumcount_in_atom_surround: Tensor,  # shape (C, A, N=13)
    shift_idxs_between: Tensor,  # shape (C, A, N=13, 3)
    count_in_grid_max: int,  # scalar
) -> tp.Tuple[Tensor, Tensor]:
    # Calculate "lower" part of the image_pairs_between buckets
    device = count_in_atom_surround.device
    mols, atoms, neighbors = count_in_atom_surround.shape
    # shape is (c-max)
    padded_atom_neighbors = torch.arange(0, count_in_grid_max, device=device)
    # shape (1, 1, 1, c-max)
    padded_atom_neighbors = padded_atom_neighbors.view(1, 1, 1, -1)
    # shape (C, A, N=13, c-max)
    padded_atom_neighbors = padded_atom_neighbors.repeat(mols, atoms, neighbors, 1)

    # Create a mask to unpad the padded neighbors
    # shape (C, A, N=13, c-max)
    mask = padded_atom_neighbors < count_in_atom_surround.unsqueeze(-1)
    padded_atom_neighbors.add_(cumcount_in_atom_surround.unsqueeze(-1))

    # Pad the shift idxs
    # shape  (C, A, N=13, 1, 3)
    shift_idxs_between = shift_idxs_between.unsqueeze(-2)
    # shape  (C, A, N=13, c-max, 3)
    shift_idxs_between = shift_idxs_between.repeat(1, 1, 1, count_in_grid_max, 1)

    # Apply the mask
    # Both shapes (B,)
    lower_between = fast_masked_select(padded_atom_neighbors.view(-1), mask, 0)
    shift_idxs_between = fast_masked_select(shift_idxs_between.view(-1, 3), mask, 0)
    return lower_between, shift_idxs_between


# TODO: Currently broken
class VerletCellList(CellList):
    _old_shift_indices: Tensor
    _old_atom_pairs: Tensor
    _old_coordinates: Tensor
    _old_cell_lenghts: Tensor

    def __init__(
        self,
        skin: float = 1.0,
    ):
        super().__init__()
        if skin <= 0.0:
            raise ValueError("skin must be a positive float")
        self.skin = skin
        self.register_buffer(
            "_old_shift_indices",
            torch.zeros(1, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_old_atom_pairs",
            torch.zeros(1, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_old_coordinates",
            torch.zeros(1),
            persistent=False,
        )
        self.register_buffer(
            "_old_cell_lenghts",
            torch.zeros(1),
            persistent=False,
        )
        self._old_values_are_cached = False
        raise ValueError("VerletCellList is currently unsupported")

    def forward(
        self,
        species: Tensor,
        coordinates: Tensor,
        cutoff: float,
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
        return_shift_values: bool = False,
    ) -> NeighborData:
        assert cutoff >= 0.0, "Cutoff must be a positive float"
        assert coordinates.shape[0] == 1, "Cell list doesn't support batches"
        displaced_coordinates, cell, pbc = self._validate_and_parse_cell_and_pbc(
            coordinates, cell, pbc
        )
        grid_shape = setup_grid(
            cell.detach(),
            cutoff,
        )
        cell_lengths = torch.linalg.norm(cell.detach(), dim=0)
        if self._can_use_old_list(displaced_coordinates, cell_lengths):
            # If a cell list is not needed use the old cached values
            # NOTE: Cached values should NOT be updated here
            atom_pairs = self._old_atom_pairs
            shift_indices: tp.Optional[Tensor] = self._old_shift_indices
        else:
            # TODO: The cell list should be calculated with a skin (?) here.
            atom_pairs, shift_indices = self._compute_cell_list(
                displaced_coordinates.detach(),
                grid_shape,
                cell,
                pbc,
            )
            self._cache_values(
                atom_pairs,
                shift_indices,
                displaced_coordinates.detach(),
                cell_lengths,
            )
        if pbc.any():
            assert shift_indices is not None
            shift_values = shift_indices.to(cell.dtype) @ cell
            coordinates = map_to_central(coordinates, cell.detach(), pbc)
            return self._screen_with_cutoff(
                cutoff,
                coordinates,
                atom_pairs,
                mask=(species == -1),
                shift_values=shift_values,
                return_shift_values=return_shift_values,
            )
        return self._screen_with_cutoff(
            cutoff,
            coordinates,
            atom_pairs,
            mask=(species == -1),
            shift_values=None,
        )

    def _cache_values(
        self,
        atom_pairs: Tensor,
        shift_indices: tp.Optional[Tensor],
        coordinates: Tensor,
        cell_lengths: Tensor,
    ):
        if shift_indices is not None:
            self._old_shift_indices = shift_indices.detach()
        self._old_atom_pairs = atom_pairs.detach()
        self._old_coordinates = coordinates.detach()
        self._old_cell_lengths = cell_lengths.detach()
        self._old_values_are_cached = True

    @torch.jit.unused
    def reset_cached_values(self, dtype: torch.dtype = torch.float) -> None:
        device = self._offset_idx3.device
        self._cache_values(
            torch.zeros(1, dtype=torch.long, device=device),
            torch.zeros(1, dtype=torch.long, device=device),
            torch.zeros(1, dtype=dtype, device=device),
            torch.zeros(1, dtype=dtype, device=device),
        )
        self._old_values_are_cached = False

    def _can_use_old_list(self, coordinates: Tensor, cell_lengths: Tensor) -> bool:
        if not self._old_values_are_cached:
            return False
        coordinates.detach_()
        cell_lengths.detach_()
        # Check if any coordinate moved more than half the skin depth,
        # If this happened, then the cell list has to be rebuilt
        cell_scaling = cell_lengths / self._old_cell_lenghts
        delta = coordinates - self._old_coordinates * cell_scaling
        dist_squared = delta.pow(2).sum(-1)
        return bool((dist_squared > (self.skin / 2) ** 2).all())


NeighborlistArg = tp.Union[
    tp.Literal[
        "full_pairwise",
        "cell_list",
        "verlet_cell_list",
        "base",
    ],
    Neighborlist,
]


def parse_neighborlist(neighborlist: NeighborlistArg = "base") -> Neighborlist:
    if neighborlist == "full_pairwise":
        neighborlist = FullPairwise()
    elif neighborlist == "cell_list":
        neighborlist = CellList()
    elif neighborlist == "verlet_cell_list":
        neighborlist = VerletCellList()
    elif neighborlist == "base":
        neighborlist = Neighborlist()
    elif not isinstance(neighborlist, Neighborlist):
        raise ValueError(f"Unsupported neighborlist: {neighborlist}")
    return tp.cast(Neighborlist, neighborlist)


_global_cell_list = CellList()


def _call_global_cell_list(
    species: Tensor,
    coordinates: Tensor,
    cutoff: float,
    cell: tp.Optional[Tensor] = None,
    pbc: tp.Optional[Tensor] = None,
    return_shift_values: bool = False,
) -> NeighborData:
    out = _global_cell_list(species, coordinates, cutoff, cell, pbc)
    # Reset state
    _global_cell_list.diff_vectors = torch.empty(0)
    return out


_call_global_all_pairs = FullPairwise()
