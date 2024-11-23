r"""Modular neighborlists to improve scaling for large systems"""

import typing as tp
import math

import torch
from torch import Tensor

from torchani.utils import map_to_central, cumsum_from_zero, fast_masked_select


class Neighbors(tp.NamedTuple):
    r"""Holds pairs of atoms that are neighbors. Result of a neighborlist calculation"""

    indices: Tensor  #: Long tensor with idxs of neighbor pairs. Shape ``(2, pairs)``
    distances: Tensor  #: The associated pair distances. Shape is ``(pairs,)``
    diff_vectors: Tensor  #: The associated difference vectors. Shape is ``(pairs, 3)``


class Triples(tp.NamedTuple):
    r"""Holds groups of 3 atoms that are neighbors. Result of `neighbors_to_triples`"""

    central_idxs: Tensor  #: Long tensor with central idxs. Shape ``(triples,)``
    side_idxs: Tensor  #: Long tensor with side idxs. Shape ``(2, triples)``
    diff_signs: Tensor  #: Long tensor of diff vector directions. Shape ``(2, triples)``
    distances: Tensor  #: Similar to `Neighbors`, for triples. ``(2, triples)``
    diff_vectors: Tensor  #: Similar to `Neighbors`, for triples. ``(2, triples, 3)``


def discard_outside_cutoff(
    neighbors: Neighbors,
    cutoff: float,
) -> Neighbors:
    r"""Discard neighbors with distances that lie outside of the given cutoff"""
    closer_indices = (neighbors.distances <= cutoff).nonzero().flatten()
    indices = neighbors.indices.index_select(1, closer_indices)
    distances = neighbors.distances.index_select(0, closer_indices)
    diff_vectors = neighbors.diff_vectors.index_select(0, closer_indices)
    return Neighbors(indices, distances, diff_vectors)


# Screen a given neighborlist using a cutoff and return a neighborlist with
# atoms that are within that cutoff, for all molecules in a coordinate set.
# neighbor_idxs must correctly index flattened coords.view(-1, 3)
#
# passing an infinite cutoff will only work for non pbc conditions
# (shift values must be None)
def narrow_down(
    cutoff: float,
    elem_idxs: Tensor,
    coords: Tensor,
    neighbor_idxs: Tensor,
    shifts: tp.Optional[Tensor] = None,
) -> Neighbors:
    r"""Takes a set of potential neighbor idxs and narrows it down to true neighbors"""
    mask = elem_idxs == -1
    if not torch.compiler.is_compiling() and mask.any():
        # Discard dumy atoms to prevent wasting resources in calculating
        # dummy distances
        mask = mask.view(-1)[neighbor_idxs.view(-1)].view(2, -1)
        non_dummy_pairs = (~torch.any(mask, dim=0)).nonzero().flatten()
        neighbor_idxs = neighbor_idxs.index_select(1, non_dummy_pairs)
        # shifts can be None when there are no pbc conditions to prevent
        # torch from launching kernels with only zeros
        if shifts is not None:
            shifts = shifts.index_select(0, non_dummy_pairs)

    # Interpret as single molecule
    coords = coords.view(-1, 3)
    if cutoff == math.inf:
        if shifts is not None:
            raise ValueError("PBC can't use an infinite cutoff")
    else:
        # Diff vector and distances need to be calculated to screen unfortunately
        # distances need to be recalculated again later, since otherwise torch
        # prepares to calculate derivatives of distances that are later discarded

        # no grad tracking on coords #
        _coords = coords.detach()
        _coords0 = _coords.index_select(0, neighbor_idxs[0])
        _coords1 = _coords.index_select(0, neighbor_idxs[1])
        _diff_vectors = _coords0 - _coords1
        if shifts is not None:
            _diff_vectors += shifts
        in_cutoff = (_diff_vectors.norm(2, -1) <= cutoff).nonzero().flatten()
        neighbor_idxs = neighbor_idxs.index_select(1, in_cutoff)
        if shifts is not None:
            shifts = shifts.index_select(0, in_cutoff)
        # ------------------- #

    coords0 = coords.index_select(0, neighbor_idxs[0])
    coords1 = coords.index_select(0, neighbor_idxs[1])
    diff_vectors = coords0 - coords1
    if shifts is not None:
        diff_vectors += shifts
    distances = diff_vectors.norm(2, -1)
    return Neighbors(neighbor_idxs, distances, diff_vectors)


def compute_bounding_cell(
    coords: Tensor, eps: float = 1e-3, displace: bool = True, square: bool = False
) -> tp.Tuple[Tensor, Tensor]:
    r"""Compute the rectangular unit cell that minimally bounds a set of coords.

    Optionally displace the coords so that they fully lie inside the cell. Displacing
    the coordinates causes small floating point differences which have a negligible
    effect on energies and forces. The ``eps`` value is used to add padding to the cell.
    """
    min_ = torch.min(coords.view(-1, 3), dim=0).values - eps
    max_ = torch.max(coords.view(-1, 3), dim=0).values + eps
    largest_dist = max_ - min_
    if square:
        cell = (
            torch.eye(3, dtype=torch.float, device=coords.device) * largest_dist.max()
        )
    else:
        cell = torch.eye(3, dtype=torch.float, device=coords.device) * largest_dist
    if displace:
        # Benchmarks show that these assert coords > 0.0 slows things down a bit
        return coords - min_, cell
    return coords, cell


class Neighborlist(torch.nn.Module):
    r"""Base class for modules that compute pairs of neighbors. Can support PBC.

    Subclasses *must* override `Neighborlist.forward`
    """

    def forward(
        self,
        cutoff: float,
        species: Tensor,
        coords: Tensor,
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
    ) -> Neighbors:
        r"""Calculate all pairs of atoms that are neighbors, given a cutoff.

        Args:
            cutoff: Cutoff value for the neighborlist. Pairs further away than this
                are not included.
            species: |elem_idxs|
            coords: |coords|
            cell: |cell|
            pbc: |pbc|
        Returns:
            `typing.NamedTuple` with all pairs of atoms that are neighbors.
        """
        raise NotImplementedError("Must be implemented by subclasses")


class AllPairs(Neighborlist):
    r"""Compute pairs of neighbors. Uses a naive algorithm.

    This is a naive implementation, with :math:`O(N^2)` scaling. It computes all pairs
    and then discards those that are further away from the cutoff. Supports PBC.
    """

    def forward(
        self,
        cutoff: float,
        species: Tensor,
        coords: Tensor,
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
    ) -> Neighbors:
        return all_pairs(cutoff, species, coords, cell, pbc)


def all_pairs(
    cutoff: float,
    species: Tensor,
    coords: Tensor,
    cell: tp.Optional[Tensor] = None,
    pbc: tp.Optional[Tensor] = None,
) -> Neighbors:
    _validate_inputs(cutoff, species, coords, cell, pbc)

    if pbc is not None:
        assert cell is not None
        neighbor_idxs, shift_idxs = _all_pairs_pbc(cutoff, species, cell, pbc)
        shifts = shift_idxs.to(cell.dtype) @ cell
        # Before screening coords, must map to central cell (not need if no PBC)
        coords = map_to_central(coords, cell, pbc)
        return narrow_down(cutoff, species, coords, neighbor_idxs, shifts)
    molecs, atoms = species.shape
    # Create a neighborlist for all molecules and all atoms.
    # Later screen dummy atoms
    device = species.device
    neighbor_idxs = torch.triu_indices(atoms, atoms, 1, device=device)
    if molecs > 1:
        neighbor_idxs = neighbor_idxs.unsqueeze(1).repeat(1, molecs, 1)
        neighbor_idxs += atoms * torch.arange(molecs, device=device).view(1, -1, 1)
        neighbor_idxs = neighbor_idxs.view(-1).view(2, -1)
    return narrow_down(cutoff, species, coords, neighbor_idxs)


def _all_pairs_pbc(
    cutoff: float,
    species: Tensor,
    cell: Tensor,
    pbc: Tensor,
) -> tp.Tuple[Tensor, Tensor]:
    cell = cell.detach()
    shifts = _all_pairs_pbc_shifts(cutoff, cell, pbc)
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
    all_neighbor_idxs = torch.cat([p12_center, p12], dim=1)
    return all_neighbor_idxs, shifts_all


# Compute the shifts of unit cell along the given cell vectors to make it
# large enough to contain all pairs of neighbor atoms with PBC under
# consideration
# Returns a long tensor of shifts. the center cell and symmetric cells are not
# included.
def _all_pairs_pbc_shifts(cutoff: float, cell: Tensor, pbc: Tensor) -> Tensor:
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
    r"""Compute pairs of neighbors using the 'Cell List' algorithm.

    This is a linearly scaling implementation that uses the 'cell list' algorithm. It
    subdivides space into cells and then computes all pairs within each cell and between
    each cell and neighboring cells. and then discards those that are further away from
    the cutoff.
    """

    def forward(
        self,
        cutoff: float,
        species: Tensor,
        coords: Tensor,
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
    ) -> Neighbors:
        return cell_list(cutoff, species, coords, cell, pbc)


class AdaptiveList(Neighborlist):
    r"""Compute pairs of neighbors using the best algorithm for the system size

    This is a linearly scaling implementation that uses the 'cell list' algorithm for
    large system sizes and the naive 'all pairs' for small sizes.
    """

    def __init__(self, threshold: int = 190, threshold_nopbc: int = 1770) -> None:
        super().__init__()
        self._thresh = threshold
        self._thresh_nopbc = threshold_nopbc

    def forward(
        self,
        cutoff: float,
        species: Tensor,
        coords: Tensor,
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
    ) -> Neighbors:
        if pbc is not None:
            return adaptive_list(cutoff, species, coords, cell, pbc, self._thresh)
        return adaptive_list(cutoff, species, coords, cell, pbc, self._thresh_nopbc)


def adaptive_list(
    cutoff: float,
    species: Tensor,
    coords: Tensor,
    cell: tp.Optional[Tensor] = None,
    pbc: tp.Optional[Tensor] = None,
    threshold: int = 190,
) -> Neighbors:
    _validate_inputs(
        cutoff,
        species,
        coords,
        cell,
        pbc,
        supports_batches=False,
        supports_individual_pbc=False,
    )
    # Forward directly to all_pairs if below threshold or in the non-pbc case
    if coords.shape[1] < threshold:
        return all_pairs(cutoff, species, coords, cell, pbc)
    # NOTE: This disallows pbc with self interactions, which may or may not be desirable
    return cell_list(cutoff, species, coords, cell, pbc)


def cell_list(
    cutoff: float,
    species: Tensor,
    coords: Tensor,
    cell: tp.Optional[Tensor] = None,
    pbc: tp.Optional[Tensor] = None,
) -> Neighbors:
    _validate_inputs(
        cutoff,
        species,
        coords,
        cell,
        pbc,
        supports_batches=False,
        supports_individual_pbc=False,
    )

    # Coordinates are displaced only in the non-pbc case, in which case the
    # displaced coords lie all inside the created cell. otherwise they are
    # the same as the input coords, but "detached"
    if pbc is not None:
        assert cell is not None
        displ_coords = coords.detach()
    else:
        # Make cell large enough to deny PBC interaction (fast, not bottleneck)
        displ_coords, cell = compute_bounding_cell(
            coords.detach(),
            eps=(cutoff + 1e-3),
        )

    # The cell is spanned by a 3D grid of "buckets" or "grid elements",
    # which has grid_shape=(GX, GY, GZ) and grid_numel=G=(GX * GY * GZ)
    #
    # Set up grid shape each step. Cheap and avoids state in the cls
    grid_shape = setup_grid(cell.detach(), cutoff)
    if pbc is not None:
        if (grid_shape == 0).any():
            raise RuntimeError("Cell is too small to perform pbc calculations")
    else:
        grid_shape = torch.max(grid_shape, grid_shape.new_ones(grid_shape.shape))

    # Since coords will be fractionalized they may lie outside the cell before this
    neighbor_idxs, shift_idxs = _cell_list(grid_shape, displ_coords.detach(), cell, pbc)
    if pbc is not None:
        shifts = shift_idxs.to(cell.dtype) @ cell
        # Before the screening step we map the coords to the central cell,
        # same as with an all-pairs calculation
        coords = map_to_central(coords, cell.detach(), pbc)
        return narrow_down(cutoff, species, coords, neighbor_idxs, shifts)
    return narrow_down(cutoff, species, coords, neighbor_idxs)


def _cell_list(
    grid_shape: Tensor,  # shape (3,)
    coords: Tensor,  # shape (C, A, 3)
    cell: Tensor,  # shape (3, 3)
    pbc: tp.Optional[Tensor],  # shape (3,)
) -> tp.Tuple[Tensor, Tensor]:
    # 1) Get location of each atom in the grid, given by a "grid_idx3" (g3) or by a
    # single flat "grid_idx" (g).
    atom_grid_idx3 = coords_to_grid_idx3(coords, cell, grid_shape)  # Shape (M, A, 3)
    atom_grid_idx = flatten_idx3(atom_grid_idx3, grid_shape)  # Shape (M, A)

    # 2) Get image pairs of atoms WITHIN atoms inside a bucket
    # To do this, first calculate:
    # - Num atoms in each bucket "count_in_grid[g]", and the max, (c-max)
    # - Cumulative num atoms *before* each bucket "comcount_in_grid[g]"
    # Both shapes (G,)
    count_in_grid, cumcount_in_grid = count_atoms_in_buckets(atom_grid_idx, grid_shape)
    count_in_grid_max: int = int(count_in_grid.max())
    # Shape (2, within-pairs) "within-pairs aka W"
    _image_pairs_within = image_pairs_within(
        count_in_grid, cumcount_in_grid, count_in_grid_max
    )

    # 3) Get image pairs BETWEEN atoms inside a bucket and its surrounding buckets
    #
    # First calc the grid_idx3 associated with the buckets
    # surrounding a given atom. "self bucket" not included.
    #
    # The surrounding buckets will either lie in the central cell or wrap
    # around one or more dimensions due to PBC. In the latter case the
    # atom_surr_idx3 may be negative, or larger than the corrseponding
    # grid_shape dim, which can be used to identify them. The required shift
    # is given by the number and value of negative and overflowing
    # dimensions.
    # surround=N=13 for 1-bucket-per-cutoff
    # shape (C, A, N=13, 3) contains negative and overflowing idxs
    # NOTE: Casting is necessary for long buffers in C++ due to a LibTorch bug
    offset_idx3 = _offset_idx3().to(coords.device).view(1, 1, -1, 3)
    atom_surr_idx3 = atom_grid_idx3.unsqueeze(-2) + offset_idx3
    # shape (C, A, N=13, 3), note that the -1 is needed here
    shift_idxs_between = -torch.div(atom_surr_idx3, grid_shape, rounding_mode="floor")
    # Modulo grid_shape is used to get rid of the negative and overflowing idxs
    # shape (C, A, surround=13) contains only positive idxs
    atom_surr_idx = flatten_idx3(atom_surr_idx3 % grid_shape, grid_shape)

    # 4) Calc upper and lower part of the image_pairs_between.
    # The "unpadded" upper part of the pairlist repeats each image idx a
    # number of times equal to the number of atoms on the surroundings of
    # each atom
    # Both shapes (C, A, N=13)
    count_in_atom_surround = count_in_grid[atom_surr_idx]  # shp (C, A, N=13)
    cumcount_in_atom_surround = cumcount_in_grid[atom_surr_idx]  # shp (C, A, N=13)

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
    upper_between = torch.repeat_interleave(atom_to_image, total_count_in_atom_surround)
    # shape (2, B)
    _image_pairs_between = torch.stack((upper_between, lower_between), dim=0)

    # 5) Get the necessary shifts. If no PBC is needed also get rid of the
    # image_pairs_between that need wrapping
    shift_idxs_within = torch.zeros(
        _image_pairs_within.shape[1],
        3,
        device=grid_shape.device,
        dtype=torch.long,
    )
    shift_idxs = torch.cat((shift_idxs_between, shift_idxs_within), dim=0)

    # 6) Concatenate all image pairs, and convert to atom pairs
    image_pairs = torch.cat((_image_pairs_between, _image_pairs_within), dim=1)
    neighbor_idxs = image_to_atom[image_pairs]
    return neighbor_idxs, shift_idxs


def _offset_idx3() -> Tensor:
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
    return torch.tensor(
        [
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
    )


# Input shapes are (molecs, atoms, 3) (3, 3) (3,)
def coords_to_grid_idx3(coords: Tensor, cell: Tensor, grid_shape: Tensor) -> Tensor:
    # 1) Fractionalize coords. After this coords lie in [0., 1.)
    fractionals = coords_to_fractional(coords, cell)  # shape (C, A, 3)
    # 2) Assign to each fractional its corresponding grid_idx3
    return torch.floor(fractionals * grid_shape).to(torch.long)


def coords_to_fractional(coords: Tensor, cell: Tensor) -> Tensor:
    # Transform and wrap all coords to be relative to the cell vectors
    #
    # Input to this function may have coords outside the box. If the
    # coordinate is 0.16 or 3.15 times the cell length, it is turned into 0.16
    # or 0.15 respectively.
    # All output coords are in the range [0.0, 1.0)
    fractional_coords = torch.matmul(coords, cell.inverse())
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
        cell_lengths, bucket_length_lower_bound, rounding_mode="floor"
    ).to(torch.long)
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
        count_in_grid_max, count_in_grid_max, offset=-1, device=device
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


# Calculate "lower" part of the image_pairs_between buckets
def lower_image_pairs_between(
    count_in_atom_surround: Tensor,  # shape (C, A, N=13)
    cumcount_in_atom_surround: Tensor,  # shape (C, A, N=13)
    shift_idxs_between: Tensor,  # shape (C, A, N=13, 3)
    count_in_grid_max: int,  # scalar
) -> tp.Tuple[Tensor, Tensor]:
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
    # Both shapes (between-pairs,) "between-pairs aka B"
    lower_between = fast_masked_select(padded_atom_neighbors.view(-1), mask, 0)
    shift_idxs_between = fast_masked_select(shift_idxs_between.view(-1, 3), mask, 0)
    return lower_between, shift_idxs_between


# TODO: testme
class VerletCellList(CellList):
    r"""Compute pairs of neighbors. Uses a cell-list algorithm with 'verlet' skin."""

    _prev_shift_idxs: Tensor
    _prev_neighbor_idxs: Tensor
    _prev_coords: Tensor
    _prev_cell_lenghts: Tensor

    def __init__(self, skin: float = 1.0):
        super().__init__()
        if skin <= 0.0:
            raise ValueError("skin must be a positive float")
        self.skin = skin
        self.register_buffer(
            "_prev_shift_idxs", torch.zeros(1, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "_prev_neighbor_idxs", torch.zeros(1, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "_prev_coords", torch.zeros(1, dtype=torch.float), persistent=False
        )
        self.register_buffer(
            "_prev_cell", torch.eye(3, dtype=torch.float), persistent=False
        )
        self._prev_values_are_cached = False

    def forward(
        self,
        cutoff: float,
        species: Tensor,
        coords: Tensor,
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
    ) -> Neighbors:
        _validate_inputs(
            cutoff,
            species,
            coords,
            cell,
            pbc,
            supports_batches=False,
            supports_individual_pbc=False,
        )

        if pbc is not None:
            assert cell is not None
            displ_coords = coords.detach()
        else:
            # Make cell large enough to avoid PBC interaction (fast, not bottleneck)
            displ_coords, cell = compute_bounding_cell(
                coords.detach(),
                eps=(cutoff + 1e-3),
            )

        # The grid uses a skin, but the narrowing uses the actual cutoff
        grid_shape = setup_grid(cell.detach(), cutoff + self.skin)
        if pbc is not None:
            if (grid_shape == 0).any():
                raise RuntimeError("Cell is too small to perform pbc calculations")
        else:
            grid_shape = torch.max(grid_shape, grid_shape.new_ones(grid_shape.shape))
        if self._can_use_prev_list(displ_coords, cell.detach()):
            # If a cell list is not needed use the old cached values
            # NOTE: Cached values should NOT be updated here
            # Casting is required due to LibTorch bug
            neighbor_idxs = self._prev_neighbor_idxs.to(torch.long)
            shift_idxs = self._prev_shift_idxs.to(torch.long)
        else:
            neighbor_idxs, shift_idxs = _cell_list(
                grid_shape, displ_coords.detach(), cell, pbc
            )
            self._cache_values(
                neighbor_idxs, shift_idxs, displ_coords.detach(), cell.detach()
            )
        if pbc is not None:
            shifts = shift_idxs.to(cell.dtype) @ cell
            coords = map_to_central(coords, cell.detach(), pbc)
            return narrow_down(cutoff, species, coords, neighbor_idxs, shifts)
        return narrow_down(cutoff, species, coords, neighbor_idxs)

    def _cache_values(
        self,
        neighbor_idxs: Tensor,
        shift_idxs: Tensor,
        coords: Tensor,
        cell: Tensor,
    ):
        self._prev_shift_idxs = shift_idxs.detach()
        self._prev_neighbor_idxs = neighbor_idxs.detach()
        self._prev_coords = coords.detach()
        self._prev_cell = cell.detach()
        self._prev_values_are_cached = True

    @torch.jit.export
    def reset_cached_values(self) -> None:
        device = self._prev_coords.device
        dtype = self._prev_coords.dtype
        self._cache_values(
            torch.zeros(1, dtype=torch.long, device=device),
            torch.zeros(1, dtype=torch.long, device=device),
            torch.zeros(1, dtype=dtype, device=device),
            torch.eye(3, dtype=dtype, device=device),
        )
        self._prev_values_are_cached = False

    def _can_use_prev_list(self, coords: Tensor, cell: Tensor) -> bool:
        if not self._prev_values_are_cached:
            return False
        coords.detach_()
        cell.detach_()
        # Check if any coordinate moved more than half the skin depth,
        # If this happened, then the cell list has to be rebuilt
        scaling = torch.linalg.norm(cell, dim=1) / torch.linalg.norm(
            self._prev_cell, dim=1
        )
        pbc = torch.tensor([True, True, True], device=coords.device)
        # delta needs to be wrt wrapped coordinates, since the shifts may need to be
        # updated too
        delta = (
            map_to_central(coords, cell, pbc)
            - map_to_central(self._prev_coords, self._prev_cell, pbc) * scaling
        )
        dist_squared = delta.pow(2).sum(-1)
        return bool((dist_squared > (self.skin / 2) ** 2).all())


NeighborlistArg = tp.Union[
    tp.Literal[
        "all_pairs",
        "adaptive",
        "cell_list",
        "verlet_cell_list",
        "base",
    ],
    Neighborlist,
]


def _parse_neighborlist(neighborlist: NeighborlistArg = "base") -> Neighborlist:
    if neighborlist == "all_pairs":
        neighborlist = AllPairs()
    elif neighborlist == "cell_list":
        neighborlist = CellList()
    elif neighborlist == "adaptive":
        neighborlist = AdaptiveList()
    elif neighborlist == "base":
        neighborlist = Neighborlist()
    elif neighborlist == "verlet_cell_list":
        neighborlist = VerletCellList()
    elif not isinstance(neighborlist, Neighborlist):
        raise ValueError(f"Unsupported neighborlist: {neighborlist}")
    return tp.cast(Neighborlist, neighborlist)


# Used to check the correctness of the cell and pbc inputs for fn and met that take them
def _validate_inputs(
    cutoff: float,
    species: Tensor,
    coords: Tensor,
    cell: tp.Optional[Tensor],
    pbc: tp.Optional[Tensor],
    supports_batches: bool = True,
    supports_individual_pbc: bool = True,
):
    # No validation if compiling or jit-scripting
    if torch.compiler.is_compiling() or torch.jit.is_scripting():
        return
    if cutoff <= 0.0:
        raise ValueError("Cutoff must be a strictly positive float")
    if not supports_batches and coords.shape[0] != 1:
        raise ValueError("This neighborlist doesn't support batches")

    if pbc is not None:
        if not pbc.any():
            raise ValueError(
                "pbc = torch.tensor([False, False, False]) is not supported anymore"
                " please use pbc = None"
            )
        if cell is None:
            raise ValueError("If pbc is not None, cell should be present")
        if not supports_individual_pbc and not pbc.all():
            raise ValueError(
                "This neighborlist doesn't support PBC only in some directions"
            )
    else:
        if cell is not None:
            raise ValueError("Cell is not supported if not using pbc")


# Wrapper because unique_consecutive doesn't have a dynamic meta kernel asof pytorch 2.5
def _unique_and_counts(sorted_flat_idxs: Tensor) -> tp.Tuple[Tensor, Tensor]:
    # Sort compute unique key
    if torch.compiler.is_compiling():
        return torch.unique(sorted_flat_idxs, return_counts=True)
    return torch.unique_consecutive(sorted_flat_idxs, return_counts=True)


# NOTE: This function is very complex, please read the following carefully
# Input: indices for pairs of atoms that are close to each other. each pair only
# appear once, i.e. only one of the pairs (1, 2) and (2, 1) exists.
# Output: indices for all central atoms and it pairs of neighbors. For example, if
# input has pair
# (0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)
# then the output would have central atom 0, 1, 2, 3, 4 and for cental atom 0, its
# pairs of neighbors are (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)
def neighbors_to_triples(neighbors: Neighbors) -> Triples:
    r"""Converts output of a neighborlist calculation into triples of atoms"""
    # convert representation from pair to central-others and sort
    sorted_flat_idxs, rev_idxs = neighbors.indices.view(-1).sort()
    unique_central_idxs, counts = _unique_and_counts(sorted_flat_idxs)

    # compute central_idxs
    pair_sizes = (counts * (counts - 1)).div(2, rounding_mode="floor")
    pair_indices = torch.repeat_interleave(pair_sizes)
    central_idxs = unique_central_idxs.index_select(0, pair_indices)  # Shape (T,)

    # For each center atom, calculate idxs
    # for all pairs of its 'local, sorted-order' side-atoms
    zcounts = torch.cat((counts.new_zeros(1), counts))
    dev = zcounts.device
    counts_max: int = int(zcounts.max())  # Dynamo can't get past this
    max_local_pairs = torch.tril_indices(counts_max, counts_max, -1, device=dev)
    mask = torch.arange(max_local_pairs.shape[1], device=dev) < pair_sizes.view(-1, 1)
    sort_local_pairs = max_local_pairs.repeat(1, pair_sizes.shape[0])[:, mask.view(-1)]
    sort_local_pairs += torch.cumsum(zcounts, dim=0).index_select(0, pair_indices)

    # Convert back from sorted-idxs to atom-idxs
    local_pairs = rev_idxs[sort_local_pairs]

    # compute mapping between representation of central-other to pair
    num_neigh = neighbors.indices.shape[1]
    # tensor[bool].mul_(2).sub_(1) converts True False -> 1, -1
    # Casting to int8 seems to make this operation a tiny bit faster (not sure why)
    sign12 = (local_pairs < num_neigh).to(torch.int8) * 2 - 1
    side_idxs = local_pairs % num_neigh

    flat_diff_vectors = neighbors.diff_vectors.index_select(0, side_idxs.view(-1))
    diff_vectors = flat_diff_vectors.view(2, -1, 3) * sign12.view(2, -1, 1)  # (2, T, 3)
    distances = neighbors.distances.index_select(0, side_idxs.view(-1)).view(2, -1)
    return Triples(central_idxs, side_idxs, sign12, distances, diff_vectors)


# Reconstruct the shift values used to calculate the diff vectors
def reconstruct_shifts(coords: Tensor, neighbors: Neighbors) -> Tensor:
    r"""Reconstruct shift values used to calculate neighbors"""
    coords0 = coords.view(-1, 3).index_select(0, neighbors.indices[0])
    coords1 = coords.view(-1, 3).index_select(0, neighbors.indices[1])
    unshifted_diff_vectors = coords0 - coords1
    return neighbors.diff_vectors - unshifted_diff_vectors
