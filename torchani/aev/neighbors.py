from typing import Tuple, Optional, Union

import torch
from torch import Tensor
from torch.nn import functional, Module
from ..utils import map_to_central, cumsum_from_zero
from ..compat import Final


def _parse_neighborlist(neighborlist: Optional[Union[Module, str]], cutoff: float):
    if neighborlist == 'full_pairwise':
        neighborlist = FullPairwise(cutoff)
    elif neighborlist == 'cell_list':
        neighborlist = CellList(cutoff=cutoff)
    elif neighborlist == 'verlet_cell_list':
        neighborlist = CellList(cutoff=cutoff, verlet=True)
    elif neighborlist is None:
        neighborlist = BaseNeighborlist(cutoff)
    else:
        assert isinstance(neighborlist, Module)
    return neighborlist


class BaseNeighborlist(Module):

    cutoff: Final[float]
    default_pbc: Tensor
    default_cell: Tensor

    def __init__(self, cutoff: float):
        """Compute pairs of atoms that are neighbors, uses pbc depending on
        weather pbc.any() is True or not

        Arguments:
            coordinates (:class:`torch.Tensor`): tensor of shape
                (molecules, atoms, 3) for atom coordinates.
            cutoff (float): the cutoff inside which atoms are considered pairs
        """
        super().__init__()
        self.cutoff = cutoff
        self.register_buffer('default_cell', torch.eye(3, dtype=torch.float), persistent=False)
        self.register_buffer('default_pbc', torch.zeros(3, dtype=torch.bool), persistent=False)

    @torch.jit.export
    def _compute_bounding_cell(self, coordinates: Tensor,
                               eps: float) -> Tuple[Tensor, Tensor]:
        # this works but its not needed for this naive implementation
        # This should return a bounding cell
        # for the molecule, in all cases, also it displaces coordinates a fixed
        # value, so that they fit inside the cell completely. This should have
        # no effects on forces or energies

        # add an epsilon to pad due to floating point precision
        min_ = torch.min(coordinates.view(-1, 3), dim=0)[0] - eps
        max_ = torch.max(coordinates.view(-1, 3), dim=0)[0] + eps
        largest_dist = max_ - min_
        coordinates = coordinates - min_
        cell = self.default_cell * largest_dist
        assert (coordinates > 0.0).all()
        assert (coordinates < torch.norm(cell, dim=1)).all()
        return coordinates, cell

    @staticmethod
    def _screen_with_cutoff(cutoff: float, coordinates: Tensor, input_neighborlist: Tensor,
                            shift_values: Optional[Tensor] = None, mask: Optional[Tensor] = None) -> Tuple[Tensor, Union[Tensor, None], Tensor, Tensor]:
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
                mask = mask.view(-1)[input_neighborlist.view(-1)].view(2, -1)
                non_dummy_pairs = (~torch.any(mask, dim=0)).nonzero().flatten()
                input_neighborlist = input_neighborlist.index_select(1, non_dummy_pairs)
                # shift_values can be None when there are no pbc conditions to prevent
                # torch from launching kernels with only zeros
                if shift_values is not None:
                    shift_values = shift_values.index_select(0, non_dummy_pairs)

        coordinates = coordinates.view(-1, 3)
        # Difference vector and distances could be obtained for free when
        # screening, unfortunately distances have to be recalculated twice each
        # time they are screened, since otherwise torch prepares to calculate
        # derivatives of multiple distances that will later be disregarded

        coordinates_ = coordinates.detach()
        # detached calculation #
        coords0 = coordinates_.index_select(0, input_neighborlist[0])
        coords1 = coordinates_.index_select(0, input_neighborlist[1])
        diff_vectors = coords0 - coords1
        if shift_values is not None:
            diff_vectors += shift_values
        distances = diff_vectors.norm(2, -1)
        in_cutoff = (distances <= cutoff).nonzero().flatten()
        # ------------------- #

        screened_neighborlist = input_neighborlist.index_select(1, in_cutoff)
        if shift_values is not None:
            shift_values = shift_values.index_select(0, in_cutoff)

        coords0 = coordinates.index_select(0, screened_neighborlist[0])
        coords1 = coordinates.index_select(0, screened_neighborlist[1])
        screened_diff_vectors = coords0 - coords1
        if shift_values is not None:
            screened_diff_vectors += shift_values
        screened_distances = screened_diff_vectors.norm(2, -1)

        return screened_neighborlist, shift_values, screened_diff_vectors, screened_distances

    @torch.jit.export
    def _recast_long_buffers(self) -> None:
        pass


class FullPairwise(BaseNeighborlist):

    default_shift_values: Tensor

    def __init__(self, cutoff: float):
        """Compute pairs of atoms that are neighbors, uses pbc depending on
        weather pbc.any() is True or not

        Arguments:
            cutoff (float): the cutoff inside which atoms are considered pairs
        """
        super().__init__(cutoff)
        self.register_buffer('default_shift_values', torch.tensor(0.0), persistent=False)

    def forward(self, species: Tensor, coordinates: Tensor, cell: Optional[Tensor] = None,
                pbc: Optional[Tensor] = None) -> Tuple[Tensor, Union[Tensor, None], Tensor, Tensor]:
        """Arguments:
            coordinates (:class:`torch.Tensor`): tensor of shape
                (molecules, atoms, 3) for atom coordinates.
            cell (:class:`torch.Tensor`): tensor of shape (3, 3) of the three vectors
                defining unit cell: tensor([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]])
            cutoff (float): the cutoff inside which atoms are considered pairs
            pbc (:class:`torch.Tensor`): boolean tensor of shape (3,) storing wheather pbc is required
        """
        assert (cell is not None and pbc is not None) or (cell is None and pbc is None)
        cell = cell if cell is not None else self.default_cell
        pbc = pbc if pbc is not None else self.default_pbc

        mask = (species == -1)
        if pbc.any():
            atom_index12, shift_indices = self._full_pairwise_pbc(species, cell, pbc)
            shift_values = shift_indices.to(cell.dtype) @ cell
            # before being screened the coordinates have to be mapped to the
            # central cell in case they are not inside it, this is not necessary
            # if there is no pbc
            coordinates = map_to_central(coordinates, cell, pbc)
            return self._screen_with_cutoff(self.cutoff, coordinates, atom_index12, shift_values, mask)
        else:
            num_molecules = species.shape[0]
            num_atoms = species.shape[1]
            # Create a pairwise neighborlist for all molecules and all atoms,
            # assuming that there are no atoms at all. Dummy species will be
            # screened later
            atom_index12 = torch.triu_indices(num_atoms, num_atoms, 1, device=species.device)
            if num_molecules > 1:
                atom_index12 = atom_index12.unsqueeze(1).repeat(1, num_molecules, 1)
                atom_index12 += num_atoms * torch.arange(num_molecules, device=mask.device).view(1, -1, 1)
                atom_index12 = atom_index12.view(-1).view(2, -1)
            return self._screen_with_cutoff(self.cutoff, coordinates, atom_index12, mask=mask)

    def _full_pairwise_pbc(self, species: Tensor,
                           cell: Tensor, pbc: Tensor) -> Tuple[Tensor, Tensor]:
        cell = cell.detach()
        shifts = self._compute_shifts(cell, pbc)
        num_atoms = species.shape[1]
        all_atoms = torch.arange(num_atoms, device=cell.device)

        # Step 2: center cell
        p12_center = torch.triu_indices(num_atoms,
                                        num_atoms,
                                        1,
                                        device=cell.device)
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

    def _compute_shifts(self, cell: Tensor, pbc: Tensor) -> Tensor:
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
        num_repeats = torch.ceil(self.cutoff * inv_distances).to(torch.long)
        num_repeats = torch.where(pbc, num_repeats, num_repeats.new_zeros(()))
        r1 = torch.arange(1, num_repeats[0].item() + 1, device=cell.device)
        r2 = torch.arange(1, num_repeats[1].item() + 1, device=cell.device)
        r3 = torch.arange(1, num_repeats[2].item() + 1, device=cell.device)
        o = torch.zeros(1, dtype=torch.long, device=cell.device)
        return torch.cat([
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
        ])


class CellList(BaseNeighborlist):

    verlet: Final[bool]
    constant_volume: Final[bool]

    skin: Tensor
    cell_diagonal: Tensor
    cell_inverse: Tensor
    total_buckets: Tensor
    scaling_for_flat_index: Tensor
    shape_buckets_grid: Tensor
    vector_idx_to_flat: Tensor
    translation_cases: Tensor
    vector_index_displacement: Tensor
    translation_displacement_indices: Tensor
    bucket_length_lower_bound: Tensor
    spherical_factor: Tensor

    def __init__(self,
                 cutoff: float,
                 buckets_per_cutoff: int = 1,
                 verlet: bool = False,
                 skin: Optional[float] = None,
                 constant_volume: bool = False):
        super().__init__(cutoff)

        # right now I will only support this, and the extra neighbors are
        # hardcoded, but full support for arbitrary buckets per cutoff is possible
        assert buckets_per_cutoff == 1, "Cell list currently only supports one bucket per cutoff"
        self.constant_volume = constant_volume
        self.verlet = verlet
        self.register_buffer('spherical_factor', torch.full(size=(3, ), fill_value=1.0), persistent=False)
        self.register_buffer('cell_diagonal', torch.zeros(1), persistent=False)
        self.register_buffer('cell_inverse', torch.zeros(1), persistent=False)
        self.register_buffer('total_buckets', torch.zeros(1, dtype=torch.long), persistent=False)
        self.register_buffer('scaling_for_flat_index', torch.zeros(1, dtype=torch.long), persistent=False)
        self.register_buffer('shape_buckets_grid', torch.zeros(1, dtype=torch.long), persistent=False)
        self.register_buffer('vector_idx_to_flat', torch.zeros(1, dtype=torch.long), persistent=False)
        self.register_buffer('translation_cases', torch.zeros(1, dtype=torch.long), persistent=False)
        self.register_buffer('vector_index_displacement', torch.zeros(1, dtype=torch.long), persistent=False)
        self.register_buffer('translation_displacement_indices', torch.zeros(1, dtype=torch.long), persistent=False)
        self.register_buffer('bucket_length_lower_bound', torch.zeros(1), persistent=False)

        if skin is None:
            if verlet:
                # default value for dynamically updated neighborlist
                skin = 1.0
            else:
                # default value for non dynamically updated neighborlist
                skin = 0.0
        self.register_buffer('skin', torch.tensor(skin), persistent=False)

        # only used for verlet option
        self.register_buffer('old_cell_diagonal', torch.zeros(1), persistent=False)
        self.register_buffer('old_shift_indices', torch.zeros(1, dtype=torch.long), persistent=False)
        self.register_buffer('old_atom_pairs', torch.zeros(1, dtype=torch.long), persistent=False)
        self.register_buffer('old_coordinates', torch.zeros(1), persistent=False)

        # buckets_per_cutoff is also the number of buckets that is scanned in
        # each direction. It determines how fine grained the grid is, with
        # respect to the cutoff. This is 2 for amber, but 1 is useful for debug
        self.buckets_per_cutoff = buckets_per_cutoff
        # Here I get the vector index displacements for the neighbors of an
        # arbitrary vector index I think these are enough (this is different
        # from pmemd)
        # I choose all the displacements except for the zero
        # displacement that does nothing, which is the last one
        # hand written order,
        # this order is basically right-to-left, top-to-bottom
        # using the middle buckets (leftmost lower corner + rightmost lower bucket)
        # and the down buckets (all)
        # so this looks like:
        #  x--
        #  xo-
        #  xxx
        # For the middle buckets and
        #  xxx
        #  xxx
        #  xxx
        # for the down buckets

        # NOTE: "0" corresponds to [0, 0, 0], but I don't really need that for
        # vector indices only for translation displacements
        vector_index_displacement = torch.tensor([[-1, 0, 0],  # 1
                                                  [-1, -1, 0],  # 2
                                                  [0, -1, 0],  # 3
                                                  [1, -1, 0],  # 4
                                                  [-1, 1, -1],  # 5
                                                  [0, 1, -1],  # 6
                                                  [1, 1, -1],  # 7
                                                  [-1, 0, -1],  # 8
                                                  [0, 0, -1],  # 9
                                                  [1, 0, -1],  # 10
                                                  [-1, -1, -1],  # 11
                                                  [0, -1, -1],  # 12
                                                  [1, -1, -1]],  # 13
                                                  dtype=torch.long)
        self.vector_index_displacement = vector_index_displacement
        # these are the translation displacement indices, used to displace the
        # image atoms
        assert self.vector_index_displacement.shape == torch.Size([13, 3])

        # I need some extra positions for the translation displacements, in
        # particular, I need some positions for displacements that don't exist
        # inside individual boxes
        extra_translation_displacements = torch.tensor([
            [-1, 1, 0],  # 14
            [0, 1, 0],  # 15
            [1, 1, 0],  # 16
            [1, 0, 0],  # 17
        ], dtype=torch.long)
        translation_displacement_indices = torch.cat((torch.tensor([[0, 0, 0]], dtype=torch.long),
                                                     self.vector_index_displacement,
                                                     extra_translation_displacements), dim=0)
        self.translation_displacement_indices = translation_displacement_indices

        assert self.translation_displacement_indices.shape == torch.Size([18, 3])
        # This is 26 for 2 buckets and 17 for 1 bucket
        # This is necessary for the image - atom map and atom - image map
        self.num_neighbors = len(self.vector_index_displacement)
        # Get the lower bound of the length of a bucket in the bucket grid
        # shape 3, (Bx, By, Bz) The length is cutoff/buckets_per_cutoff +
        # epsilon
        self._register_bucket_length_lower_bound()

        # variables are not set until we have received a cell at least once
        self.cell_variables_are_set = False
        self.old_values_are_cached = False

    @torch.jit.export
    def _recast_long_buffers(self) -> None:
        # for cell list
        self.total_buckets = self.total_buckets.to(dtype=torch.long)
        self.scaling_for_flat_index = self.scaling_for_flat_index.to(dtype=torch.long)
        self.shape_buckets_grid = self.shape_buckets_grid.to(dtype=torch.long)
        self.vector_idx_to_flat = self.vector_idx_to_flat.to(dtype=torch.long)
        self.translation_cases = self.translation_cases.to(dtype=torch.long)
        self.vector_index_displacement = self.vector_index_displacement.to(dtype=torch.long)
        self.translation_displacement_indices = self.translation_displacement_indices.to(dtype=torch.long)

    def forward(self, species: Tensor,
                coordinates: Tensor,
                cell: Optional[Tensor] = None,
                pbc: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor], Tensor, Tensor]:

        assert coordinates.shape[0] == 1, "Cell list doesn't support batches"
        if cell is None:
            assert (pbc is None or not pbc.any())
        # if cell is None then a bounding cell for the molecule is obtained
        # from the coordinates, in this case the coordinates are assumed to be
        # mapped to the central cell, since it is meaningless for the coordinates
        # not to be mapped to the central cell if no pbc is requrested
        pbc = pbc if pbc is not None else self.default_pbc
        assert pbc.all() or (not pbc.any()), "Cell list is only implemented for PBC in all directions or no pbc"

        if cell is None:
            # displaced coordinates are only used for computation if pbc is not required
            coordinates_displaced, cell = self._compute_bounding_cell(coordinates.detach(), eps=1e-3)
        else:
            coordinates_displaced = coordinates.detach()

        if (not self.constant_volume) or (not self.cell_variables_are_set):
            # Cell parameters need to be set only once for constant V simulations,
            # and every time for variable V, (constant P, NPT) simulations
            self._setup_variables(cell.detach())

        if self.verlet and self.old_values_are_cached and (not self._need_new_list(coordinates_displaced.detach())):
            # If a new cell list is not needed use the old cached values
            # IMPORTANT: here cached values should NOT be updated, moving cache
            # to the new step is incorrect
            atom_pairs = self.old_atom_pairs
            shift_indices: Optional[Tensor] = self.old_shift_indices
        else:
            # The cell list is calculated with a skin here. Since coordinates are
            # fractionalized before cell calculation, it is not needed for them to
            # be imaged to the central cell, they can lie outside the cell.
            atom_pairs, shift_indices = self._calculate_cell_list(coordinates_displaced.detach(), pbc)
            # 'Verlet' prevent unnecessary rebuilds of the neighborlist
            if self.verlet:
                self._cache_values(atom_pairs, shift_indices, coordinates_displaced.detach())

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
            return self._screen_with_cutoff(self.cutoff, coordinates, atom_pairs, shift_values, (species == -1))
        else:
            return self._screen_with_cutoff(self.cutoff, coordinates, atom_pairs, mask=(species == -1))

    def _calculate_cell_list(self, coordinates: Tensor, pbc: Tensor) -> Tuple[Tensor, Union[Tensor, None]]:
        # 1) Fractionalize coordinates
        fractional_coordinates = self._fractionalize_coordinates(coordinates)

        # 2) Get vector indices and flattened indices for atoms in unit cell
        # shape C x A x 3 this gives \vb{g}(a), the vector bucket idx
        # shape C x A this gives f(a), the flat bucket idx for atom a
        atom_vector_index, atom_flat_index = self._get_bucket_indices(fractional_coordinates)

        # 3) get image_indices -> atom_indices and inverse mapping
        # NOTE: there is not necessarily a requirement to do this here
        # both shape (A,); a(i) and i(a)
        # NOTE: watch out, since sorting is not stable this may scramble the atoms
        # in the same box, so that the atidx you get after applying
        # atidx_from_imidx[something] will not be the correct order
        # since what we want is the pairs this is fine, pairs are agnostic to
        # species.
        imidx_from_atidx, atidx_from_imidx = self._get_imidx_converters(atom_flat_index)

        # FIRST WE WANT "WITHIN" IMAGE PAIRS
        # 1) Get the number of atoms in each bucket (as indexed with f idx)
        # this gives A*, A(f) , "A(f' <= f)" = Ac(f) (cumulative) f being the
        # flat bucket index, A being the number of atoms for that bucket,
        # and Ac being the cumulative number of atoms up to that bucket
        out_in_flat_bucket = self._get_atoms_in_flat_bucket_counts(atom_flat_index)
        flat_bucket_count, flat_bucket_cumcount, max_in_bucket = out_in_flat_bucket

        # 2) this are indices WITHIN the central buckets
        within_image_pairs = self._get_within_image_pairs(flat_bucket_count, flat_bucket_cumcount, max_in_bucket)

        # NOW WE WANT "BETWEEN" IMAGE PAIRS
        # 1) Get the vector indices of all (pure) neighbors of each atom
        # this gives \vb{g}(a, n) and f(a, n)
        # shapes 1 x A x Eta x 3 and 1 x A x Eta respectively
        neighbor_vector_indices, neighbor_flat_indices = self._get_neighbor_indices(atom_vector_index)

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
        neighbor_count = flat_bucket_count[neighbor_flat_indices]
        neighbor_cumcount = flat_bucket_cumcount[neighbor_flat_indices]
        neighbor_translation_types = self._get_neighbor_translation_types(neighbor_vector_indices)
        lower, between_pairs_translation_types = self._get_lower_between_image_pairs(neighbor_count,
                                                                               neighbor_cumcount,
                                                                               max_in_bucket,
                                                                               neighbor_translation_types)
        neighborhood_count = neighbor_count.sum(-1).squeeze()
        upper = torch.repeat_interleave(imidx_from_atidx.squeeze(), neighborhood_count)
        assert lower.shape == upper.shape
        between_image_pairs = torch.stack((upper, lower), dim=0)

        if not pbc.any():
            # select only the pairs that don't need any translation
            non_pbc_pairs = (between_pairs_translation_types == 0).nonzero().flatten()
            between_image_pairs = between_image_pairs.index_select(1, non_pbc_pairs)
            shift_indices = None
        else:
            between_pairs_shift_indices = self.translation_displacement_indices.index_select(0, between_pairs_translation_types)
            assert between_pairs_shift_indices.shape[-1] == 3
            within_pairs_shift_indices = torch.zeros(len(within_image_pairs[0]),
                                                3,
                                                device=between_pairs_shift_indices.device, dtype=torch.long)
            # -1 is necessary to ensure correct shifts
            shift_indices = -torch.cat((between_pairs_shift_indices, within_pairs_shift_indices), dim=0)

        # concatenate within and between
        image_pairs = torch.cat((between_image_pairs, within_image_pairs), dim=1)
        atom_pairs = atidx_from_imidx[image_pairs]

        return atom_pairs, shift_indices

    def _setup_variables(self, cell: Tensor):
        current_device = cell.device
        # 1) Update the cell diagonal and translation displacements
        # sizes of each side are given by norm of each basis vector of the unit cell
        self.cell_diagonal = torch.linalg.norm(cell, dim=0)
        self.cell_inverse = torch.inverse(cell)

        # 2) Get max bucket index (Gx, Gy, Gz)
        # which give the size of the grid of buckets that fully covers the
        # whole volume of the unit cell U, given by "cell", and the number of
        # flat buckets (F) (equal to the total number of buckets, F )
        #
        # Gx, Gy, Gz is 1 + maximum index for vector g. Flat bucket indices are
        # indices for the buckets written in row major order (or equivalently
        # dictionary order), the number F = Gx * Gy * Gz

        # bucket_length_lower_bound = B, unit cell U_mu = B * 3 - epsilon this
        # means I can cover it with 3 buckets plus some extra space that is
        # less than a bucket, so I just stretch the buckets a little bit. In
        # this particular case shape_buckets_grid = [3, 3, 3]
        self.shape_buckets_grid = torch.div(
            self.cell_diagonal, self.bucket_length_lower_bound, rounding_mode='floor').to(torch.long)

        self.total_buckets = self.shape_buckets_grid.prod()
        if self.total_buckets == 0:
            raise RuntimeError("Cell is too small to perform pbc calculations")

        # 3) This is needed to scale and flatten last dimension of bucket indices
        # for row major this is (Gy * Gz, Gz, 1)
        self.scaling_for_flat_index = torch.ones(3,
                                                 dtype=torch.long,
                                                 device=current_device)
        self.scaling_for_flat_index[
            0] *= self.shape_buckets_grid[1] * self.shape_buckets_grid[2]
        self.scaling_for_flat_index[1] *= self.shape_buckets_grid[2]

        # 4) create the vector_index -> flat_index conversion tensor
        # it is not really necessary to perform circular padding,
        # since we can index the array using negative indices!
        vector_idx_to_flat = torch.arange(0, self.total_buckets.item(),
                                          device=current_device)
        vector_idx_to_flat = vector_idx_to_flat.view(
            int(self.shape_buckets_grid[0]),
            int(self.shape_buckets_grid[1]),
            int(self.shape_buckets_grid[2]))
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
        x = functional.pad(x, (1, 1, 1, 1, 1, 1), mode='circular')
        return x.squeeze()

    def _register_bucket_length_lower_bound(self,
                                            extra_space: float = 1e-5):
        # Get the size (Bx, By, Bz) of the buckets in the grid.
        # extra space by default is consistent with Amber
        #
        # The spherical factor is different from 1 in the case of nonorthogonal
        # boxes and accounts for the "spherical protrusion", which is related
        # to the fact that the sphere of radius "cutoff" around an atom needs
        # some extra space in nonorthogonal boxes.
        #
        # note that this is not actually the bucket length used in the grid,
        # it is only a lower bound used to calculate the grid size
        spherical_factor = self.spherical_factor
        self.bucket_length_lower_bound = (spherical_factor * self.cutoff / self.buckets_per_cutoff) + extra_space

    def _to_flat_index(self, x: Tensor) -> Tensor:
        # Converts a tensor with bucket indices in the last dimension to a
        # tensor with flat bucket indices in the last dimension.
        #
        # If your tensor is (N1, ..., Nd, 3) this transforms the tensor into
        # (N1, ..., Nd), which holds the flat bucket indices this can be done a
        # different way, same as between but this is possibly faster (?) NOTE:
        # should benchmark this or simplify the code
        assert self.scaling_for_flat_index is not None,\
            "Scaling for flat index has not been computed"
        assert x.shape[-1] == 3
        return (x * self.scaling_for_flat_index).sum(-1)

    def _expand_into_neighbors(self, x: Tensor) -> Tensor:
        # transforms a tensor of shape (... 3) with vector indices
        # in the last dimension into a tensor of shape (..., Eta, 3)
        # where Eta is the number of neighboring buckets, indexed by n
        assert self.vector_index_displacement is not None,\
            "Displacement for neighbors has not been computed"
        assert x.shape[-1] == 3
        x = x.unsqueeze(-2) + self.vector_index_displacement
        # sanity check
        assert x.shape[-1] == 3
        assert x.shape[-2] == self.vector_index_displacement.shape[0]
        return x

    def _fractionalize_coordinates(self, coordinates: Tensor) -> Tensor:
        # Scale coordinates to box size
        #
        # Make all coordinates relative to the box size. This means for
        # instance that if the coordinate is 3.15 times the cell length, it is
        # turned into 3.15; if it is 0.15 times the cell length, it is turned
        # into 0.15, etc
        fractional_coordinates = torch.matmul(coordinates, self.cell_inverse)
        # this is done to account for possible coordinates outside the box,
        # which amber does, in order to calculate diffusion coefficients, etc
        fractional_coordinates -= fractional_coordinates.floor()
        # fractional_coordinates should be in the range [0, 1.0)
        fractional_coordinates[fractional_coordinates >= 1.0] += -1.0
        fractional_coordinates[fractional_coordinates < 0.0] += 1.0

        assert not torch.isnan(fractional_coordinates).any(),\
                "Some fractional coordinates are NaN."
        assert not torch.isinf(fractional_coordinates).any(),\
                "Some fractional coordinates are +-Inf."
        assert (fractional_coordinates < 1.0).all(),\
            f"Some fractional coordinates are too large {fractional_coordinates[fractional_coordinates >= 1.]}"
        assert (fractional_coordinates >= 0.0).all(),\
            f"Some coordinates are too small {fractional_coordinates.masked_select(fractional_coordinates < 0.)}"
        return fractional_coordinates

    def _fractional_to_vector_bucket_indices(self, fractional: Tensor) -> Tensor:
        # transforms a tensor of fractional coordinates (shape (..., 3))
        # into a tensor of vector bucket indices (same shape)
        # Since the number of indices to iterate over is a cartesian product of 3
        # vectors it will grow with L^3 (volume) I think it is intelligent to first
        # get all indices and then apply as needed I will call the buckets "main
        # bucket" and "neighboring buckets"
        assert self.shape_buckets_grid is not None, "shape_buckets_grid not computed"
        out = torch.floor(fractional * (self.shape_buckets_grid).view(1, 1, -1))
        out = out.to(torch.long)
        return out

    @staticmethod
    def _get_imidx_converters(x: Tensor) -> Tuple[Tensor, Tensor]:
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

        # imidx_from_atidx returns tensors that convert image indices into atom
        # indices and viceversa
        # move to device necessary? not sure
        atidx_from_imidx = torch.argsort(x.squeeze()).to(x.device)
        imidx_from_atidx = torch.argsort(atidx_from_imidx).to(x.device)
        return imidx_from_atidx, atidx_from_imidx

    def _get_atoms_in_flat_bucket_counts(
            self, atom_flat_index: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # NOTE: count in flat bucket: 3 0 0 0 ... 2 0 0 0 ... 1 0 1 0 ...,
        # shape is total buckets F cumulative buckets count has the number of
        # atoms BEFORE a given bucket cumulative buckets count: 0 3 3 3 ... 3 5
        # 5 5 ... 5 6 6 7 ...
        atom_flat_index = atom_flat_index.squeeze()
        flat_bucket_count = torch.bincount(atom_flat_index,
                                           minlength=int(self.total_buckets)).to(
                                               atom_flat_index.device)
        flat_bucket_cumcount = cumsum_from_zero(flat_bucket_count).to(
            atom_flat_index.device)

        # this is A*
        max_in_bucket = flat_bucket_count.max()
        return flat_bucket_count, flat_bucket_cumcount, max_in_bucket

    def _get_within_image_pairs(self, flat_bucket_count: Tensor,
                                flat_bucket_cumcount: Tensor,
                                max_in_bucket: Tensor) -> Tensor:
        # note: in this function wpairs == "with_pairs"
        # max_in_bucket = maximum number of atoms contained in any bucket
        current_device = flat_bucket_count.device

        # get all indices f that have pairs inside
        # these are A(w) and Ac(w), and wpairs_flat_index is actually f(w)
        wpairs_flat_index = (flat_bucket_count > 1).nonzero().squeeze()
        wpairs_flat_bucket_count = flat_bucket_count.index_select(0, wpairs_flat_index)
        wpairs_flat_bucket_cumcount = flat_bucket_cumcount.index_select(0, wpairs_flat_index)

        padded_pairs = torch.triu_indices(int(max_in_bucket),
                                          int(max_in_bucket),
                                          offset=1,
                                          device=current_device)
        # sort along first row
        padded_pairs = padded_pairs.index_select(1, torch.argsort(padded_pairs[1]))
        # shape (2, pairs) + shape (wpairs, 1, 1) = shape (wpairs, 2, pairs)

        # basically this repeats the padded pairs "wpairs" times and adds to
        # all of them the cumulative counts, then we unravel all pairs, which
        # remain in the correct order in the second row (the order within same
        # numbers in the first row is actually not essential)
        padded_pairs = padded_pairs.view(2, 1, -1) + wpairs_flat_bucket_cumcount.view(1, -1, 1)
        padded_pairs = padded_pairs.view(2, -1)

        # NOTE this code is very confusing, it could probably use some comments / simplification
        # NOTE: JIT bug makes it so that we have to add the "one" tensor here
        one = torch.tensor(1, device=current_device, dtype=torch.long)
        max_pairs_in_bucket = (max_in_bucket * (max_in_bucket - one)).div_(2, rounding_mode='floor')
        wpairs_count_pairs = (wpairs_flat_bucket_count * (wpairs_flat_bucket_count - one)).div_(2, rounding_mode='floor')

        mask = torch.arange(0, int(max_pairs_in_bucket), device=current_device)
        mask = mask.expand(wpairs_flat_index.numel(), -1)
        mask = (mask < wpairs_count_pairs.view(-1, 1)).view(-1)
        within_image_pairs = padded_pairs.index_select(1, mask.nonzero().squeeze())
        return within_image_pairs

    def _get_lower_between_image_pairs(
            self, neighbor_count: Tensor, neighbor_cumcount: Tensor,
            max_in_bucket: Tensor,
            neighbor_translation_types: Tensor) -> Tuple[Tensor, Tensor]:
        # neighbor_translation_types has shape 1 x At x Eta
        # 3) now I need the LOWER part
        # this gives, for each atom, for each neighbor bucket, all the
        # unpadded, unshifted atom neighbors
        # this is basically broadcasted to the shape of fna
        # shape is 1 x A x eta x A*
        atoms = neighbor_count.shape[1]
        padded_atom_neighbors = torch.arange(0,
                                             int(max_in_bucket),
                                             device=neighbor_count.device)
        padded_atom_neighbors = padded_atom_neighbors.view(1, 1, 1, -1)
        # repeat is needed instead of expand here due to += neighbor_cumcount
        padded_atom_neighbors = padded_atom_neighbors.repeat(
            1, atoms, self.num_neighbors, 1)

        # repeat the neighbor translation types to account for all neighboring atoms
        # repeat is needed instead of expand due to reshaping later
        neighbor_translation_types = neighbor_translation_types.unsqueeze(
            -1).repeat(1, 1, 1, padded_atom_neighbors.shape[-1])

        # now I need to add A(f' < fna) shift the padded atom neighbors to get
        # image indices I need to check here that the cumcount is correct since
        # it was technically done with imidx so I need to check correctnes of
        # both counting schemes, but first I create the mask to unpad
        # and then I shift to the correct indices
        mask = (padded_atom_neighbors < neighbor_count.unsqueeze(-1))
        padded_atom_neighbors.add_(neighbor_cumcount.unsqueeze(-1))
        # the mask should have the same shape as padded_atom_neighbors, and
        # now all that is left is to apply the mask in order to unpad
        assert padded_atom_neighbors.shape == mask.shape
        assert neighbor_translation_types.shape == mask.shape
        # the following calls are equivalent to masked select but significantly faster,
        # since masked_select is slow
        # y = torch.masked_select(x, mask) == y = x.view(-1).index_select(0, mask.view(-1).nonzero().squeeze())
        lower = padded_atom_neighbors.view(-1).index_select(0, mask.view(-1).nonzero().squeeze())
        between_pairs_translation_types = neighbor_translation_types.view(-1)\
                                          .index_select(0, mask.view(-1).nonzero().squeeze())
        return lower, between_pairs_translation_types

    def _get_bucket_indices(self, fractional_coordinates: Tensor) -> Tuple[Tensor, Tensor]:
        atom_vector_index = self._fractional_to_vector_bucket_indices(fractional_coordinates)
        atom_flat_index = self._to_flat_index(atom_vector_index)
        return atom_vector_index, atom_flat_index

    def _get_neighbor_indices(
            self, atom_vector_index: Tensor) -> Tuple[Tensor, Tensor]:
        current_device = atom_vector_index.device
        # This is actually pure neighbors, so it doesn't have
        # "the bucket itself"
        # These are
        # - g(a, n),  shape 1 x A x Eta x 3
        # - f(a, n),  shape 1 x A x Eta
        # These give, for each atom, the flat index or the vector index of its
        # neighbor buckets (neighbor buckets indexed by n).
        neighbor_vector_indices = self._expand_into_neighbors(
            atom_vector_index)
        # these vector indices have the information that says whether to shift
        # each pair and what amount to shift it

        atoms = neighbor_vector_indices.shape[1]
        neighbors = neighbor_vector_indices.shape[2]

        neighbor_vector_indices.add_(torch.ones(1, dtype=torch.long, device=current_device))
        neighbor_vector_indices = neighbor_vector_indices.view(-1, 3)
        # NOTE: This is needed instead of unbind due to torchscript bug
        neighbor_flat_indices = self.vector_idx_to_flat[
            neighbor_vector_indices[:, 0], neighbor_vector_indices[:, 1],
            neighbor_vector_indices[:, 2]]

        neighbor_flat_indices = neighbor_flat_indices.view(
            1, atoms, neighbors)
        neighbor_vector_indices = neighbor_vector_indices.view(1, atoms, neighbors, 3)
        return neighbor_vector_indices, neighbor_flat_indices

    def _get_neighbor_translation_types(self, neighbor_vector_indices: Tensor) -> Tensor:
        atoms = neighbor_vector_indices.shape[1]
        neighbors = neighbor_vector_indices.shape[2]
        neighbor_vector_indices = neighbor_vector_indices.view(-1, 3)
        neighbor_translation_types = self.translation_cases[
            neighbor_vector_indices[:, 0], neighbor_vector_indices[:, 1],
            neighbor_vector_indices[:, 2]]
        neighbor_translation_types = neighbor_translation_types.view(
            1, atoms, neighbors)
        return neighbor_translation_types

    def _cache_values(self, atom_pairs: Tensor,
                            shift_indices: Optional[Tensor],
                            coordinates: Tensor):

        self.old_atom_pairs = atom_pairs.detach()
        if shift_indices is not None:
            self.old_shift_indices = shift_indices.detach()
        self.old_coordinates = coordinates.detach()
        self.old_cell_diagonal = self.cell_diagonal.detach()
        self.old_values_are_cached = True

    def reset_cached_values(self) -> None:
        float_dtype = self.cell_diagonal.dtype
        device = self.cell_diagonal.device
        self._cache_values(torch.zeros(1, dtype=torch.long, device=device),
                           torch.zeros(1, dtype=torch.long, device=device),
                           torch.zeros(1, dtype=float_dtype, device=device))
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
