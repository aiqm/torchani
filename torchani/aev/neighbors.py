import torch
from torch import Tensor
from typing import Tuple, Optional, Union
from ..compat import Final


def _parse_neighborlist(neighborlist, cutoff):
    if neighborlist == 'full_pairwise':
        neighborlist = FullPairwise(cutoff)
    else:
        assert isinstance(neighborlist, torch.nn.Module)
    return neighborlist


class BaseNeighborlist(torch.nn.Module):

    cutoff: Final[float]

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
        self.default_cell: Tensor
        self.default_pbc: Tensor

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


class FullPairwise(BaseNeighborlist):

    def __init__(self, cutoff: float):
        """Compute pairs of atoms that are neighbors, uses pbc depending on
        weather pbc.any() is True or not

        Arguments:
            cutoff (float): the cutoff inside which atoms are considered pairs
        """
        super().__init__(cutoff)
        self.register_buffer('default_shift_values', torch.tensor(0.0), persistent=False)
        self.default_shift_values: Tensor

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
        # torch.triu_indices is faster than combinations
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
