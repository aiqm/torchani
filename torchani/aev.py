import torch
from torch import Tensor
import math
from typing import Tuple, Optional, NamedTuple
from torch.jit import Final


class SpeciesAEV(NamedTuple):
    species: Tensor
    aevs: Tensor


def cutoff_cosine(distances: Tensor, cutoff: float) -> Tensor:
    # assuming all elements in distances are smaller than cutoff
    return 0.5 * torch.cos(distances * (math.pi / cutoff)) + 0.5


def radial_terms(Rcr: float, EtaR: Tensor, ShfR: Tensor, distances: Tensor) -> Tensor:
    """Compute the radial subAEV terms of the center atom given neighbors

    This correspond to equation (3) in the `ANI paper`_. This function just
    compute the terms. The sum in the equation is not computed.
    The input tensor have shape (conformations, atoms, N), where ``N``
    is the number of neighbor atoms within the cutoff radius and output
    tensor should have shape
    (conformations, atoms, ``self.radial_sublength()``)

    .. _ANI paper:
        http://pubs.rsc.org/en/Content/ArticleLanding/2017/SC/C6SC05720A#!divAbstract
    """
    distances = distances.unsqueeze(-1).unsqueeze(-1)
    fc = cutoff_cosine(distances, Rcr)
    # Note that in the equation in the paper there is no 0.25
    # coefficient, but in NeuroChem there is such a coefficient.
    # We choose to be consistent with NeuroChem instead of the paper here.
    ret = 0.25 * torch.exp(-EtaR * (distances - ShfR)**2) * fc
    # At this point, ret now have shape
    # (conformations, atoms, N, ?, ?) where ? depend on constants.
    # We then should flat the last 2 dimensions to view the subAEV as one
    # dimension vector
    return ret.flatten(start_dim=-2)


def angular_terms(Rca: float, ShfZ: Tensor, EtaA: Tensor, Zeta: Tensor,
                  ShfA: Tensor, vectors1: Tensor, vectors2: Tensor) -> Tensor:
    """Compute the angular subAEV terms of the center atom given neighbor pairs.

    This correspond to equation (4) in the `ANI paper`_. This function just
    compute the terms. The sum in the equation is not computed.
    The input tensor have shape (conformations, atoms, N), where N
    is the number of neighbor atom pairs within the cutoff radius and
    output tensor should have shape
    (conformations, atoms, ``self.angular_sublength()``)

    .. _ANI paper:
        http://pubs.rsc.org/en/Content/ArticleLanding/2017/SC/C6SC05720A#!divAbstract
    """
    vectors1 = vectors1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    vectors2 = vectors2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    distances1 = vectors1.norm(2, dim=-5)
    distances2 = vectors2.norm(2, dim=-5)

    # 0.95 is multiplied to the cos values to prevent acos from
    # returning NaN.
    cos_angles = 0.95 * torch.nn.functional.cosine_similarity(vectors1, vectors2, dim=-5)
    angles = torch.acos(cos_angles)

    fcj1 = cutoff_cosine(distances1, Rca)
    fcj2 = cutoff_cosine(distances2, Rca)
    factor1 = ((1 + torch.cos(angles - ShfZ)) / 2) ** Zeta
    factor2 = torch.exp(-EtaA * ((distances1 + distances2) / 2 - ShfA) ** 2)
    ret = 2 * factor1 * factor2 * fcj1 * fcj2
    # At this point, ret now have shape
    # (conformations, atoms, N, ?, ?, ?, ?) where ? depend on constants.
    # We then should flat the last 4 dimensions to view the subAEV as one
    # dimension vector
    return ret.flatten(start_dim=-4)


def compute_shifts(cell: Tensor, pbc: Tensor, cutoff: float) -> Tensor:
    """Compute the shifts of unit cell along the given cell vectors to make it
    large enough to contain all pairs of neighbor atoms with PBC under
    consideration

    Arguments:
        cell (:class:`torch.Tensor`): tensor of shape (3, 3) of the three
        vectors defining unit cell:
            tensor([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]])
        cutoff (float): the cutoff inside which atoms are considered pairs
        pbc (:class:`torch.Tensor`): boolean vector of size 3 storing
            if pbc is enabled for that direction.

    Returns:
        :class:`torch.Tensor`: long tensor of shifts. the center cell and
            symmetric cells are not included.
    """
    reciprocal_cell = cell.inverse().t()
    inv_distances = reciprocal_cell.norm(2, -1)
    num_repeats = torch.ceil(cutoff * inv_distances).to(torch.long)
    num_repeats = torch.where(pbc, num_repeats, torch.zeros_like(num_repeats))
    r1 = torch.arange(1, num_repeats[0] + 1, device=cell.device)
    r2 = torch.arange(1, num_repeats[1] + 1, device=cell.device)
    r3 = torch.arange(1, num_repeats[2] + 1, device=cell.device)
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


def neighbor_pairs(padding_mask: Tensor, coordinates: Tensor, cell: Tensor,
                   shifts: Tensor, cutoff: float) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute pairs of atoms that are neighbors

    Arguments:
        padding_mask (:class:`torch.Tensor`): boolean tensor of shape
            (molecules, atoms) for padding mask. 1 == is padding.
        coordinates (:class:`torch.Tensor`): tensor of shape
            (molecules, atoms, 3) for atom coordinates.
        cell (:class:`torch.Tensor`): tensor of shape (3, 3) of the three vectors
            defining unit cell: tensor([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]])
        cutoff (float): the cutoff inside which atoms are considered pairs
        shifts (:class:`torch.Tensor`): tensor of shape (?, 3) storing shifts
    """
    coordinates = coordinates.detach()
    cell = cell.detach()
    num_atoms = padding_mask.shape[1]
    all_atoms = torch.arange(num_atoms, device=cell.device)

    # Step 2: center cell
    # torch.triu_indices is faster than combinations
    p1_center, p2_center = torch.triu_indices(num_atoms, num_atoms, 1,
                                              device=cell.device).unbind(0)
    shifts_center = shifts.new_zeros((p1_center.shape[0], 3))

    # Step 3: cells with shifts
    # shape convention (shift index, molecule index, atom index, 3)
    num_shifts = shifts.shape[0]
    all_shifts = torch.arange(num_shifts, device=cell.device)
    shift_index, p1, p2 = torch.cartesian_prod(all_shifts, all_atoms, all_atoms).unbind(-1)
    shifts_outide = shifts.index_select(0, shift_index)

    # Step 4: combine results for all cells
    shifts_all = torch.cat([shifts_center, shifts_outide])
    p1_all = torch.cat([p1_center, p1])
    p2_all = torch.cat([p2_center, p2])
    shift_values = shifts_all.to(cell.dtype) @ cell

    # step 5, compute distances, and find all pairs within cutoff
    distances = (coordinates.index_select(1, p1_all) - coordinates.index_select(1, p2_all) + shift_values).norm(2, -1)
    padding_mask = (padding_mask.index_select(1, p1_all)) | (padding_mask.index_select(1, p2_all))
    distances.masked_fill_(padding_mask, math.inf)
    in_cutoff = (distances <= cutoff).nonzero()
    molecule_index, pair_index = in_cutoff.unbind(1)
    molecule_index *= num_atoms
    atom_index1 = p1_all[pair_index]
    atom_index2 = p2_all[pair_index]
    shifts = shifts_all.index_select(0, pair_index)
    return molecule_index + atom_index1, molecule_index + atom_index2, shifts


def neighbor_pairs_nopbc(padding_mask: Tensor, coordinates: Tensor, cell: Tensor,
                         shifts: Tensor, cutoff: float) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute pairs of atoms that are neighbors (doesn't use PBC)

    This function bypasses the calculation of shifts and duplication
    of atoms in order to make calculations faster

    Arguments:
        padding_mask (:class:`torch.Tensor`): boolean tensor of shape
            (molecules, atoms) for padding mask. 1 == is padding.
        coordinates (:class:`torch.Tensor`): tensor of shape
            (molecules, atoms, 3) for atom coordinates.
        cutoff (float): the cutoff inside which atoms are considered pairs
    """
    coordinates = coordinates.detach()
    current_device = coordinates.device
    num_atoms = padding_mask.shape[1]
    p1_all, p2_all = torch.triu_indices(num_atoms, num_atoms, 1,
                                        device=current_device).unbind(0)

    distances = (coordinates.index_select(1, p1_all) - coordinates.index_select(1, p2_all)).norm(2, -1)
    padding_mask = (padding_mask.index_select(1, p1_all)) | (padding_mask.index_select(1, p2_all))
    distances.masked_fill_(padding_mask, math.inf)
    in_cutoff = (distances <= cutoff).nonzero()
    molecule_index, pair_index = in_cutoff.unbind(1)
    molecule_index *= num_atoms
    atom_index1 = p1_all[pair_index] + molecule_index
    atom_index2 = p2_all[pair_index] + molecule_index
    # shifts
    shifts = shifts.new_zeros((p1_all.shape[0], 3)).index_select(0, pair_index)
    return atom_index1, atom_index2, shifts


def triu_index(num_species: int) -> Tensor:
    species1, species2 = torch.triu_indices(num_species, num_species).unbind(0)
    pair_index = torch.arange(species1.shape[0], dtype=torch.long)
    ret = torch.zeros(num_species, num_species, dtype=torch.long)
    ret[species1, species2] = pair_index
    ret[species2, species1] = pair_index
    return ret


def cumsum_from_zero(input_: Tensor) -> Tensor:
    cumsum = torch.cumsum(input_, dim=0)
    cumsum = torch.cat([input_.new_zeros(1), cumsum[:-1]])
    return cumsum


def triple_by_molecule(atom_index1: Tensor, atom_index2: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Input: indices for pairs of atoms that are close to each other.
    each pair only appear once, i.e. only one of the pairs (1, 2) and
    (2, 1) exists.

    Output: indices for all central atoms and it pairs of neighbors. For
    example, if input has pair (0, 1), (0, 2), (0, 3), (0, 4), (1, 2),
    (1, 3), (1, 4), (2, 3), (2, 4), (3, 4), then the output would have
    central atom 0, 1, 2, 3, 4 and for cental atom 0, its pairs of neighbors
    are (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)
    """
    # convert representation from pair to central-others
    ai1 = torch.cat([atom_index1, atom_index2])
    sorted_ai1, rev_indices = ai1.sort()

    # sort and compute unique key
    unique_results = torch.unique_consecutive(sorted_ai1, return_inverse=True, return_counts=True)
    uniqued_central_atom_index = unique_results[0]
    counts = unique_results[-1]

    # compute central_atom_index
    pair_sizes = (counts * (counts - 1) / 2).long()
    pair_indices = torch.repeat_interleave(pair_sizes)
    central_atom_index = uniqued_central_atom_index.index_select(0, pair_indices)

    # do local combinations within unique key, assuming sorted
    m = counts.max().item() if counts.numel() > 0 else 0
    n = pair_sizes.shape[0]
    intra_pair_indices = torch.tril_indices(m, m, -1, device=ai1.device).t().unsqueeze(0).expand(n, -1, -1)
    mask = (torch.arange(intra_pair_indices.shape[1], device=ai1.device) < pair_sizes.unsqueeze(1)).flatten()
    sorted_local_index1, sorted_local_index2 = intra_pair_indices.flatten(0, 1)[mask, :].unbind(-1)
    cumsum = cumsum_from_zero(counts).index_select(0, pair_indices)
    sorted_local_index1 += cumsum
    sorted_local_index2 += cumsum

    # unsort result from last part
    local_index1 = rev_indices[sorted_local_index1]
    local_index2 = rev_indices[sorted_local_index2]

    # compute mapping between representation of central-other to pair
    n = atom_index1.shape[0]
    sign1 = ((local_index1 < n).to(torch.long) * 2) - 1
    sign2 = ((local_index2 < n).to(torch.long) * 2) - 1
    return central_atom_index, local_index1 % n, local_index2 % n, sign1, sign2


def compute_aev(species: Tensor, coordinates: Tensor, cell: Tensor,
                shifts: Tensor, triu_index: Tensor,
                constants: Tuple[float, Tensor, Tensor, float, Tensor, Tensor, Tensor, Tensor],
                sizes: Tuple[int, int, int, int, int]) -> Tensor:
    Rcr, EtaR, ShfR, Rca, ShfZ, EtaA, Zeta, ShfA = constants
    num_species, radial_sublength, radial_length, angular_sublength, angular_length = sizes
    num_molecules = species.shape[0]
    num_atoms = species.shape[1]
    num_species_pairs = angular_length // angular_sublength
    # PBC calculation is bypassed if there are no shifts
    if shifts.numel() == 0:
        atom_index1, atom_index2, shifts = neighbor_pairs_nopbc(species == -1, coordinates, cell, shifts, Rcr)
    else:
        atom_index1, atom_index2, shifts = neighbor_pairs(species == -1, coordinates, cell, shifts, Rcr)
    coordinates = coordinates.flatten(0, 1)
    shift_values = shifts.to(cell.dtype) @ cell
    vec = coordinates.index_select(0, atom_index1) - coordinates.index_select(0, atom_index2) + shift_values
    species = species.flatten()
    species1 = species[atom_index1]
    species2 = species[atom_index2]

    distances = vec.norm(2, -1)

    # compute radial aev
    radial_terms_ = radial_terms(Rcr, EtaR, ShfR, distances)
    radial_aev = radial_terms_.new_zeros((num_molecules * num_atoms * num_species, radial_sublength))
    index1 = atom_index1 * num_species + species2
    index2 = atom_index2 * num_species + species1
    radial_aev.index_add_(0, index1, radial_terms_)
    radial_aev.index_add_(0, index2, radial_terms_)
    radial_aev = radial_aev.reshape(num_molecules, num_atoms, radial_length)

    # Rca is usually much smaller than Rcr, using neighbor list with cutoff=Rcr is a waste of resources
    # Now we will get a smaller neighbor list that only cares about atoms with distances <= Rca
    even_closer_indices = (distances <= Rca).nonzero().flatten()
    atom_index1 = atom_index1.index_select(0, even_closer_indices)
    atom_index2 = atom_index2.index_select(0, even_closer_indices)
    species1 = species1.index_select(0, even_closer_indices)
    species2 = species2.index_select(0, even_closer_indices)
    vec = vec.index_select(0, even_closer_indices)

    # compute angular aev
    central_atom_index, pair_index1, pair_index2, sign1, sign2 = triple_by_molecule(atom_index1, atom_index2)
    vec1 = vec.index_select(0, pair_index1) * sign1.unsqueeze(1).to(vec.dtype)
    vec2 = vec.index_select(0, pair_index2) * sign2.unsqueeze(1).to(vec.dtype)
    species1_ = torch.where(sign1 == 1, species2[pair_index1], species1[pair_index1])
    species2_ = torch.where(sign2 == 1, species2[pair_index2], species1[pair_index2])
    angular_terms_ = angular_terms(Rca, ShfZ, EtaA, Zeta, ShfA, vec1, vec2)
    angular_aev = angular_terms_.new_zeros((num_molecules * num_atoms * num_species_pairs, angular_sublength))
    index = central_atom_index * num_species_pairs + triu_index[species1_, species2_]
    angular_aev.index_add_(0, index, angular_terms_)
    angular_aev = angular_aev.reshape(num_molecules, num_atoms, angular_length)
    return torch.cat([radial_aev, angular_aev], dim=-1)


class AEVComputer(torch.nn.Module):
    r"""The AEV computer that takes coordinates as input and outputs aevs.

    Arguments:
        Rcr (float): :math:`R_C` in equation (2) when used at equation (3)
            in the `ANI paper`_.
        Rca (float): :math:`R_C` in equation (2) when used at equation (4)
            in the `ANI paper`_.
        EtaR (:class:`torch.Tensor`): The 1D tensor of :math:`\eta` in
            equation (3) in the `ANI paper`_.
        ShfR (:class:`torch.Tensor`): The 1D tensor of :math:`R_s` in
            equation (3) in the `ANI paper`_.
        EtaA (:class:`torch.Tensor`): The 1D tensor of :math:`\eta` in
            equation (4) in the `ANI paper`_.
        Zeta (:class:`torch.Tensor`): The 1D tensor of :math:`\zeta` in
            equation (4) in the `ANI paper`_.
        ShfA (:class:`torch.Tensor`): The 1D tensor of :math:`R_s` in
            equation (4) in the `ANI paper`_.
        ShfZ (:class:`torch.Tensor`): The 1D tensor of :math:`\theta_s` in
            equation (4) in the `ANI paper`_.
        num_species (int): Number of supported atom types.

    .. _ANI paper:
        http://pubs.rsc.org/en/Content/ArticleLanding/2017/SC/C6SC05720A#!divAbstract
    """
    Rcr: Final[float]
    Rca: Final[float]
    num_species: Final[int]

    radial_sublength: Final[int]
    radial_length: Final[int]
    angular_sublength: Final[int]
    angular_length: Final[int]
    aev_length: Final[int]
    sizes: Final[Tuple[int, int, int, int, int]]

    def __init__(self, Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species):
        super(AEVComputer, self).__init__()
        self.Rcr = Rcr
        self.Rca = Rca
        assert Rca <= Rcr, "Current implementation of AEVComputer assumes Rca <= Rcr"
        self.num_species = num_species

        # convert constant tensors to a ready-to-broadcast shape
        # shape convension (..., EtaR, ShfR)
        self.register_buffer('EtaR', EtaR.view(-1, 1))
        self.register_buffer('ShfR', ShfR.view(1, -1))
        # shape convension (..., EtaA, Zeta, ShfA, ShfZ)
        self.register_buffer('EtaA', EtaA.view(-1, 1, 1, 1))
        self.register_buffer('Zeta', Zeta.view(1, -1, 1, 1))
        self.register_buffer('ShfA', ShfA.view(1, 1, -1, 1))
        self.register_buffer('ShfZ', ShfZ.view(1, 1, 1, -1))

        # The length of radial subaev of a single species
        self.radial_sublength = self.EtaR.numel() * self.ShfR.numel()
        # The length of full radial aev
        self.radial_length = self.num_species * self.radial_sublength
        # The length of angular subaev of a single species
        self.angular_sublength = self.EtaA.numel() * self.Zeta.numel() * self.ShfA.numel() * self.ShfZ.numel()
        # The length of full angular aev
        self.angular_length = (self.num_species * (self.num_species + 1)) // 2 * self.angular_sublength
        # The length of full aev
        self.aev_length = self.radial_length + self.angular_length
        self.sizes = self.num_species, self.radial_sublength, self.radial_length, self.angular_sublength, self.angular_length

        self.register_buffer('triu_index', triu_index(num_species).to(device=self.EtaR.device))

        # Set up default cell and compute default shifts.
        # These values are used when cell and pbc switch are not given.
        cutoff = max(self.Rcr, self.Rca)
        default_cell = torch.eye(3, dtype=self.EtaR.dtype, device=self.EtaR.device)
        default_pbc = torch.zeros(3, dtype=torch.bool, device=self.EtaR.device)
        default_shifts = compute_shifts(default_cell, default_pbc, cutoff)
        self.register_buffer('default_cell', default_cell)
        self.register_buffer('default_shifts', default_shifts)

    def constants(self):
        return self.Rcr, self.EtaR, self.ShfR, self.Rca, self.ShfZ, self.EtaA, self.Zeta, self.ShfA

    def forward(self, input_: Tuple[Tensor, Tensor],
                cell: Optional[Tensor] = None,
                pbc: Optional[Tensor] = None) -> SpeciesAEV:
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
            unchanged, and AEVs is a tensor of shape ``(N, A, self.aev_length())``
        """
        species, coordinates = input_

        if cell is None and pbc is None:
            cell = self.default_cell
            shifts = self.default_shifts
        else:
            assert (cell is not None and pbc is not None)
            cutoff = max(self.Rcr, self.Rca)
            shifts = compute_shifts(cell, pbc, cutoff)

        aev = compute_aev(species, coordinates, cell, shifts, self.triu_index, self.constants(), self.sizes)
        return SpeciesAEV(species, aev)
