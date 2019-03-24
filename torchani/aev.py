from __future__ import division
import torch
from . import _six  # noqa:F401
import math
from . import utils
from torch import Tensor
from typing import Tuple


# @torch.jit.script
def cutoff_cosine(distances, cutoff):
    # type: (Tensor, float) -> Tensor
    return torch.where(
        distances <= cutoff,
        0.5 * torch.cos(math.pi * distances / cutoff) + 0.5,
        torch.zeros_like(distances)
    )


# @torch.jit.script
def radial_terms(Rcr, EtaR, ShfR, distances):
    # type: (float, Tensor, Tensor, Tensor) -> Tensor
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
    # We then should flat the last 4 dimensions to view the subAEV as one
    # dimension vector
    return ret.flatten(start_dim=-2)


# @torch.jit.script
def angular_terms(Rca, ShfZ, EtaA, Zeta, ShfA, vectors1, vectors2):
    # type: (float, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
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
    factor2 = torch.exp(-EtaA * s((distances1 + distances2) / 2 - ShfA) ** 2)
    ret = 2 * factor1 * factor2 * fcj1 * fcj2
    # At this point, ret now have shape
    # (conformations, atoms, N, ?, ?, ?, ?) where ? depend on constants.
    # We then should flat the last 4 dimensions to view the subAEV as one
    # dimension vector
    return ret.flatten(start_dim=-4)


# @torch.jit.script
def compute_shifts(cell, pbc_switch, cutoff):
    """Compute the shifts of unit cell along the given cell vectors to make it
    large enough to contain all pairs of neighbor atoms with PBC under
    consideration

    Arguments:
        cell (:class:`torch.Tensor`): tensor of shape (3, 3) of the three
        vectors defining unit cell:
            tensor([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]])
        cutoff (float): the cutoff inside which atoms are considered pairs
        pbc_switch (:class:`torch.Tensor`): boolean vector of size 3 storing
            if pbc is enabled for that direction.

    Returns:
        :class:`torch.Tensor`: long tensor of shifts. the center cell and
            symmetric cells are not included.
    """
    # type: (Tensor, Tensor, float) -> Tensor
    reciprocal_cell = cell.inverse().t()
    inv_distances = reciprocal_cell.norm(2, -1)
    num_repeats = torch.ceil(cutoff * inv_distances).to(torch.long)
    num_repeats = torch.where(pbc_switch, num_repeats, torch.zeros_like(num_repeats))
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


# @torch.jit.script
def neighbor_pairs(padding_mask, coordinates, cell, shifts, cutoff):
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
    # type: (Tensor, Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor, Tensor, Tensor]

    coordinates = coordinates.detach()
    cell = cell.detach()
    num_atoms = padding_mask.shape[1]
    all_atoms = torch.arange(num_atoms, device=cell.device)

    # Step 2: center cell
    p1_center, p2_center = torch.combinations(all_atoms).unbind(-1)
    shifts_center = shifts.new_zeros(p1_center.shape[0], 3)

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
    shift_values = torch.mm(shifts_all.to(cell.dtype), cell)

    # step 5, compute distances, and find all pairs within cutoff
    distances = (coordinates.index_select(1, p1_all) - coordinates.index_select(1, p2_all) + shift_values).norm(2, -1)
    padding_mask = (padding_mask.index_select(1, p1_all)) | (padding_mask.index_select(1, p2_all))
    distances.masked_fill_(padding_mask, math.inf)
    in_cutoff = (distances <= cutoff).nonzero()
    molecule_index, pair_index = in_cutoff.unbind(1)
    atom_index1 = p1_all[pair_index]
    atom_index2 = p2_all[pair_index]
    shifts = shifts_all.index_select(0, pair_index)
    return molecule_index, atom_index1, atom_index2, shifts


# torch.jit.script
def triu_index(num_species):
    species = torch.arange(num_species)
    species1, species2 = torch.combinations(species, r=2, with_replacement=True).unbind(-1)
    pair_index = torch.arange(species1.shape[0])
    ret = torch.zeros(num_species, num_species, dtype=torch.long)
    ret[species1, species2] = pair_index
    ret[species2, species1] = pair_index
    return ret


# torch.jit.script
def convert_pair_index(index):
    """Let's say we have a pair:
    index: 0 1 2 3 4 5 6 7 8 9 ...
    elem1: 0 0 1 0 1 2 0 1 2 3 ...
    elem2: 1 2 2 3 3 3 4 4 4 4 ...
    This function convert index back to elem1 and elem2

    To implement this, divide it into groups, the first group contains 1
    elements, the second contains 2 elements, ..., the nth group contains
    n elements.

    Let's say we want to compute the elem1 and elem2 for index i. We first find
    the number of complete groups contained in index 0, 1, ..., i - 1
    (all inclusive, not including i), then i will be in the next group. Let's
    say there are N complete groups, then these N groups contains
    N * (N + 1) / 2 elements, solving for the largest N that satisfies
    N * (N + 1) / 2 <= i, will get the N we want.
    """
    n = (torch.sqrt(1.0 + 8.0 * index.to(torch.float)) - 1.0) / 2.0
    n = torch.floor(n).to(torch.long)
    num_elems = n * (n + 1) / 2
    return index - num_elems, n + 1


# torch.jit.script
def triple_by_molecule(molecule_index, atom_index1, atom_index2):
    # convert representation from pair to central-other
    n = molecule_index.shape[0]
    mi = molecule_index.repeat(2)
    ai1 = torch.cat([atom_index1, atom_index2])
    ai2 = torch.cat([atom_index2, atom_index1])

    # sort and compute unique key
    m_ac, rev_indices, counts = torch.stack([mi, ai1], dim=1).unique(dim=0, sorted=True, return_inverse=True, return_counts=True)
    uniqued_molecule_index, uniqued_central_atom_index = m_ac.unbind(1)

    # do local combinations within unique key, assuming sorted
    pair_sizes = counts * (counts - 1) // 2
    total_size = pair_sizes.sum()
    molecule_index = torch.numpy_repeat(uniqued_molecule_index, pair_sizes)
    central_atom_index = torch.numpy_repeat(uniqued_central_atom_index, pair_sizes)
    cumsum = torch.numpy_repeat(torch.cumsum(pair_sizes, dim=0) - pair_sizes[0], pair_sizes)
    sorted_local_pair_index = torch.arange(total_size, device=molecule_index.device) - cumsum
    sorted_local_index1, sorted_local_index2 = convert_pair_index(sorted_local_pair_index)
    sorted_local_index1 += cumsum
    sorted_local_index2 += cumsum

    # unsort result from last part
    argsort = rev_indices.argsort()
    local_index1 = argsort[sorted_local_index1]
    local_index2 = argsort[sorted_local_index2]

    # compute mapping between representation of central-other to pair
    sign1 = torch.where(local_index1 < n, torch.ones_like(local_index1), -torch.ones_like(local_index1))
    sign2 = torch.where(local_index2 < n, torch.ones_like(local_index2), -torch.ones_like(local_index2))
    pair_index1 = torch.where(local_index1 < n, local_index1, local_index1 - n)
    pair_index1 = torch.where(local_index2 < n, local_index2, local_index2 - n)
    return molecule_index, central_atom_index, pair_index1, pair_index2, sign1, sign2


# torch.jit.script
def compute_aev(species, coordinates, cell, pbc_switch, triu_index, constants, sizes):
    Rcr, EtaR, ShfR, Rca, ShfZ, EtaA, Zeta, ShfA = constants
    num_species, radial_sublength, radial_length, angular_sublength, angular_length, aev_length = sizes
    num_molecules = species.shape[0]
    num_atoms = species.shape[1]
    cutoff = max(Rcr, Rca)

    shifts = compute_shifts(cell, pbc_switch, cutoff)
    molecule_index, atom_index1, atom_index2, shifts = neighbor_pairs(species == -1, coordinates, cell, shifts, cutoff)
    species1 = species[molecule_index, atom_index1]
    species2 = species[molecule_index, atom_index2]
    shift_values = torch.mm(shifts.to(cell.dtype), cell)

    vec = coordinates[molecule_index, atom_index1, :] - coordinates[molecule_index, atom_index2, :] + shift_values
    distances = vec.norm(2, -1)

    # compute radial aev
    radial_terms_ = radial_terms(Rcr, EtaR, ShfR, distances)
    radial_aev = radial_terms_.new_zeros(num_molecules, num_atoms, num_species, radial_sublength)
    radial_aev[molecule_index, atom_index1, species2, :] += radial_terms_
    radial_aev[molecule_index, atom_index2, species1, :] += radial_terms_
    radial_aev = radial_aev.reshape(num_molecules, num_atoms, radial_length)

    # compute angular aev
    molecule_index, central_atom_index, pair_index1, pair_index2, sign1, sign2 = triple_by_molecule(molecule_index, atom_index1, atom_index2)
    vec1 = vec.index_select(0, pair_index1) * sign1
    vec2 = vec.index_select(0, pair_index2) * sign2
    species1 = species2[pair_index1]
    species2 = species2[pair_index2]
    angular_terms_ = angular_terms(Rca, ShfZ, EtaA, Zeta, ShfA, vec1, vec2)
    angular_aev = torch.zeros(num_molecules, num_atoms, angular_length // angular_sublength, angular_sublength)
    angular_aev = angular_aev.reshape(num_molecules, num_atoms, angular_length)
    angular_aev[molecule_index, central_atom_index, triu_index[species1, species2]] += angular_terms_
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
    __constants__ = ['Rcr', 'Rca', 'num_species', 'radial_sublength',
                     'radial_length', 'angular_sublength', 'angular_length',
                     'aev_length']

    def __init__(self, Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species):
        super(AEVComputer, self).__init__()
        self.Rcr = Rcr
        self.Rca = Rca
        # convert constant tensors to a ready-to-broadcast shape
        # shape convension (..., EtaR, ShfR)
        self.register_buffer('EtaR', EtaR.view(-1, 1))
        self.register_buffer('ShfR', ShfR.view(1, -1))
        # shape convension (..., EtaA, Zeta, ShfA, ShfZ)
        self.register_buffer('EtaA', EtaA.view(-1, 1, 1, 1))
        self.register_buffer('Zeta', Zeta.view(1, -1, 1, 1))
        self.register_buffer('ShfA', ShfA.view(1, 1, -1, 1))
        self.register_buffer('ShfZ', ShfZ.view(1, 1, 1, -1))
        self.constants = self.Rcr, self.EtaR, self.ShfR, self.Rca, self.ShfZ, self.EtaA, self.Zeta, self.ShfA

        self.num_species = num_species
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
        self.sizes = self.num_species, self.radial_sublength, self.radial_length, self.angular_sublength, self.angular_length, self.aev_length

        self.register_buffer('triu_index', triu_index(num_species))

    # @torch.jit.script_method
    def forward(self, species_coordinates):
        """Compute AEVs

        Arguments:
            species_coordinates (tuple): Two tensors: species and coordinates.
                species must have shape ``(C, A)`` and coordinates must have
                shape ``(C, A, 3)``, where ``C`` is the number of molecules
                in a chunk, and ``A`` is the number of atoms.

        Returns:
            tuple: Species and AEVs. species are the species from the input
            unchanged, and AEVs is a tensor of shape
            ``(C, A, self.aev_length())``
        """
        # type: (Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]
        species, coordinates = species_coordinates
        cell = torch.eye(3, dtype=self.EtaR.dtype, device=self.EtaR.device)
        pbc = torch.zeros(3, dtype=torch.uint8, device=self.EtaR.device)
        return species, compute_aev(species, coordinates, cell, pbc, self.triu_index, self.constants, self.sizes)
