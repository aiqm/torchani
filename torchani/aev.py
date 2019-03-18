from __future__ import division
import torch
from . import _six  # noqa:F401
import math
from . import utils
from torch import Tensor
from typing import Tuple
import copy


@torch.jit.script
def _cutoff_cosine(distances, cutoff):
    # type: (Tensor, float) -> Tensor
    return torch.where(
        distances <= cutoff,
        0.5 * torch.cos(math.pi * distances / cutoff) + 0.5,
        torch.zeros_like(distances)
    )


@torch.jit.script
def _radial_subaev_terms(Rcr, EtaR, ShfR, distances):
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
    fc = _cutoff_cosine(distances, Rcr)
    # Note that in the equation in the paper there is no 0.25
    # coefficient, but in NeuroChem there is such a coefficient.
    # We choose to be consistent with NeuroChem instead of the paper here.
    ret = 0.25 * torch.exp(-EtaR * (distances - ShfR)**2) * fc
    # At this point, ret now have shape
    # (conformations, atoms, N, ?, ?) where ? depend on constants.
    # We then should flat the last 4 dimensions to view the subAEV as one
    # dimension vector
    return ret.flatten(start_dim=-2)


@torch.jit.script
def _angular_subaev_terms(Rca, ShfZ, EtaA, Zeta, ShfA, vectors1, vectors2):
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
    vectors1 = vectors1.unsqueeze(
        -1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    vectors2 = vectors2.unsqueeze(
        -1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    distances1 = vectors1.norm(2, dim=-5)
    distances2 = vectors2.norm(2, dim=-5)

    # 0.95 is multiplied to the cos values to prevent acos from
    # returning NaN.
    cos_angles = 0.95 * \
        torch.nn.functional.cosine_similarity(
            vectors1, vectors2, dim=-5)
    angles = torch.acos(cos_angles)

    fcj1 = _cutoff_cosine(distances1, Rca)
    fcj2 = _cutoff_cosine(distances2, Rca)
    factor1 = ((1 + torch.cos(angles - ShfZ)) / 2) ** Zeta
    factor2 = torch.exp(-EtaA *
                        ((distances1 + distances2) / 2 - ShfA) ** 2)
    ret = 2 * factor1 * factor2 * fcj1 * fcj2
    # At this point, ret now have shape
    # (conformations, atoms, N, ?, ?, ?, ?) where ? depend on constants.
    # We then should flat the last 4 dimensions to view the subAEV as one
    # dimension vector
    return ret.flatten(start_dim=-4)


@torch.jit.script
def _combinations(tensor, dim=0):
    # type: (Tensor, int) -> Tuple[Tensor, Tensor]
    n = tensor.shape[dim]
    if n == 0:
        return tensor, tensor
    r = torch.arange(n, dtype=torch.long, device=tensor.device)
    index1, index2 = torch.combinations(r).unbind(-1)
    return tensor.index_select(dim, index1), \
        tensor.index_select(dim, index2)


@torch.jit.script
def _terms_and_indices(Rcr, EtaR, ShfR, Rca, ShfZ, EtaA, Zeta, ShfA,
                       distances, vec):
    """Returns radial and angular subAEV terms, these terms will be sorted
    according to their distances to central atoms, and only these within
    cutoff radius are valid. The returned indices stores the source of data
    before sorting.
    """
    # type: (float, Tensor, Tensor, float, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]  # noqa: E501

    radial_terms = _radial_subaev_terms(Rcr, EtaR,
                                        ShfR, distances)

    vec = _combinations(vec, -2)
    angular_terms = _angular_subaev_terms(Rca, ShfZ, EtaA,
                                          Zeta, ShfA, *vec)

    return radial_terms, angular_terms


@torch.jit.script
def default_neighborlist(species, coordinates, cutoff):
    # type: (Tensor, Tensor, float) -> Tuple[Tensor, Tensor, Tensor]
    """Default neighborlist computer"""

    vec = coordinates.unsqueeze(2) - coordinates.unsqueeze(1)
    # vec has hape (conformations, atoms, atoms, 3) storing Rij vectors

    distances = vec.norm(2, -1)
    # distances has shape (conformations, atoms, atoms) storing Rij distances

    padding_mask = (species == -1).unsqueeze(1)
    distances = distances.masked_fill(padding_mask, math.inf)

    distances, indices = distances.sort(-1)

    min_distances, _ = distances.flatten(end_dim=1).min(0)
    in_cutoff = (min_distances <= cutoff).nonzero().flatten()[1:]
    indices = indices.index_select(-1, in_cutoff)

    # TODO: remove this workaround after gather support broadcasting
    atoms = coordinates.shape[1]
    species_ = species.unsqueeze(1).expand(-1, atoms, -1)
    neighbor_species = species_.gather(-1, indices)

    neighbor_distances = distances.index_select(-1, in_cutoff)

    # TODO: remove this workaround when gather support broadcasting
    # https://github.com/pytorch/pytorch/pull/9532
    indices_ = indices.unsqueeze(-1).expand(-1, -1, -1, 3)
    neighbor_coordinates = vec.gather(-2, indices_)
    return neighbor_species, neighbor_distances, neighbor_coordinates


@torch.jit.script
def _compute_mask_r(species_r, num_species):
    # type: (Tensor, int) -> Tensor
    """Get mask of radial terms for each supported species from indices"""
    mask_r = (species_r.unsqueeze(-1) ==
              torch.arange(num_species, dtype=torch.long,
                           device=species_r.device))
    return mask_r


@torch.jit.script
def _compute_mask_a(species_a, present_species):
    """Get mask of angular terms for each supported species from indices"""
    species_a1, species_a2 = _combinations(species_a, -1)
    mask_a1 = (species_a1.unsqueeze(-1) == present_species).unsqueeze(-1)
    mask_a2 = (species_a2.unsqueeze(-1).unsqueeze(-1) == present_species)
    mask = mask_a1 & mask_a2
    mask_rev = mask.permute(0, 1, 2, 4, 3)
    mask_a = mask | mask_rev
    return mask_a


@torch.jit.script
def _assemble(radial_terms, angular_terms, present_species,
              mask_r, mask_a, num_species, angular_sublength):
    """Returns radial and angular AEV computed from terms according
    to the given partition information.

    Arguments:
        radial_terms (:class:`torch.Tensor`): shape (conformations, atoms,
            neighbors, ``self.radial_sublength()``)
        angular_terms (:class:`torch.Tensor`): shape (conformations, atoms,
            pairs, ``self.angular_sublength()``)
        present_species (:class:`torch.Tensor`):  Long tensor for species
            of atoms present in the molecules.
        mask_r (:class:`torch.Tensor`): shape (conformations, atoms,
            neighbors, supported species)
        mask_a (:class:`torch.Tensor`): shape (conformations, atoms,
            pairs, present species, present species)
    """
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, int, int) -> Tuple[Tensor, Tensor]  # noqa: E501

    conformations = radial_terms.shape[0]
    atoms = radial_terms.shape[1]

    # assemble radial subaev
    present_radial_aevs = (
        radial_terms.unsqueeze(-2) *
        mask_r.unsqueeze(-1).to(radial_terms.dtype)
    ).sum(-3)
    # present_radial_aevs has shape
    # (conformations, atoms, present species, radial_length)
    radial_aevs = present_radial_aevs.flatten(start_dim=2)

    # assemble angular subaev
    rev_indices = torch.full((num_species,), -1, dtype=present_species.dtype,
                             device=present_species.device)
    rev_indices[present_species] = torch.arange(present_species.numel(),
                                                dtype=torch.long,
                                                device=radial_terms.device)
    angular_aevs = []
    zero_angular_subaev = torch.zeros(conformations, atoms, angular_sublength,
                                      dtype=radial_terms.dtype,
                                      device=radial_terms.device)
    for s1 in range(num_species):
        # TODO: make PyTorch support range(start, end) and
        # range(start, end, step) and remove the workaround
        # below. The inner for loop should be:
        # for s2 in range(s1, num_species):
        for s2 in range(num_species - s1):
            s2 += s1
            i1 = int(rev_indices[s1])
            i2 = int(rev_indices[s2])
            if i1 >= 0 and i2 >= 0:
                mask = mask_a[:, :, :, i1, i2].unsqueeze(-1) \
                                              .to(radial_terms.dtype)
                subaev = (angular_terms * mask).sum(-2)
            else:
                subaev = zero_angular_subaev
            angular_aevs.append(subaev)

    return radial_aevs, torch.cat(angular_aevs, dim=2)


@torch.jit.script
def _compute_aev(num_species, angular_sublength, Rcr, EtaR, ShfR, Rca, ShfZ,
                 EtaA, Zeta, ShfA, species, species_, distances, vec):
    # type: (int, int, float, Tensor, Tensor, float, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]  # noqa: E501

    present_species = utils.present_species(species)

    radial_terms, angular_terms = _terms_and_indices(
        Rcr, EtaR, ShfR, Rca, ShfZ, EtaA, Zeta, ShfA, distances, vec)
    mask_r = _compute_mask_r(species_, num_species)
    mask_a = _compute_mask_a(species_, present_species)

    radial, angular = _assemble(radial_terms, angular_terms,
                                present_species, mask_r, mask_a,
                                num_species, angular_sublength)
    fullaev = torch.cat([radial, angular], dim=2)
    return species, fullaev


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
        neighborlist_computer (:class:`collections.abc.Callable`): initial
            value of :attr:`neighborlist`

    Attributes:
        neighborlist (:class:`collections.abc.Callable`): The callable
            (species:Tensor, coordinates:Tensor, cutoff:float)
            -> Tuple[Tensor, Tensor, Tensor] that returns the species,
            distances and relative coordinates of neighbor atoms. The input
            species and coordinates tensor have the same shape convention as
            the input of :class:`AEVComputer`. The returned neighbor
            species and coordinates tensor must have shape ``(C, A, N)`` and
            ``(C, A, N, 3)`` correspoindingly, where ``C`` is the number of
            conformations in a chunk, ``A`` is the number of atoms, and ``N``
            is the maximum number of neighbors that an atom could have.

    .. _ANI paper:
        http://pubs.rsc.org/en/Content/ArticleLanding/2017/SC/C6SC05720A#!divAbstract
    """
    __constants__ = ['Rcr', 'Rca', 'num_species', 'radial_sublength',
                     'radial_length', 'angular_sublength', 'angular_length',
                     'aev_length']

    def __init__(self, Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ,
                 num_species, neighborlist_computer=default_neighborlist):
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

        self.num_species = num_species
        self.neighborlist = neighborlist_computer

        # The length of radial subaev of a single species
        self.radial_sublength = self.EtaR.numel() * self.ShfR.numel()
        # The length of full radial aev
        self.radial_length = self.num_species * self.radial_sublength
        # The length of angular subaev of a single species
        self.angular_sublength = self.EtaA.numel() * self.Zeta.numel() * \
            self.ShfA.numel() * self.ShfZ.numel()
        # The length of full angular aev
        self.angular_length = (self.num_species * (self.num_species + 1)) \
            // 2 * self.angular_sublength
        # The length of full aev
        self.aev_length = self.radial_length + self.angular_length

    def __getstate__(self):
        # FIXME: ScriptModule is not pickable, so deep copy does not work
        # this is a workaround for pickling object of AEVComputer
        d = copy.copy(self.__dict__)
        if d['_modules']['neighborlist'] is default_neighborlist:
            d['_modules']['neighborlist'] = 'default_neighborlist'
        return d

    def __setstate__(self, d):
        # FIXME: ScriptModule is not pickable, so deep copy does not work
        # this is a workaround for pickling object of AEVComputer
        if d['_modules']['neighborlist'] == 'default_neighborlist':
            d['_modules']['neighborlist'] = default_neighborlist
        self.__dict__ = d

    # @torch.jit.script_method
    def forward(self, species_coordinates):
        """Compute AEVs

        Arguments:
            species_coordinates (tuple): Two tensors: species and coordinates.
                species must have shape ``(C, A)`` and coordinates must have
                shape ``(C, A, 3)``, where ``C`` is the number of conformations
                in a chunk, and ``A`` is the number of atoms.

        Returns:
            tuple: Species and AEVs. species are the species from the input
            unchanged, and AEVs is a tensor of shape
            ``(C, A, self.aev_length())``
        """
        # type: (Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]

        species, coordinates = species_coordinates
        max_cutoff = max(self.Rcr, self.Rca)
        species_, distances, vec = self.neighborlist(species, coordinates,
                                                     max_cutoff)
        return _compute_aev(
            self.num_species, self.angular_sublength, self.Rcr, self.EtaR,
            self.ShfR, self.Rca, self.ShfZ, self.EtaA, self.Zeta, self.ShfA,
            species, species_, distances, vec)
