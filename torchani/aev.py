import torch
import itertools
import math
from . import utils


def _cutoff_cosine(distances, cutoff):
    return torch.where(
        distances <= cutoff,
        0.5 * torch.cos(math.pi * distances / cutoff) + 0.5,
        torch.zeros_like(distances)
    )


def default_neighborlist(species, coordinates, cutoff):
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

    def radial_sublength(self):
        """Returns the length of radial subaev of a single species"""
        return self.EtaR.numel() * self.ShfR.numel()

    def radial_length(self):
        """Returns the length of full radial aev"""
        return self.num_species * self.radial_sublength()

    def angular_sublength(self):
        """Returns the length of angular subaev of a single species"""
        return self.EtaA.numel() * self.Zeta.numel() * self.ShfA.numel() * \
            self.ShfZ.numel()

    def angular_length(self):
        """Returns the length of full angular aev"""
        s = self.num_species
        return (s * (s + 1)) // 2 * self.angular_sublength()

    def aev_length(self):
        """Returns the length of full aev"""
        return self.radial_length() + self.angular_length()

    def _radial_subaev_terms(self, distances):
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
        fc = _cutoff_cosine(distances, self.Rcr)
        # Note that in the equation in the paper there is no 0.25
        # coefficient, but in NeuroChem there is such a coefficient.
        # We choose to be consistent with NeuroChem instead of the paper here.
        ret = 0.25 * torch.exp(-self.EtaR * (distances - self.ShfR)**2) * fc
        # At this point, ret now have shape
        # (conformations, atoms, N, ?, ?) where ? depend on constants.
        # We then should flat the last 4 dimensions to view the subAEV as one
        # dimension vector
        return ret.flatten(start_dim=-2)

    def _angular_subaev_terms(self, vectors1, vectors2):
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

        fcj1 = _cutoff_cosine(distances1, self.Rca)
        fcj2 = _cutoff_cosine(distances2, self.Rca)
        factor1 = ((1 + torch.cos(angles - self.ShfZ)) / 2) ** self.Zeta
        factor2 = torch.exp(-self.EtaA *
                            ((distances1 + distances2) / 2 - self.ShfA) ** 2)
        ret = 2 * factor1 * factor2 * fcj1 * fcj2
        # At this point, ret now have shape
        # (conformations, atoms, N, ?, ?, ?, ?) where ? depend on constants.
        # We then should flat the last 4 dimensions to view the subAEV as one
        # dimension vector
        return ret.flatten(start_dim=-4)

    def _terms_and_indices(self, species, coordinates):
        """Returns radial and angular subAEV terms, these terms will be sorted
        according to their distances to central atoms, and only these within
        cutoff radius are valid. The returned indices stores the source of data
        before sorting.
        """
        max_cutoff = max([self.Rcr, self.Rca])
        species_, distances, vec = self.neighborlist(species, coordinates,
                                                     max_cutoff)
        radial_terms = self._radial_subaev_terms(distances)

        vec = self._combinations(vec, -2)
        angular_terms = self._angular_subaev_terms(*vec)

        # Returned tensors has shape:
        # (conformations, atoms, neighbors, ``self.radial_sublength()``)
        # (conformations, atoms, pairs, ``self.angular_sublength()``)
        # (conformations, atoms, neighbors)
        # (conformations, atoms, pairs)
        return radial_terms, angular_terms, species_

    def _combinations(self, tensor, dim=0):
        n = tensor.shape[dim]
        if n == 0:
            return tensor, tensor
        r = torch.arange(n, dtype=torch.long, device=tensor.device)
        index1, index2 = torch.combinations(r).unbind(-1)
        return tensor.index_select(dim, index1), \
            tensor.index_select(dim, index2)

    def _compute_mask_r(self, species_r):
        """Get mask of radial terms for each supported species from indices"""
        mask_r = (species_r.unsqueeze(-1) ==
                  torch.arange(self.num_species, device=self.EtaR.device))
        return mask_r

    def _compute_mask_a(self, species_a, present_species):
        """Get mask of angular terms for each supported species from indices"""
        species_a1, species_a2 = self._combinations(species_a, -1)
        mask_a1 = (species_a1.unsqueeze(-1) == present_species).unsqueeze(-1)
        mask_a2 = (species_a2.unsqueeze(-1).unsqueeze(-1) == present_species)
        mask = mask_a1 * mask_a2
        mask_rev = mask.permute(0, 1, 2, 4, 3)
        mask_a = (mask + mask_rev) > 0
        return mask_a

    def _assemble(self, radial_terms, angular_terms, present_species,
                  mask_r, mask_a):
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
        conformations = radial_terms.shape[0]
        atoms = radial_terms.shape[1]

        # assemble radial subaev
        present_radial_aevs = (
            radial_terms.unsqueeze(-2) *
            mask_r.unsqueeze(-1).type(radial_terms.dtype)
        ).sum(-3)
        # present_radial_aevs has shape
        # (conformations, atoms, present species, radial_length)
        radial_aevs = present_radial_aevs.flatten(start_dim=2)

        # assemble angular subaev
        rev_indices = {present_species[i].item(): i
                       for i in range(len(present_species))}
        angular_aevs = []
        zero_angular_subaev = torch.zeros(
            conformations, atoms, self.angular_sublength(),
            dtype=self.EtaR.dtype, device=self.EtaR.device)
        for s1, s2 in itertools.combinations_with_replacement(
                                        range(self.num_species), 2):
            if s1 in rev_indices and s2 in rev_indices:
                i1 = rev_indices[s1]
                i2 = rev_indices[s2]
                mask = mask_a[..., i1, i2].unsqueeze(-1).type(self.EtaR.dtype)
                subaev = (angular_terms * mask).sum(-2)
            else:
                subaev = zero_angular_subaev
            angular_aevs.append(subaev)

        return radial_aevs, torch.cat(angular_aevs, dim=2)

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
        species, coordinates = species_coordinates

        present_species = utils.present_species(species)

        radial_terms, angular_terms, species_ = \
            self._terms_and_indices(species, coordinates)
        mask_r = self._compute_mask_r(species_)
        mask_a = self._compute_mask_a(species_, present_species)

        radial, angular = self._assemble(radial_terms, angular_terms,
                                         present_species, mask_r, mask_a)
        fullaev = torch.cat([radial, angular], dim=2)
        return species, fullaev
