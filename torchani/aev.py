import torch
import itertools
import math
from . import utils


def _cutoff_cosine(distances, cutoff):
    """Compute the elementwise cutoff cosine function

    The cutoff cosine function is define in
    https://arxiv.org/pdf/1610.08935.pdf equation 2

    Parameters
    ----------
    distances : torch.Tensor
        The pytorch tensor that stores Rij values. This tensor can
        have any shape since the cutoff cosine function is computed
        elementwise.
    cutoff : float
        The cutoff radius, i.e. the Rc in the equation. For any Rij > Rc,
        the function value is defined to be zero.

    Returns
    -------
    torch.Tensor
        The tensor of the same shape as `distances` that stores the
        computed function values.
    """
    return torch.where(
        distances <= cutoff,
        0.5 * torch.cos(math.pi * distances / cutoff) + 0.5,
        torch.zeros_like(distances)
    )


class AEVComputer(torch.nn.Module):
    """AEV computer

    Attributes
    ----------
    filename : str
        The name of the file that stores constant.
    Rcr, Rca, EtaR, ShfR, Zeta, ShfZ, EtaA, ShfA : torch.Tensor
        Tensor storing constants.
    species : list(str)
        Chemical symbols of supported atom types
    """

    def __init__(self, Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, species):
        super(AEVComputer, self).__init__()
        self.register_buffer('Rcr', Rcr)
        self.register_buffer('Rca', Rca)
        # convert constant tensors to a ready-to-broadcast shape
        # shape convension (..., EtaR, ShfR)
        self.register_buffer('EtaR', EtaR.view(-1, 1))
        self.register_buffer('ShfR', ShfR.view(1, -1))
        # shape convension (..., EtaA, Zeta, ShfA, ShfZ)
        self.register_buffer('EtaA', EtaA.view(-1, 1, 1, 1))
        self.register_buffer('Zeta', Zeta.view(1, -1, 1, 1))
        self.register_buffer('ShfA', ShfA.view(1, 1, -1, 1))
        self.register_buffer('ShfZ', ShfZ.view(1, 1, 1, -1))

        self.species = species

    def radial_sublength(self):
        """Returns the length of radial subaev of a single species"""
        return self.EtaR.numel() * self.ShfR.numel()

    def radial_length(self):
        """Returns the length of full radial aev"""
        return len(self.species) * self.radial_sublength()

    def angular_sublength(self):
        """Returns the length of angular subaev of a single species"""
        return self.EtaA.numel() * self.Zeta.numel() * self.ShfA.numel() * \
            self.ShfZ.numel()

    def angular_length(self):
        """Returns the length of full angular aev"""
        species = len(self.species)
        return int((species * (species + 1)) / 2) * self.angular_sublength()

    def aev_length(self):
        """Returns the length of full aev"""
        return self.radial_length() + self.angular_length()

    def radial_subaev_terms(self, distances):
        """Compute the radial subAEV terms of the center atom given neighbors

        The radial AEV is define in
        https://arxiv.org/pdf/1610.08935.pdf equation 3.
        The sum computed by this method is over all given neighbors,
        so the caller of this method need to select neighbors if the
        caller want a per species subAEV.

        Parameters
        ----------
        distances : torch.Tensor
            Pytorch tensor of shape (..., neighbors) storing the |Rij|
            length where i are the center atoms, and j are their neighbors.

        Returns
        -------
        torch.Tensor
            A tensor of shape (..., neighbors, `radial_sublength`) storing
            the subAEVs.
        """
        distances = distances.unsqueeze(-1).unsqueeze(-1)
        fc = _cutoff_cosine(distances, self.Rcr)
        # Note that in the equation in the paper there is no 0.25
        # coefficient, but in NeuroChem there is such a coefficient.
        # We choose to be consistent with NeuroChem instead of the paper here.
        ret = 0.25 * torch.exp(-self.EtaR * (distances - self.ShfR)**2) * fc
        return ret.flatten(start_dim=-2)

    def angular_subaev_terms(self, vectors1, vectors2):
        """Compute the angular subAEV terms of the center atom given neighbor pairs.

        The angular AEV is define in
        https://arxiv.org/pdf/1610.08935.pdf equation 4.
        The sum computed by this method is over all given neighbor pairs,
        so the caller of this method need to select neighbors if the caller
        want a per species subAEV.

        Parameters
        ----------
        vectors1, vectors2: torch.Tensor
            Tensor of shape (..., pairs, 3) storing the Rij vectors of pairs
            of neighbors. The vectors1(..., j, :) and vectors2(..., j, :) are
            the Rij vectors of the two atoms of pair j.

        Returns
        -------
        torch.Tensor
            Tensor of shape (..., pairs, `angular_sublength`) storing the
            subAEVs.
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
        # ret now have shape (..., pairs, ?, ?, ?, ?) where ? depend on
        # constants

        # flat the last 4 dimensions to view the subAEV as one dimension vector
        return ret.flatten(start_dim=-4)

    def terms_and_indices(self, species, coordinates):
        """Compute radial and angular subAEV terms, and original indices.

        Terms will be sorted according to their distances to central atoms,
        and only these within cutoff radius are valid. The returned indices
        contains what would their original indices be if they were unsorted.

        Parameters
        ----------
        species : torch.Tensor
            The tensor that specifies the species of atoms in the molecule.
            The tensor must have shape (conformations, atoms)
        coordinates : torch.Tensor
            The tensor that specifies the xyz coordinates of atoms in the
            molecule. The tensor must have shape (conformations, atoms, 3)

        Returns
        -------
        (radial_terms, angular_terms, indices_r, indices_a)
        radial_terms : torch.Tensor
            Tensor shaped (conformations, atoms, neighbors, `radial_sublength`)
            for the (unsummed) radial subAEV terms.
        angular_terms : torch.Tensor
            Tensor of shape (conformations, atoms, pairs, `angular_sublength`)
            for the (unsummed) angular subAEV terms.
        indices_r : torch.Tensor
            Tensor of shape (conformations, atoms, neighbors).
            Let l = indices_r(i,j,k), then this means that
            radial_terms(i,j,k,:) is in the subAEV term of conformation i
            between atom j and atom l.
        indices_a : torch.Tensor
            Same as indices_r, except that the cutoff radius is Rca instead of
            Rcr.
        """

        vec = coordinates.unsqueeze(2) - coordinates.unsqueeze(1)
        """Shape (conformations, atoms, atoms, 3) storing Rij vectors"""

        distances = vec.norm(2, -1)
        """Shape (conformations, atoms, atoms) storing Rij distances"""

        padding_mask = (species == -1).unsqueeze(1)
        distances = torch.where(
            padding_mask,
            torch.tensor(math.inf, dtype=self.EtaR.dtype,
                         device=self.EtaR.device),
            distances)

        distances, indices = distances.sort(-1)

        min_distances, _ = distances.flatten(end_dim=1).min(0)
        inRcr = (min_distances <= self.Rcr).nonzero().flatten()[
            1:]  # TODO: can we use something like find_first?
        inRca = (min_distances <= self.Rca).nonzero().flatten()[1:]

        distances = distances.index_select(-1, inRcr)
        indices_r = indices.index_select(-1, inRcr)
        radial_terms = self.radial_subaev_terms(distances)

        indices_a = indices.index_select(-1, inRca)
        new_shape = list(indices_a.shape) + [3]
        # TODO: can we add something like expand_dim(dim=0, repeat=3)
        _indices_a = indices_a.unsqueeze(-1).expand(*new_shape)
        # TODO: can we make gather broadcast??
        vec = vec.gather(-2, _indices_a)
        # TODO: can we move combinations to ATen?
        vec = self.combinations(vec, -2)
        angular_terms = self.angular_subaev_terms(*vec)

        return radial_terms, angular_terms, indices_r, indices_a

    def combinations(self, tensor, dim=0):
        n = tensor.shape[dim]
        r = torch.arange(n, dtype=torch.long, device=tensor.device)
        grid_x, grid_y = torch.meshgrid([r, r])
        index1 = grid_y.masked_select(
            torch.triu(torch.ones(n, n, device=tensor.device),
                       diagonal=1) == 1)
        index2 = grid_x.masked_select(
            torch.triu(torch.ones(n, n, device=tensor.device),
                       diagonal=1) == 1)
        return tensor.index_select(dim, index1), \
            tensor.index_select(dim, index2)

    def compute_mask_r(self, species, indices_r):
        """Partition indices according to their species, radial part

        Parameters
        ----------
        indices_r : torch.Tensor
            Tensor of shape (conformations, atoms, neighbors).
            Let l = indices_r(i,j,k), then this means that
            radial_terms(i,j,k,:) is in the subAEV term of conformation i
            between atom j and atom l.

        Returns
        -------
        torch.Tensor
            Tensor of shape (conformations, atoms, neighbors, all species)
            storing the mask for each species.
        """
        species_r = species.gather(-1, indices_r)
        """Tensor of shape (conformations, atoms, neighbors) storing species
        of neighbors."""
        mask_r = (species_r.unsqueeze(-1) ==
                  torch.arange(len(self.species), device=self.EtaR.device))
        return mask_r

    def compute_mask_a(self, species, indices_a, present_species):
        """Partition indices according to their species, angular part

        Parameters
        ----------
        species_a : torch.Tensor
            Tensor of shape (conformations, atoms, neighbors) storing the
            species of neighbors.
        present_species : torch.Tensor
            Long tensor for the species, already uniqued.

        Returns
        -------
        torch.Tensor
            Tensor of shape (conformations, atoms, pairs, present species,
            present species) storing the mask for each pair.
        """
        species_a = species.gather(-1, indices_a)
        species_a1, species_a2 = self.combinations(species_a, -1)
        mask_a1 = (species_a1.unsqueeze(-1) == present_species).unsqueeze(-1)
        mask_a2 = (species_a2.unsqueeze(-1).unsqueeze(-1) == present_species)
        mask = mask_a1 * mask_a2
        mask_rev = mask.permute(0, 1, 2, 4, 3)
        mask_a = (mask + mask_rev) > 0
        return mask_a

    def assemble(self, radial_terms, angular_terms, present_species,
                 mask_r, mask_a):
        """Assemble radial and angular AEV from computed terms according
        to the given partition information.

        Parameters
        ----------
        radial_terms : torch.Tensor
            Tensor shaped (conformations, atoms, neighbors, `radial_sublength`)
            for the (unsummed) radial subAEV terms.
        angular_terms : torch.Tensor
            Tensor of shape (conformations, atoms, pairs, `angular_sublength`)
            for the (unsummed) angular subAEV terms.
        present_species : torch.Tensor
            Long tensor for species of atoms present in the molecules.
        mask_r : torch.Tensor
            Tensor of shape (conformations, atoms, neighbors, present species)
            storing the mask for each species.
        mask_a : torch.Tensor
            Tensor of shape (conformations, atoms, pairs, present species,
            present species) storing the mask for each pair.

        Returns
        -------
        (torch.Tensor, torch.Tensor)
            Returns (radial AEV, angular AEV), both are pytorch tensor of
            `dtype`. The radial AEV must be of shape (conformations, atoms,
            radial_length) The angular AEV must be of shape (conformations,
            atoms, angular_length)
        """
        conformations = radial_terms.shape[0]
        atoms = radial_terms.shape[1]

        # assemble radial subaev
        present_radial_aevs = (
            radial_terms.unsqueeze(-2) *
            mask_r.unsqueeze(-1).type(radial_terms.dtype)
        ).sum(-3)
        """shape (conformations, atoms, present species, radial_length)"""
        radial_aevs = present_radial_aevs.flatten(start_dim=2)

        # assemble angular subaev
        # TODO: can we use find_first?
        rev_indices = {present_species[i].item(): i
                       for i in range(len(present_species))}
        """shape (conformations, atoms, present species,
                  present species, angular_length)"""
        angular_aevs = []
        zero_angular_subaev = torch.zeros(
            # TODO: can we make stack and cat broadcast?
            conformations, atoms, self.angular_sublength(),
            dtype=self.EtaR.dtype, device=self.EtaR.device)
        for s1, s2 in itertools.combinations_with_replacement(
                                        range(len(self.species)), 2):
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
        species, coordinates = species_coordinates

        present_species = utils.present_species(species)

        # TODO: remove this workaround after gather support broadcasting
        atoms = coordinates.shape[1]
        species_ = species.unsqueeze(1).expand(-1, atoms, -1)

        radial_terms, angular_terms, indices_r, indices_a = \
            self.terms_and_indices(species, coordinates)
        mask_r = self.compute_mask_r(species_, indices_r)
        mask_a = self.compute_mask_a(species_, indices_a, present_species)

        radial, angular = self.assemble(radial_terms, angular_terms,
                                        present_species, mask_r, mask_a)
        fullaev = torch.cat([radial, angular], dim=2)
        return species, fullaev
