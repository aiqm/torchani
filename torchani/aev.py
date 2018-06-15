import torch
import itertools
import numpy
from .aev_base import AEVComputer
from . import buildin_const_file, default_dtype, default_device
from . import _utils


def _cpu(x):
    # TODO: remove this when pytorch support `torch.unique` on GPU
    return x.to('cpu')


def _cutoff_cosine(distances, cutoff):
    """Compute the elementwise cutoff cosine function

    The cutoff cosine function is define in https://arxiv.org/pdf/1610.08935.pdf equation 2

    Parameters
    ----------
    distances : torch.Tensor
        The pytorch tensor that stores Rij values. This tensor can have any shape since the cutoff
        cosine function is computed elementwise.
    cutoff : float
        The cutoff radius, i.e. the Rc in the equation. For any Rij > Rc, the function value is defined to be zero.

    Returns
    -------
    torch.Tensor
        The tensor of the same shape as `distances` that stores the computed function values.
    """
    return torch.where(distances <= cutoff, 0.5 * torch.cos(numpy.pi * distances / cutoff) + 0.5, torch.zeros_like(distances))


class SortedAEV(AEVComputer):
    """The AEV computer assuming input coordinates sorted by species

    Attributes
    ----------
    timers : dict
        Dictionary storing the the benchmark result. It has the following keys:
            radial_subaev : time spent on computing radial subaev
            angular_subaev : time spent on computing angular subaev
            total : total time for computing everything.
    """

    def __init__(self, benchmark=False, device=default_device, dtype=default_dtype, const_file=buildin_const_file):
        super(SortedAEV, self).__init__(benchmark, dtype, device, const_file)
        if benchmark:
            self.radial_subaev_terms = self._enable_benchmark(
                self.radial_subaev_terms, 'radial terms')
            self.angular_subaev_terms = self._enable_benchmark(
                self.angular_subaev_terms, 'angular terms')
            self.terms_and_indices = self._enable_benchmark(
                self.terms_and_indices, 'terms and indices')
            self.partition = self._enable_benchmark(
                self.partition, 'partition')
            self.assemble = self._enable_benchmark(self.assemble, 'assemble')
            self.forward = self._enable_benchmark(self.forward, 'total')

    def _d(self, x):
        # TODO: remove this when pytorch support `torch.unique` on GPU
        return x.to(self.device)

    def species_to_tensor(self, species):
        """Convert species list into a long tensor.

        Parameters
        ----------
        species : list
            List of string for the species of each atoms.

        Returns
        -------
        torch.Tensor
            Long tensor for the species, where a value k means the species is
            the same as self.species[k].
        """
        indices = {self.species[i]: i for i in range(len(self.species))}
        values = [indices[i] for i in species]
        return torch.tensor(values, dtype=torch.long, device=self.device)

    def radial_subaev_terms(self, distances):
        """Compute the radial subAEV terms of the center atom given neighbors

        The radial AEV is define in https://arxiv.org/pdf/1610.08935.pdf equation 3.
        The sum computed by this method is over all given neighbors, so the caller
        of this method need to select neighbors if the caller want a per species subAEV.

        Parameters
        ----------
        distances : torch.Tensor
            Pytorch tensor of shape (..., neighbors) storing the |Rij| length where i are the
            center atoms, and j are their neighbors.

        Returns
        -------
        torch.Tensor
            A tensor of shape (..., neighbors, `radial_sublength`) storing the subAEVs.
        """
        distances = distances.unsqueeze(-1).unsqueeze(-1)
        fc = _cutoff_cosine(distances, self.Rcr)
        # Note that in the equation in the paper there is no 0.25 coefficient, but in NeuroChem there is such a coefficient. We choose to be consistent with NeuroChem instead of the paper here.
        ret = 0.25 * torch.exp(-self.EtaR * (distances - self.ShfR)**2) * fc
        new_shape = list(ret.shape[:-2]) + [-1]
        return ret.view(*new_shape)

    def angular_subaev_terms(self, vectors1, vectors2):
        """Compute the angular subAEV terms of the center atom given neighbor pairs.

        The angular AEV is define in https://arxiv.org/pdf/1610.08935.pdf equation 4.
        The sum computed by this method is over all given neighbor pairs, so the caller
        of this method need to select neighbors if the caller want a per species subAEV.

        Parameters
        ----------
        vectors1, vectors2: torch.Tensor
            Tensor of shape (..., pairs, 3) storing the Rij vectors of pairs of neighbors.
            The vectors1(..., j, :) and vectors2(..., j, :) are the Rij vectors of the
            two atoms of pair j.

        Returns
        -------
        torch.Tensor
            Tensor of shape (..., pairs, `angular_sublength`) storing the subAEVs.
        """
        vectors1 = vectors1.unsqueeze(
            -1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        vectors2 = vectors2.unsqueeze(
            -1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        distances1 = vectors1.norm(2, dim=-5)
        distances2 = vectors2.norm(2, dim=-5)

        # 0.95 is multiplied to the cos values to prevent acos from returning NaN.
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
        # ret now have shape (..., pairs, ?, ?, ?, ?) where ? depend on constants
        # flat the last 4 dimensions to view the subAEV as one dimension vector
        new_shape = list(ret.shape[:-4]) + [-1]
        return ret.view(*new_shape)

    def terms_and_indices(self, coordinates):
        """Compute radial and angular subAEV terms, and original indices.

        Terms will be sorted according to their distances to central atoms, and only
        these within cutoff radius are valid. The returned indices contains what would
        their original indices be if they were unsorted.

        Parameters
        ----------
        coordinates : torch.Tensor
            The tensor that specifies the xyz coordinates of atoms in the molecule.
            The tensor must have shape (conformations, atoms, 3)

        Returns
        -------
        (radial_terms, angular_terms, indices_r, indices_a)
        radial_terms : torch.Tensor
            Tensor of shape (conformations, atoms, neighbors, `radial_sublength`) for
            the (unsummed) radial subAEV terms.
        angular_terms : torch.Tensor
            Tensor of shape (conformations, atoms, pairs, `angular_sublength`) for the
            (unsummed) angular subAEV terms.
        indices_r : torch.Tensor
            Tensor of shape (conformations, atoms, neighbors). Let l = indices_r(i,j,k),
            then this means that radial_terms(i,j,k,:) is in the subAEV term of conformation
            i between atom j and atom l. 
        indices_a : torch.Tensor
            Same as indices_r, except that the cutoff radius is Rca instead of Rcr.
        """

        vec = coordinates.unsqueeze(2) - coordinates.unsqueeze(1)
        """Shape (conformations, atoms, atoms, 3) storing Rij vectors"""

        distances = vec.norm(2, -1)
        """Shape (conformations, atoms, atoms) storing Rij distances"""

        distances, indices = distances.sort(-1)

        min_distances = distances.min(0)[0].min(0)[0]
        inRcr = (min_distances <= self.Rcr).nonzero().view(-1)[1:]
        inRca = (min_distances <= self.Rca).nonzero().view(-1)[1:]

        distances = distances.index_select(-1, inRcr)
        indices_r = indices.index_select(-1, inRcr)
        radial_terms = self.radial_subaev_terms(distances)

        indices_a = indices.index_select(-1, inRca)
        new_shape = list(indices_a.shape) + [3]
        _indices_a = indices_a.unsqueeze(-1).expand(*new_shape)
        vec = vec.gather(-2, _indices_a)
        vec = _utils.combinations(vec, -2)
        angular_terms = self.angular_subaev_terms(
            *vec) if vec is not None else None

        return radial_terms, angular_terms, indices_r, indices_a

    def partition(self, indices_r, indices_a, species):
        """Partition indices according to their species

        Parameters
        ----------
        indices_r : torch.Tensor
            Tensor of shape (conformations, atoms, neighbors). Let l = indices_r(i,j,k),
            then this means that radial_terms(i,j,k,:) is in the subAEV term of conformation
            i between atom j and atom l. 
        indices_a : torch.Tensor
            Same as indices_r, except that the cutoff radius is Rca instead of Rcr.
        species : torch.Tensor
            Long tensor for the species, where a value k means the species is
            the same as self.species[k].

        Returns
        -------
        (present_species, mask_r, mask_a1, mask_a2)
        present_species : torch.Tensor
            Long tensor for species of atoms present in the molecules.
        mask_r : torch.Tensor
            Tensor of shape (conformations, atoms, neighbors, present species) storing
            the mask for each species.
        mask_a : torch.Tensor
            Tensor of shape (conformations, atoms, pairs, present species, present species)
            storing the mask for each pair.
        """
        species_r = species[indices_r]
        species_a = species[indices_a]
        species_a = _utils.combinations(species_a, -1)
        if species_a is not None:
            # TODO: can we remove this if pytorch support 0 size tensors?
            species_a1, species_a2 = species_a

        present_species = self._d(_cpu(species).unique()).sort()[0]

        mask_r = (species_r.unsqueeze(-1) ==
                  torch.arange(len(self.species), device=self.device))

        if species_a is not None:
            mask_a1 = (species_a1.unsqueeze(-1) ==
                       present_species).unsqueeze(-1)
            mask_a2 = (species_a2.unsqueeze(-1).unsqueeze(-1)
                       == present_species)
            mask = mask_a1 * mask_a2
            mask_rev = mask.permute(0, 1, 2, 4, 3)
            mask_a = (mask + mask_rev) > 0
            return present_species, mask_r, mask_a
        else:
            return present_species, mask_r, None

    def assemble(self, radial_terms, angular_terms, present_species, mask_r, mask_a):
        """Assemble radial and angular AEV from computed terms according to the given partition information.

        Parameters
        ----------
        radial_terms : torch.Tensor
            Tensor of shape (conformations, atoms, neighbors, `radial_sublength`) for
            the (unsummed) radial subAEV terms.
        angular_terms : torch.Tensor
            Tensor of shape (conformations, atoms, pairs, `angular_sublength`) for the
            (unsummed) angular subAEV terms.
        present_species : torch.Tensor
            Long tensor for species of atoms present in the molecules.
        mask_r : torch.Tensor
            Tensor of shape (conformations, atoms, neighbors, present species) storing
            the mask for each species.
        mask_a : torch.Tensor
            Tensor of shape (conformations, atoms, pairs, present species, present species)
            storing the mask for each pair.

        Returns
        -------
        (torch.Tensor, torch.Tensor)
            Returns (radial AEV, angular AEV), both are pytorch tensor of `dtype`.
            The radial AEV must be of shape (conformations, atoms, radial_length)
            The angular AEV must be of shape (conformations, atoms, angular_length)
        """
        conformations = radial_terms.shape[0]
        atoms = radial_terms.shape[1]

        # assemble radial subaev
        present_radial_aevs = (radial_terms.unsqueeze(-2)
                               * mask_r.unsqueeze(-1).type(self.dtype)).sum(-3)
        """Tensor of shape (conformations, atoms, present species, radial_length)"""
        new_shape = list(radial_terms.shape[:2]) + [-1]
        radial_aevs = present_radial_aevs.view(*new_shape)

        # assemble angular subaev
        rev_indices = {present_species[i].item(): i
                       for i in range(len(present_species))}
        """Tensor of shape (conformations, atoms, present species, present species, angular_length)"""
        angular_aevs = []
        zero_angular_subaev = torch.zeros(
            conformations, atoms, self.angular_sublength, dtype=self.dtype, device=self.device)
        for s1, s2 in itertools.combinations_with_replacement(range(len(self.species)), 2):
            # TODO: can we remove this if pytorch support 0 size tensors?
            if s1 in rev_indices and s2 in rev_indices and mask_a is not None:
                i1 = rev_indices[s1]
                i2 = rev_indices[s2]
                mask = mask_a[..., i1, i2].unsqueeze(-1).type(self.dtype)
                subaev = (angular_terms * mask).sum(-2)
            else:
                subaev = zero_angular_subaev
            angular_aevs.append(subaev)

        return radial_aevs, torch.cat(angular_aevs, dim=2)

    def forward(self, coordinates, species):
        species = self.species_to_tensor(species)
        radial_terms, angular_terms, indices_r, indices_a = self.terms_and_indices(
            coordinates)
        present_species, mask_r, mask_a = self.partition(
            indices_r, indices_a, species)
        return self.assemble(radial_terms, angular_terms, present_species, mask_r, mask_a)

    def export_radial_subaev_onnx(self, filename):
        """Export the operation that compute radial subaev into onnx format

        Parameters
        ----------
        filename : string
            Name of the file to store exported networks.
        """
        class M(torch.nn.Module):
            def __init__(self, outerself):
                super(M, self).__init__()
                self.outerself = outerself

            def forward(self, center, neighbors):
                return self.outerself.radial_subaev(center, neighbors)
        dummy_center = torch.randn(1, 3)
        dummy_neighbors = torch.randn(1, 5, 3)
        torch.onnx.export(M(self), (dummy_center, dummy_neighbors), filename)
