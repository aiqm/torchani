import torch
import itertools
from .aev_base import AEVComputer
from . import buildin_const_file, default_dtype, default_device
from . import _utils


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
            self.terms_and_indices = self._enable_benchmark(self.terms_and_indices, 'terms and indices')
            self.partition = self._enable_benchmark(self.partition, 'partition')
            self.assemble = self._enable_benchmark(self.assemble, 'assemble')
            self.forward = self._enable_benchmark(self.forward, 'total')

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
        fc = AEVComputer._cutoff_cosine(distances, self.Rcr)
        # Note that in the equation in the paper there is no 0.25 coefficient, but in NeuroChem there is such a coefficient. We choose to be consistent with NeuroChem instead of the paper here.
        ret = 0.25 * torch.exp(-self.EtaR * (distances - self.ShfR)**2) * fc
        return ret.view(*ret.shape[:-2], -1)

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
        vectors1 = vectors1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        vectors2 = vectors1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        distances1 = vectors1.norm(2, dim=-5)
        distances2 = vectors2.norm(2, dim=-5)

        # 0.95 is multiplied to the cos values to prevent acos from returning NaN.
        cos_angles = 0.95 * \
            torch.nn.functional.cosine_similarity(
                vectors1, vectors2, dim=-5)
        angles = torch.acos(cos_angles)

        fcj1 = AEVComputer._cutoff_cosine(distances1, self.Rca)
        fcj2 = AEVComputer._cutoff_cosine(distances2, self.Rca)
        factor1 = ((1 + torch.cos(angles - self.ShfZ)) / 2) ** self.Zeta
        factor2 = torch.exp(-self.EtaA *
                            ((distances1 + distances2) / 2 - self.ShfA) ** 2)
        ret = 2 * factor1 * factor2 * fcj1 * fcj2
        # ret now have shape (..., pairs, ?, ?, ?, ?) where ? depend on constants
        # flat the last 4 dimensions to view the subAEV as one dimension vector
        return ret.view(*ret.shape[:-4], -1)

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
            Tensor of shape (conformations, atoms, pairs, `radial_sublength`) for the
            (unsummed) angular subAEV terms.
        indices_r : torch.Tensor
            Tensor of shape (conformations, atoms, neighbors). Let l = indices_r(i,j,k),
            then this means that radial_terms(i,j,k,:) is in the subAEV term of conformation
            i between atom j and atom l. 
        indices_a : torch.Tensor
            Same as indices_r, except that the cutoff radius is Rca rather than Rcr.
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
        _indices_a = indices_a.unsqueeze(-1).expand(*indices_a.shape, 3)
        vec = vec.gather(-2, _indices_a)
        vec = _utils.combinations(vec, -2)
        angular_terms = self.angular_subaev_terms(*vec)

        return radial_terms, angular_terms, indices_r, indices_a

    def partition(self, indices_r, indices_a, species):
        """Partition indices according to their species
        
        Parameters
        ----------
        indices_r : torch.Tensor
            See the return value of `self.terms_and_indices`
        indices_a : torch.Tensor
            See the return value of `self.terms_and_indices`
        species : list of string
            The list that specifies the species of each atom. The length of the list
            must be the number of atoms.

        Returns
        -------
        dict
            The keys of the dict are species appears in the molecules. The values of the
            dict are a torch.Tensor triple of (_indices_r, _indices_a1, _indices_a2), where
            _indices_r is the indices of `indices_r` whose elements has the specified species,
            and _indices_a1 (_indices_a2) are the indices of combinations of `indices_a` where
            the first element of the pair has the specified species.
        """
        indices_a1, indices_a2 = _utils.combinations(indices_a, -1)
        atoms = len(species)
        partition = {}
        rev_species = species[::-1]
        for s in set(species):
            begin = species.index(s)
            end = atoms - rev_species.index(s)
            _indices_r = (indices_r >= begin) * (indices_r < end)
            _indices_r = _indices_r.nonzero().view(-1)
            _indices_a1 = (indices_a1 >= begin) * (indices_a1 < end)
            _indices_a1 = _indices_a1.nonzero().view(-1)
            _indices_a2 = (indices_a2 >= begin) * (indices_a2 < end)
            _indices_a2 = _indices_a2.nonzero().view(-1)
            partition[s] = (_indices_r, _indices_a1, _indices_a2)
        return partition

    def assemble(self, radial_terms, angular_terms, partition):
        """Assemble radial and angular AEV from computed terms according to the given partition information.

        Parameters
        ----------
        radial_terms : torch.Tensor
            See the return value of `self.terms_and_indices`
        angular_terms : torch.Tensor
            See the return value of `self.terms_and_indices`
        partition : dict
            See the return value of `self.partition`

        Returns
        -------
        (torch.Tensor, torch.Tensor)
            Returns (radial AEV, angular AEV), both are pytorch tensor of `dtype`.
            The radial AEV must be of shape (conformations, atoms, radial_length)
            The angular AEV must be of shape (conformations, atoms, angular_length)
        """
        conformations = radial_terms.shape[0]
        atoms = radial_terms.shape[1]
        zero_radial_subaev = torch.zeros(conformations, atoms, self.radial_sublength, dtype=self.dtype, device=self.device)
        zero_angular_subaev = torch.zeros(conformations, atoms, self.angular_sublength, dtype=self.dtype, device=self.device)

        radial_aevs = []
        angular_aevs = []
        for s in self.species:
            # assemble radial subaev
            if s in partition:
                _indices = partition[s][0]
                subaev = radial_terms.index_select(-2, _indices).sum(-2)
            else:
                subaev = zero_radial_subaev
            radial_aevs.append(subaev)
            # assemble angular subaev
            for s2 in self.species:
                if s in partition and s2 in partition:
                    _indices1 = partition[s][1]
                    _indices2 = partition[s2][1]
                    _indices = _indices1 * _indices2
                    print(_indices)
                    subaev = angular_terms.index_select(-2, _indices).sum(-2)
                else:
                    subaev = zero_angular_subaev
            angular_aevs.append(subaev)

        return torch.cat(radial_aevs, dim=1), torch.cat(angular_aevs, dim=1)

    def forward(self, coordinates, species):
        radial_terms, angular_terms, indices_r, indices_a = self.terms_and_indices(coordinates)
        partition = self.partition(indices_r, indices_a, species)
        return self.assemble(radial_terms, angular_terms, partition)

    def forward2(self, coordinates, species):
        conformations = coordinates.shape[0]
        atoms = coordinates.shape[1]

        # partition coordinates by species
        coordinates_by_species = {}
        atoms_by_species = {}
        rev_species = species[::-1]
        for s in set(species):
            begin = species.index(s)
            end = atoms - rev_species.index(s)
            coordinates_by_species[s] = coordinates[:, begin:end, :]
            atoms_by_species[s] = end - begin
        del coordinates

        vectors = {}
        """Dictionary storing Rij vectors of different species
        Key: (s0, s1), where s0 is species of center atom, and s1 is the species of neighbor
        Value: tensor of shape (conformations, atoms of s0, atoms of s1, 3)
        """

        radial_subaevs = {}
        """Dictionary storing radial subAEVs of different species
        Key: (s0, s1), where s0 is species of center atom, and s1 is the species of neighbor
        Value: tensor of shape (conformations, atoms of s0, self.radial_sublength)
        """

        angular_subaevs = {}
        """Dictionary storing radial subAEVs of different species
        Key: (s0, frozenset([s1,s2])), where s0 is species of center atom, and s1 and s2 are the species of neighbor
        Value: tensor of shape (conformations, atoms of s0, self.angular_sublength)
        """

        species_dedup = set(species)
        # compute radial AEV and prepare vectors for angular AEV computation
        for s1, s2 in itertools.product(species_dedup, species_dedup):
            coordinate1 = coordinates_by_species[s1]
            coordinate2 = coordinates_by_species[s2]

            if s1 == s2 and coordinate1.shape[1] == 1:
                continue

            vec = coordinate1.unsqueeze(2) - coordinate2.unsqueeze(1)
            """Shape (conformations, atoms of s1, atoms of s2, 3) storing Rij vectors"""

            distances = vec.norm(2, -1)
            """Shape (conformations, atoms of s1, atoms of s2) storing Rij distances"""

            # sort vec and distances according to distances to the center
            distances, indices = distances.sort(-1)
            min_distances = distances.min(0)[0].min(0)[0]
            inRcr = (min_distances <= self.Rcr).nonzero().view(-1)
            inRca = (min_distances <= self.Rca).nonzero().view(-1)
            if s1 == s2:
                if torch.numel(inRcr) > 0:
                    inRcr = inRcr[1:]
                if torch.numel(inRca) > 0:
                    inRca = inRca[1:]

            if torch.numel(inRcr) > 0:
                radial_subaevs[(s1, s2)] = self.radial_subaev(
                    distances.index_select(-1, inRcr))

            if torch.numel(inRca) > 0:
                indices = indices.index_select(-1, inRca)
                indices = indices.unsqueeze(-1).expand(*indices.shape, 3)
                vectors[(s1, s2)] = vec.gather(-2, indices)

        del coordinate1, coordinate2, vec, distances, indices, min_distances, inRcr, inRca

        # compute angular AEV
        for s0 in species_dedup:
            # case where the the two neighbor atoms are of the same species
            for s1 in species_dedup:
                if (s0, s1) in vectors:
                    vec = vectors[(s0, s1)]
                    if vec.shape[2] >= 2:
                        vec_pairs = _utils.combinations(
                            vec, 2, dim=2, newdim=3)
                        angular_subaevs[(s0, frozenset([s1]))
                                        ] = self.angular_subaev(vec_pairs)
            # case where the two eighbor atoms are of different species
            for s1, s2 in itertools.combinations(species_dedup, 2):
                if (s0, s1) in vectors and (s0, s2) in vectors:
                    vec1 = vectors[(s0, s1)]
                    vec2 = vectors[(s0, s2)]
                    vec_pairs = _utils.cartesian_prod(
                        vec1, vec2, dim=2, newdim=3)
                    angular_subaevs[(s0, frozenset([s1, s2]))
                                    ] = self.angular_subaev(vec_pairs)

        del vectors

        # assemble subAEVs to construct full radial and angular AEV
        species_dedup = sorted(species_dedup, key=self.species.index)
        radial_aevs = []
        angular_aevs = []
        for s0 in species_dedup:
            atoms = atoms_by_species[s0]
            zero_radial_subaev = torch.zeros(
                conformations, atoms, self.radial_sublength, dtype=self.dtype, device=self.device)
            zero_angular_subaev = torch.zeros(
                conformations, atoms, self.angular_sublength, dtype=self.dtype, device=self.device)

            radial_aev = []
            for s1 in self.species:
                if (s0, s1) in radial_subaevs:
                    radial_aev.append(radial_subaevs[(s0, s1)])
                else:
                    radial_aev.append(zero_radial_subaev)
            radial_aev = torch.cat(radial_aev, dim=-1)
            radial_aevs.append(radial_aev)

            angular_aev = []
            for s1, s2 in itertools.combinations_with_replacement(self.species, 2):
                key = (s0, frozenset([s1, s2]))
                if key in angular_subaevs:
                    angular_aev.append(angular_subaevs[key])
                else:
                    angular_aev.append(zero_angular_subaev)
            angular_aev = torch.cat(angular_aev, dim=-1)
            angular_aevs.append(angular_aev)

        return torch.cat(radial_aevs, dim=1), torch.cat(angular_aevs, dim=1)


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
