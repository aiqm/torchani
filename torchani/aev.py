import torch
import itertools
from .aev_base import AEVComputer
from . import buildin_const_file, default_dtype, default_device
from . import _utils


class AEV(AEVComputer):
    """The AEV computer fully implemented using pytorch, making use of neighbor list

    Attributes
    ----------
    timers : dict
        Dictionary storing the the benchmark result. It has the following keys:
            neighborlist : time spent on computing neighborlist
            aev : time spent on computing AEV, after the nighborlist is given
            total : total time for computing everything, including neighborlist and AEV.
    """

    def __init__(self, benchmark=False, device=default_device, dtype=default_dtype, const_file=buildin_const_file):
        super(AEV, self).__init__(benchmark, dtype, device, const_file)
        if benchmark:
            self.compute_neighborlist = self._enable_benchmark(
                self.compute_neighborlist, 'neighborlist')
            self.compute_aev_using_neighborlist = self._enable_benchmark(
                self.compute_aev_using_neighborlist, 'aev')
            self.forward = self._enable_benchmark(self.forward, 'total')

    def radial_subaev(self, center, neighbors):
        """Compute the radial subAEV of the center atom given neighbors

        The radial AEV is define in https://arxiv.org/pdf/1610.08935.pdf equation 3.
        The sum computed by this method is over all given neighbors, so the caller
        of this method need to select neighbors if the caller want a per species subAEV.

        Parameters
        ----------
        center : pytorch tensor of `dtype`
            A tensor of shape (conformations, 3) that stores the xyz coordinate of the
            center atoms.
        neighbors : pytorch tensor of `dtype`
            A tensor of shape (conformations, N, 3) where N is the number of neighbors.
            The tensor stores the xyz coordinate of the neighbor atoms. Note that different
            conformations might have different neighbor atoms within the cutoff radius, if
            this is the case, the union of neighbors of all conformations should be given for
            this parameter.

        Returns
        -------
        pytorch tensor of `dtype`
            A tensor of shape (conformations, `per_species_radial_length()`) storing the subAEVs.
        """
        atoms = neighbors.shape[1]
        Rij_vec = neighbors - center.view(-1, 1, 3)
        """pytorch tensor of shape (conformations, N, 3) storing the Rij vectors where i is the
        center atom, and j is a neighbor. The Rij of conformation n is stored as (n,j,:)"""
        distances = torch.sqrt(torch.sum(Rij_vec ** 2, dim=-1))
        """pytorch tensor of shape (conformations, N) storing the |Rij| length where i is the
        center atom, and j is a neighbor. The |Rij| of conformation n is stored as (n,j)"""

        # use broadcasting semantics to do Cartesian product on constants
        # shape convension (conformations, atoms, EtaR, ShfR)
        distances = distances.view(-1, atoms, 1, 1)
        fc = AEVComputer._cutoff_cosine(distances, self.constants['Rcr'])
        eta = self.constants['EtaR'].view(1, 1, -1, 1)
        radius_shift = self.constants['ShfR'].view(1, 1, 1, -1)
        # Note that in the equation in the paper there is no 0.25 coefficient, but in NeuroChem there is such a coefficient. We choose to be consistent with NeuroChem instead of the paper here.
        ret = 0.25 * torch.exp(-eta * (distances - radius_shift)**2) * fc
        # end of shape convension
        ret = torch.sum(ret, dim=1)
        # flat the last two dimensions to view the subAEV as one dimensional vector
        return ret.view(-1, self.per_species_radial_length())

    def angular_subaev(self, center, neighbors):
        """Compute the angular subAEV of the center atom given neighbor pairs.

        The angular AEV is define in https://arxiv.org/pdf/1610.08935.pdf equation 4.
        The sum computed by this method is over all given neighbor pairs, so the caller
        of this method need to select neighbors if the caller want a per species subAEV.

        Parameters
        ----------
        center : pytorch tensor of `dtype`
            A tensor of shape (conformations, 3) that stores the xyz coordinate of the
            center atoms.
        neighbors : pytorch tensor of `dtype`
            A tensor of shape (conformations, N, 2, 3) where N is the number of neighbor pairs.
            The tensor stores the xyz coordinate of the 2 atoms in neighbor pairs. Note that
            different conformations might have different neighbor pairs within the cutoff radius,
            if this is the case, the union of neighbors of all conformations should be given for
            this parameter.

        Returns
        -------
        pytorch tensor of `dtype`
            A tensor of shape (conformations, `per_species_angular_length()`) storing the subAEVs.
        """
        pairs = neighbors.shape[1]
        Rij_vec = neighbors - center.view(-1, 1, 1, 3)
        """pytorch tensor of shape (conformations, N, 2, 3) storing the Rij vectors where i is the
        center atom, and j is a neighbor. The vector (n,k,l,:) is the Rij where j refer to the l-th
        atom of the k-th pair."""
        R_distances = torch.sqrt(torch.sum(Rij_vec ** 2, dim=-1))
        """pytorch tensor of shape (conformations, N, 2) storing the |Rij| length where i is the
        center atom, and j is a neighbor. The value at (n,k,l) is the |Rij| where j refer to the
        l-th atom of the k-th pair."""

        # Compute the product of two distances |Rij| * |Rik| where j and k are the two atoms in
        # a pair. The result tensor would have shape (conformations, pairs)
        Rijk_distance_prods = R_distances[:, :, 0] * R_distances[:, :, 1]

        # Compute the inner product Rij (dot) Rik where j and k are the two atoms in a pair.
        # The result tensor would have shape (conformations, pairs)
        Rijk_inner_prods = torch.sum(
            Rij_vec[:, :, 0, :] * Rij_vec[:, :, 1, :], dim=-1)

        # Compute the angles jik with i in the center and j and k are the two atoms in a pair.
        # The result tensor would have shape (conformations, pairs)
        # 0.95 is multiplied to the cos values to prevent acos from returning NaN.
        cos_angles = 0.95 * Rijk_inner_prods / Rijk_distance_prods
        angles = torch.acos(cos_angles)

        # use broadcasting semantics to combine constants
        # shape convension (conformations, pairs, EtaA, Zeta, ShfA, ShfZ)
        angles = angles.view(-1, pairs, 1, 1, 1, 1)
        Rij = R_distances.view(-1, pairs, 2, 1, 1, 1, 1)
        fcj = AEVComputer._cutoff_cosine(Rij, self.constants['Rca'])
        eta = self.constants['EtaA'].view(1, 1, -1, 1, 1, 1)
        zeta = self.constants['Zeta'].view(1, 1, 1, -1, 1, 1)
        radius_shifts = self.constants['ShfA'].view(1, 1, 1, 1, -1, 1)
        angle_shifts = self.constants['ShfZ'].view(1, 1, 1, 1, 1, -1)
        ret = 2 * ((1 + torch.cos(angles - angle_shifts)) / 2) ** zeta * \
            torch.exp(-eta * ((Rij[:, :, 0, :, :, :, :] + Rij[:, :, 1, :, :, :, :]) / 2 - radius_shifts)
                      ** 2) * fcj[:, :, 0, :, :, :, :] * fcj[:, :, 1, :, :, :, :]
        # end of shape convension
        ret = torch.sum(ret, dim=1)
        # flat the last 4 dimensions to view the subAEV as one dimension vector
        return ret.view(-1, self.per_species_angular_length())

    def compute_neighborlist(self, coordinates, species):
        """Compute neighbor list of each atom, and group neighbors by species"""

        coordinates = coordinates.detach()
        atoms = coordinates.shape[1]

        R_vecs = coordinates.unsqueeze(1) - coordinates.unsqueeze(2)
        """pytorch tensor of `dtype`: A tensor of shape (conformations, atoms, atoms, 3)
        that stores Rij vectors. The 3 dimensional vector at (N, i, j, :) is the Rij vector
        of conformation N.
        """

        R_distances = torch.sqrt(torch.sum(R_vecs ** 2, dim=-1))
        """pytorch tensor of `dtype`: A tensor of shape (conformations, atoms, atoms)
        that stores |Rij|, i.e. the length of Rij vectors. The value at (N, i, j) is
        the |Rij| of conformation N.
        """

        # Compute the list of atoms inside cutoff radius
        # The list is stored as a tensor of shape (atoms, atoms)
        # where the value at (i,j) == 1 means j is a neighbor of i, otherwise
        # the value should be 0
        in_Rcr = R_distances <= self.constants['Rcr']
        in_Rcr = torch.sum(in_Rcr, dim=0) > 0
        # set diagnoal elements to 0
        in_Rcr = in_Rcr * \
            (1 - torch.eye(atoms, dtype=in_Rcr.dtype, device=self.device))
        in_Rca = R_distances <= self.constants['Rca']
        in_Rca = torch.sum(in_Rca, dim=0) > 0
        # set diagnoal elements to 0
        in_Rca = in_Rca * \
            (1 - torch.eye(atoms, dtype=in_Rca.dtype, device=self.device))

        # Compute species selectors for all supported species.
        # A species selector is a tensor of shape (atoms,)
        # where the value == 1 means the atom with that index is the species
        # of the selector otherwise value should be 0.
        species_selectors = {}
        for i in range(atoms):
            s = species[i]
            if s not in species_selectors:
                species_selectors[s] = torch.zeros(
                    atoms, dtype=in_Rcr.dtype, device=self.device)
            species_selectors[s][i] = 1

        # Compute the list of neighbors of various species
        # The list is stored as a tensor of shape (atoms, atoms)
        # where the value at (i,j) == 1 means j is a neighbor of i and j has the
        # specified species, otherwise the value should be 0
        species_neighbors = {}
        for s in species_selectors:
            selector = species_selectors[s].unsqueeze(0)
            species_neighbors[s] = (in_Rcr * selector, in_Rca * selector)

        return species_neighbors

    def compute_aev_using_neighborlist(self, coordinates, species, species_neighbors):
        conformations = coordinates.shape[0]
        atoms = coordinates.shape[1]
        # helper exception for control flow

        class AEVIsZero(Exception):
            pass

        # compute radial AEV
        radial_aevs = []
        """The list whose elements are full radial AEV of each atom"""
        for i in range(atoms):
            radial_aev = []
            """The list whose elements are atom i's per species subAEV of each species"""
            for s in self.species:
                try:
                    if s not in species_neighbors:
                        raise AEVIsZero()
                    indices = species_neighbors[s][0][i, :].nonzero().view(-1)
                    """Indices of atoms that have species s and position inside cutoff radius"""
                    if indices.shape[0] <= 0:
                        raise AEVIsZero()
                    neighbors = coordinates.index_select(1, indices)
                    """pytroch tensor of shape (conformations, N, 3) storing coordinates of
                    neighbor atoms that have desired species, where N is the number of neighbors.
                    """
                    radial_aev.append(self.radial_subaev(
                        coordinates[:, i, :], neighbors))
                except AEVIsZero:
                    # If no neighbor atoms have desired species, fill the subAEV with zeros
                    radial_aev.append(torch.zeros(
                        conformations, self.per_species_radial_length(), dtype=self.dtype, device=self.device))
            radial_aev = torch.cat(radial_aev, dim=1)
            radial_aevs.append(radial_aev)
        radial_aevs = torch.stack(radial_aevs, dim=1)

        # compute angular AEV
        angular_aevs = []
        """The list whose elements are full angular AEV of each atom"""
        for i in range(atoms):
            angular_aev = []
            """The list whose elements are atom i's per species subAEV of each species"""
            for j, k in itertools.combinations_with_replacement(self.species, 2):
                try:
                    if j not in species_neighbors or k not in species_neighbors:
                        raise AEVIsZero()
                    indices_j = species_neighbors[j][1][i, :].nonzero(
                    ).view(-1)
                    if indices_j.shape[0] < 1:
                        raise AEVIsZero()
                    """Indices of atoms that have species j and position inside cutoff radius"""
                    neighbors_j = coordinates.index_select(1, indices_j)
                    """pytroch tensor of shape (conformations, N, 3) storing coordinates of
                    neighbors atoms that have desired species j, where N is the number of neighbors.
                    """
                    if j != k:
                        # the two atoms in the pair have different species
                        indices_k = species_neighbors[k][1][i, :].nonzero(
                        ).view(-1)
                        """Indices of atoms that have species k and position inside cutoff radius"""
                        if indices_k.shape[0] < 1:
                            raise AEVIsZero()
                        neighbors_k = coordinates.index_select(1, indices_k)
                        """pytroch tensor of shape (conformations, N, 3) storing coordinates of
                        neighbors atoms that have desired species k, where N is the number of neighbors.
                        """
                        neighbors = _utils.cartesian_prod(
                            neighbors_j, neighbors_k, dim=1, newdim=2)
                    else:
                        # the two atoms in the pair have the same species j
                        if indices_j.shape[0] < 2:
                            raise AEVIsZero()
                        neighbors = _utils.combinations(
                            neighbors_j, 2, dim=1, newdim=2)
                    angular_aev.append(self.angular_subaev(
                        coordinates[:, i, :], neighbors))
                except AEVIsZero:
                    # If unable to find pair of neighbor atoms with desired species, fill the subAEV with zeros
                    angular_aev.append(torch.zeros(
                        conformations, self.per_species_angular_length(), dtype=self.dtype, device=self.device))
            angular_aev = torch.cat(angular_aev, dim=1)
            angular_aevs.append(angular_aev)
        angular_aevs = torch.stack(angular_aevs, dim=1)

        return radial_aevs, angular_aevs

    def forward(self, coordinates, species):
        # For the docstring of this method, refer to the base class
        species_neighbors = self.compute_neighborlist(coordinates, species)
        return self.compute_aev_using_neighborlist(coordinates, species, species_neighbors)

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
