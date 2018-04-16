import torch
import numpy
import itertools
from .aev_base import AEVComputer
from . import buildin_const_file


class AEV(AEVComputer):
    """The AEV computer fully implemented using pytorch"""

    def __init__(self, dtype=torch.cuda.float32, const_file=buildin_const_file):
        super(AEV, self).__init__(dtype, const_file)

        # assign supported species indices
        self._species_indices = {}
        index = 0
        for i in self.species:
            self._species_indices[i] = index
            index += 1

    @staticmethod
    def _cutoff_cosine(distances, cutoff):
        """Compute the elementwise cutoff cosine function

        The cutoff cosine function is define in https://arxiv.org/pdf/1610.08935.pdf equation 2

        Parameters
        ----------
        distances : pytorch tensor of `dtype`
            The pytorch tensor that stores Rij values. This tensor can have any shape since the cutoff
            cosine function is computed elementwise.
        cutoff : float
            The cutoff radius, i.e. the Rc in the equation. For any Rij > Rc, the function value is defined to be zero.

        Returns
        -------
        pytorch tensor of `dtype`
            The tensor of the same shape as `distances` that stores the computed function values.
        """
        return torch.where(distances <= cutoff, 0.5 * torch.cos(numpy.pi * distances / cutoff) + 0.5, torch.zeros_like(distances))

    def _compute_radial_terms(self, distances):
        """Compute the single terms of per atom radial AEV

        The radial AEV is define in https://arxiv.org/pdf/1610.08935.pdf equation 3. This function does
        not compute the sum, but compute each individual term in the sum instead.

        Parameters
        ----------
        distances : pytorch tensor of `dtype`
            The pytorch tensor that stores |Rij| values. This tensor must have shape (conformations, atoms, atoms),
            where (N,i,j) stores the of |Rij| of conformation N.

        Returns
        -------
        pytorch tensor of `dtype`
            The returned tensor have a shape (conformations, atoms, atoms, `per_species_radial_length()`).
            The vector at (N,i,j,:) storing the per atom radial subAEV of Rij of conformation N.
        """
        # use broadcasting semantics to do Cartesian product on constants
        # shape convension (conformations, atoms, atoms, EtaR, ShfR)
        atoms = distances.shape[1]
        distances = distances.view(-1, atoms, atoms, 1, 1)
        fc = AEV._cutoff_cosine(distances, self.constants['Rcr'])
        eta = torch.Tensor(self.constants['EtaR']).type(
            self.dtype).view(1, 1, 1, -1, 1)
        radius_shift = torch.Tensor(self.constants['ShfR']).type(
            self.dtype).view(1, 1, 1, 1, -1)
        # Note that in the equation in the paper there is no 0.25 coefficient, but in NeuroChem there is such a coefficient. We choose to be consistent with NeuroChem instead of the paper here.
        ret = 0.25 * torch.exp(-eta * (distances - radius_shift)**2) * fc
        # end of shape convension
        # flat the last two dimensions to view the subAEV as one dimension vector
        return ret.view(-1, atoms, atoms, self.per_species_radial_length())

    def _compute_angular_terms(self, R_vecs, R_distances):
        """Compute the single terms of per atom pair angular AEV

        The angular AEV is define in https://arxiv.org/pdf/1610.08935.pdf equation 4. This function does
        not compute the sum, but compute each individual term in the sum instead.

        Parameters
        ----------
        R_vecs : pytorch tensor of `dtype`
            The pytorch tensor that stores Rij vectors. This tensor must have shape (conformations, atoms, atoms, 3),
            where (N,i,j,:) stores the vector Rij of conformation N.
        R_distances : pytorch tensor of `dtype`
            The pytorch tensor that stores |Rij| values. This tensor must have shape (conformations, atoms, atoms),
            where (N,i,j) stores the |Rij| value of conformation N.

        Returns
        -------
        pytorch tensor of `dtype`
            The returned tensor have a shape (conformations, atoms, atoms, atoms, `per_species_angular_length()`).
            The vector at (N,i,j,k,:) storing the per atom pair angular subAEV of Rij and Rik of conformation N.
        """
        atoms = R_distances.shape[1]

        # Compute the product of two distances |Rij| * |Rik|, the result tensor would have
        # shape (conformations, atoms, atoms, atoms)
        Rijk_distance_prods = R_distances.unsqueeze(
            2) * R_distances.unsqueeze(3)

        # Compute the inner product Rij (dot) Rik, the result tensor would have
        # shape (conformations, atoms, atoms, atoms)
        Rijk_inner_prods = torch.sum(
            R_vecs.unsqueeze(2) * R_vecs.unsqueeze(3), dim=-1)

        # Compute the angles jik with i in the center, the result tensor would have
        # shape (conformations, atoms, atoms, atoms)
        # 0.95 is multiplied to the cos values to prevent acos from returning NaN.
        cos_angles = 0.95 * Rijk_inner_prods / Rijk_distance_prods
        angles = torch.acos(cos_angles)

        # use broadcasting semantics to combine constants
        # shape convension (conformations, atoms, atoms, atoms, EtaA, Zeta, ShfA, ShfZ)
        angles = angles.view(-1, atoms, atoms, atoms, 1, 1, 1, 1)
        Rij = R_distances.view(-1, atoms, atoms, 1, 1, 1, 1, 1)
        Rik = R_distances.view(-1, atoms, 1, atoms, 1, 1, 1, 1)
        fcj = AEV._cutoff_cosine(Rij, self.constants['Rca'])
        fck = AEV._cutoff_cosine(Rik, self.constants['Rca'])
        eta = torch.Tensor(self.constants['EtaA']).type(
            self.dtype).view(1, 1, 1, 1, -1, 1, 1, 1)
        zeta = torch.Tensor(self.constants['Zeta']).type(
            self.dtype).view(1, 1, 1, 1, 1, -1, 1, 1)
        radius_shifts = torch.Tensor(self.constants['ShfA']).type(
            self.dtype).view(1, 1, 1, 1, 1, 1, -1, 1)
        angle_shifts = torch.Tensor(self.constants['ShfZ']).type(
            self.dtype).view(1, 1, 1, 1, 1, 1, 1, -1)
        ret = 2 * ((1 + torch.cos(angles - angle_shifts)) / 2) ** zeta * \
            torch.exp(-eta * ((Rij + Rik) / 2 - radius_shifts)
                      ** 2) * fcj * fck
        # end of shape convension
        # flat the last 4 dimensions to view the subAEV as one dimension vector
        return ret.view(-1, atoms, atoms, atoms, self.per_species_angular_length())

    def _sum_radial_terms(self, radial_terms, species):
        """Sum up the computed radial subAEV terms"""
        conformations = radial_terms.shape[0]
        atoms = len(species)

        # Here we need to group radial subAEV terms by their species, and sum up each group.
        # Instead of using a for loop to do the sum, we first unsqueeze `radial_terms` from
        # shape (conformations, atoms, atoms, `per_species_radial_length()`) into
        # shape (conformations, atoms, 1, atoms, `per_species_radial_length()`), where
        # the extra dimension will becomes the group index. The unsqueeze `radial_terms` will
        # then be multilied to the tensor `radial_sum_indices` which specifies which term goes
        # which group of which atom. Then follows a sum operation on the specified axes to compute
        # the sum.

        radial_sum_indices = torch.zeros(1, atoms, len(self.species),
                                         atoms, 1, dtype=self.dtype)
        """pytorch tensor of `dtype`: The tensor that specifies which atom goes which group.
        This tensor has shape (1, atoms, `len(self.species)`, atoms, 1), where the value at
        index (0, i, j, k, 0) == 1 means in the sum of atom i's radial subAEV of species j
        the term k is included . Otherwise the value should be 0.
        """
        # compute `radial_sum_indices`` below
        for i in range(len(species)):
            for j in range(len(species)):
                if j == i:
                    continue
                key = species[j]
                radial_sum_indices[0, i, self._species_indices[key], j, 0] = 1

        radial_aevs = radial_terms.unsqueeze(2) * radial_sum_indices
        return torch.sum(radial_aevs, dim=3).view(conformations, atoms, -1)

    def _sum_angular_terms(self, angular_terms, species):
        """Sum up the computed angular subAEV terms"""

        # Since broadcasting semantics based approaches like `_sum_radial_terms` takes
        # too much memory for the angular case, we use python `for` loops to sum up the
        # radial terms.

        conformations = angular_terms.shape[0]
        atoms = angular_terms.shape[1]

        angular_aevs = []
        """list: The list whose element is the angular AEV of each atom"""
        for i in range(atoms):
            angular_sum_by_species = {}
            """dict: The dictionary that stores the angular sum of this atom of each species"""

            # compute the sums
            for j in range(atoms):
                if j == i:
                    continue
                for k in range(j+1, atoms):
                    if k == i:
                        continue
                    angular_term = angular_terms[:, i, j, k, :]
                    key = frozenset(species[j]+species[k])
                    if key in angular_sum_by_species:
                        angular_sum_by_species[key] += angular_term
                    else:
                        angular_sum_by_species[key] = angular_term

            # convert `angular_sum_by_species` into a list with fixed species order
            angular_aev = []
            per_species_angular_length = self.per_species_angular_length()
            for k, l in itertools.combinations_with_replacement(self.species, 2):
                key = frozenset([k, l])
                if key in angular_sum_by_species:
                    angular_aev.append(angular_sum_by_species[key])
                else:
                    angular_aev.append(torch.zeros(
                        conformations, per_species_angular_length, dtype=self.dtype))

            # append angular_aev of this atom to `angular_aevs`
            angular_aevs.append(torch.cat(angular_aev, dim=1))
        return torch.stack(angular_aevs, dim=1)

    def __call__(self, coordinates, species):
        # For the docstring of this method, refer to the base class

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

        radial_terms = self._compute_radial_terms(R_distances)
        radial_aevs = self._sum_radial_terms(radial_terms, species)

        angular_terms = self._compute_angular_terms(R_vecs, R_distances)
        angular_aevs = self._sum_angular_terms(angular_terms, species)

        return radial_aevs, angular_aevs
