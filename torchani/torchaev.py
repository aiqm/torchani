import torch
import numpy
import itertools
import pkg_resources
from .aev_base import AEVComputer

default_const_file = pkg_resources.resource_filename(__name__, 'data/rHCNO-4.6R_16-3.1A_a4-8_3.params')

class AEV(AEVComputer):

    def __init__(self, dtype=torch.cuda.float32, const_file=default_const_file):
        super(AEV, self).__init__(dtype, const_file)
        
        # assign supported species indices
        self.species_indices = {}
        index = 0
        for i in self.species:
            self.species_indices[i] = index
            index += 1

    @staticmethod
    def _cutoff_cosine(distances, cutoff):
        return torch.where(distances <= cutoff, 0.5 * torch.cos(numpy.pi * distances / cutoff) + 0.5, torch.zeros_like(distances))

    def _compute_radial_terms(self, distances):
        # use broadcasting semantics to combine constants
        # shape convension (conformations, atoms, atoms, eta, radius_shift)
        atoms = distances.shape[1]
        distances = distances.view(-1, atoms, atoms, 1, 1)
        fc = AEV._cutoff_cosine(distances, self.constants['Rcr'])
        eta = torch.Tensor(self.constants['EtaR']).type(self.dtype).view(1,1,1,-1,1)
        radius_shift = torch.Tensor(self.constants['ShfR']).type(self.dtype).view(1,1,1,1,-1)
        ret = 0.25 * torch.exp(-eta * (distances - radius_shift)**2) * fc
        # end of shape convension
        # reshape to (conformations, atoms, atoms, ?)
        return ret.view(-1, atoms, atoms, self.per_species_radial_length())

    def _compute_angular_terms(self, angles, R_distances):
        # use broadcasting semantics to combine constants
        # shape convension (conformations, atoms, atoms, atoms, eta, zeta, radius_shift, angle_shift)
        atoms = angles.shape[1]
        angles = angles.view(-1, atoms, atoms, atoms, 1, 1, 1, 1)
        Rij = R_distances.view(-1, atoms, atoms, 1, 1, 1, 1, 1)
        Rik = R_distances.view(-1, atoms, 1, atoms, 1, 1, 1, 1)
        fcj = AEV._cutoff_cosine(Rij, self.constants['Rca'])
        fck = AEV._cutoff_cosine(Rik, self.constants['Rca'])
        eta = torch.Tensor(self.constants['EtaA']).type(self.dtype).view(1, 1, 1, 1, -1, 1, 1, 1)
        zeta = torch.Tensor(self.constants['Zeta']).type(self.dtype).view(1, 1, 1, 1, 1, -1, 1, 1)
        radius_shifts = torch.Tensor(self.constants['ShfA']).type(self.dtype).view(1, 1, 1, 1, 1, 1, -1, 1)
        angle_shifts = torch.Tensor(self.constants['ShfZ']).type(self.dtype).view(1, 1, 1, 1, 1, 1, 1, -1)
        ret = 2 * ((1 + torch.cos(angles - angle_shifts)) / 2) ** zeta * torch.exp(-eta * ((Rij + Rik) / 2 - radius_shifts) ** 2) * fcj * fck
        # end of shape convension
        # reshape to (conformations, ?)
        return ret.view(-1, atoms, atoms, atoms, self.per_species_angular_length())
    
    def _fill_angular_sum_by_zero(self, angular_sum_by_species, conformations):
        possible_keys = set([frozenset([i,j]) for i,j in itertools.combinations_with_replacement(self.species, 2)])
        complementary = possible_keys - set(angular_sum_by_species.keys())
        for i in complementary:
            angular_sum_by_species[i] = torch.zeros(conformations, self.per_species_angular_length(), dtype=self.dtype)

    def _radial_sum_indices(self, species):
        # returns a tensor of shape (1, atoms, atoms, 1, len(self.species))
        # which selects where the sum goes
        atoms = len(species)
        ret = torch.zeros(1, atoms, atoms, 1, len(self.species), dtype=self.dtype)
        for i in range(len(species)):
            for j in range(len(species)):
                if j == i:
                    continue
                key = species[j]
                ret[0,i,j,0,self.species_indices[key]] = 1
        return ret

    def __apply__(self, coordinates, species):
        conformations = coordinates.shape[0]
        atoms = coordinates.shape[1]

        R_vecs = coordinates.unsqueeze(1) - coordinates.unsqueeze(2)  # shape (conformations, atoms, atoms, 3)
        R_distances = torch.sqrt(torch.sum(R_vecs ** 2, dim=-1))  # shape (conformations, atoms, atoms)
        radial_terms = self._compute_radial_terms(R_distances)  # shape (conformations, atoms, atoms, self.per_species_radial_length())
        radial_sum_indices = self._radial_sum_indices(species)
        radial_aevs = radial_terms.unsqueeze(-1) * radial_sum_indices
        radial_aevs = torch.sum(radial_aevs, dim=2).view(conformations, atoms, -1)

        Rijk_distance_prods = R_distances.unsqueeze(2) * R_distances.unsqueeze(3)  # shape (conformations, atoms, atoms, atoms)
        Rijk_inner_prods = torch.sum(R_vecs.unsqueeze(2) * R_vecs.unsqueeze(3), dim=-1)  # shape (conformations, atoms, atoms, atoms)
        cos_angles = 0.95 * Rijk_inner_prods / Rijk_distance_prods  # shape (conformations, atoms, atoms, atoms)
        angles = torch.acos(cos_angles)   # shape (conformations, atoms, atoms, atoms)
        angular_terms = self._compute_angular_terms(angles, R_distances)  # shape (conformations, atoms, atoms, atoms, self.per_species_angular_length())
        angular_aevs = []
        for i in range(atoms):
            angular_sum_by_species = {}
            for j in range(atoms):
                if j == i:
                    continue
                for k in range(j+1,atoms):
                    if k == i:
                        continue
                    angular_term = angular_terms[:,i,j,k,:]
                    key = frozenset(species[j]+species[k])
                    if key in angular_sum_by_species:
                        angular_sum_by_species[key] += angular_term
                    else:
                        angular_sum_by_species[key] = angular_term
            self._fill_angular_sum_by_zero(angular_sum_by_species, conformations)
            angular_aev = torch.cat(list(angular_sum_by_species.values()), dim=1)
            angular_aevs.append(angular_aev)
        angular_aevs = torch.stack(angular_aevs, dim=1)

        return radial_aevs, angular_aevs