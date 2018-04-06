import torch
import numpy
import itertools
import time

class ani:

    def __init__(self, const_file, self_energy_file, dtype=torch.cuda.float32):
        # load constants
        self.constants = {}
        self.dtype = dtype
        with open(const_file) as f:
            for i in f:
                try:
                    line = [x.strip() for x in i.split('=')]
                    name = line[0]
                    value = line[1]
                    if name == 'Rcr' or name == 'Rca':
                        self.constants[name] = float(value)
                    elif name in ['EtaR', 'ShfR', 'Zeta', 'ShfZ', 'EtaA', 'ShfA']:
                        value = [ float(x.strip()) for x in value.replace('[','').replace(']','').split(',')]
                        self.constants[name] = value
                    elif name == 'Atyp':
                        value = [ x.strip() for x in value.replace('[','').replace(']','').split(',')]
                        self.species = value
                except:
                    pass  # ignore unrecognizable line
        # load self energies
        self.self_energies = {}
        with open(self_energy_file) as f:
            for i in f:
                try:
                    line = [x.strip() for x in i.split('=')]
                    name = line[0].split(',')[0].strip()
                    value = float(line[1])
                    self.self_energies[name] = value
                except:
                    pass  # ignore unrecognizable line

    def radial_length(self):
        return len(self.constants['EtaR']) * len(self.constants['ShfR'])
    
    def angular_length(self):
        return len(self.constants['EtaA']) * len(self.constants['Zeta']) * len(self.constants['ShfA']) * len(self.constants['ShfZ'])

    def angular_iter(self):
        return itertools.product(self.constants['EtaA'], self.constants['Zeta'], self.constants['ShfA'], self.constants['ShfZ'])

    @staticmethod
    def cutoff_cosine(distances, cutoff):
        return torch.where(distances <= cutoff, 0.5 * torch.cos(numpy.pi * distances / cutoff) + 0.5, torch.zeros_like(distances))

    def compute_radial_terms(self, distances):
        # use broadcasting semantics to combine constants
        # shape convension (conformations, atoms, atoms, eta, radius_shift)
        atoms = distances.shape[1]
        distances = distances.view(-1, atoms, atoms, 1, 1)
        fc = ani.cutoff_cosine(distances, self.constants['Rcr'])
        eta = torch.Tensor(self.constants['EtaR']).type(self.dtype).view(1,1,1,-1,1)
        radius_shift = torch.Tensor(self.constants['ShfR']).type(self.dtype).view(1,1,1,1,-1)
        ret = 0.25 * torch.exp(-eta * (distances - radius_shift)**2) * fc
        # end of shape convension
        # reshape to (conformations, atoms, atoms, ?)
        return ret.view(-1, atoms, atoms, self.radial_length())

    def compute_angular_term(self, angle, Rij, Rik):
        # use broadcasting semantics to combine constants
        # shape convension (conformations, eta, zeta, radius_shift, angle_shift)
        angle = angle.view(-1, 1, 1, 1, 1)
        Rij = Rij.view(-1, 1, 1, 1, 1)
        Rik = Rik.view(-1, 1, 1, 1, 1)
        fcj = ani.cutoff_cosine(Rij, self.constants['Rca'])
        fck = ani.cutoff_cosine(Rik, self.constants['Rca'])
        eta = torch.Tensor(self.constants['EtaA']).type(self.dtype).view(1, -1, 1, 1, 1)
        zeta = torch.Tensor(self.constants['Zeta']).type(self.dtype).view(1, 1, -1, 1, 1)
        radius_shift = torch.Tensor(self.constants['ShfA']).type(self.dtype).view(1, 1, 1, -1, 1)
        angle_shift = torch.Tensor(self.constants['ShfZ']).type(self.dtype).view(1, 1, 1, 1, -1)
        ret = 2 * ((1 + torch.cos(angle - angle_shift)) / 2) ** zeta * torch.exp(-eta * ((Rij + Rik) / 2 - radius_shift) ** 2) * fcj * fck
        # end of shape convension
        # reshape to (conformations, ?)
        return ret.view(-1, self.angular_length())

    def fill_radial_sum_by_zero(self, radial_sum_by_species, conformations):
        complementary = set(self.species) - set(radial_sum_by_species.keys())
        for i in complementary:
            radial_sum_by_species[i] = torch.zeros(conformations, self.radial_length(), dtype=self.dtype)
    
    def fill_angular_sum_by_zero(self, angular_sum_by_species, conformations):
        possible_keys = set([frozenset([i,j]) for i,j in itertools.combinations(self.species, 2)] + [frozenset([i]) for i in self.species])
        complementary = possible_keys - set(angular_sum_by_species.keys())
        for i in complementary:
            angular_sum_by_species[i] = torch.zeros(conformations, self.angular_length(), dtype=self.dtype)

    @staticmethod
    def atom_pair_index(atoms, i, j):
        return 1

    @staticmethod
    def pairs(atoms):
        return atoms * (atoms - 1) / 2

    def compute_aev(self, coordinates, species):
        conformations = coordinates.shape[0]
        atoms = coordinates.shape[1]
        radial_aevs = []
        angular_aevs = []
        R_vecs = coordinates.unsqueeze(1) - coordinates.unsqueeze(2)  # shape (conformations, atoms, atoms, 3)
        pair_idx_mask = (torch.ones(atoms,atoms).triu() == 1).unsqueeze(0).unsqueeze(-1).expand(conformations,-1,-1,3)
        R_distances = torch.sqrt(torch.sum(R_vecs ** 2, dim=-1))  # shape (conformations, atoms, atoms)
        radial_terms = self.compute_radial_terms(R_distances)  # shape (conformations, atoms, atoms, self.radial_length())
        Rijk_distance_prods = R_distances.unsqueeze(2) * R_distances.unsqueeze(3)  # shape (conformations, atoms, atoms, atoms)
        Rijk_inner_prods = torch.sum(R_vecs.unsqueeze(2) * R_vecs.unsqueeze(3), dim=-1)  # shape (conformations, atoms, atoms, atoms)
        cos_angles = 0.95 * Rijk_inner_prods / Rijk_distance_prods  # shape (conformations, atoms, atoms, atoms)
        angles = torch.acos(cos_angles)
        for i in range(atoms):
            radial_sum_by_species = {}
            angular_sum_by_species = {}
            for j in range(atoms):
                if j == i:
                    continue
                Rij_distance = R_distances[:,i,j]
                radial_term = radial_terms[:,i,j,:]
                if species[j] in radial_sum_by_species:
                    radial_sum_by_species[species[j]] += radial_term
                else:
                    radial_sum_by_species[species[j]] = radial_term
                for k in range(j+1,atoms):
                    if k == i:
                        continue
                    Rik_distance = R_distances[:,i,k]
                    angle = angles[:,i,j,k]
                    angular_term = self.compute_angular_term(angle, Rij_distance, Rik_distance)
                    key = frozenset(species[j]+species[k])
                    if key in angular_sum_by_species:
                        angular_sum_by_species[key] += angular_term
                    else:
                        angular_sum_by_species[key] = angular_term
            self.fill_radial_sum_by_zero(radial_sum_by_species, conformations)
            self.fill_angular_sum_by_zero(angular_sum_by_species, conformations)
            radial_aev = torch.cat(list(radial_sum_by_species.values()), dim=1)  # shape (conformations, torchani.radial.length)
            angular_aev = torch.cat(list(angular_sum_by_species.values()), dim=1)  # shape (conformations, torchani.angular.length)
            radial_aevs.append(radial_aev)
            angular_aevs.append(angular_aev)
        radial_aevs = torch.stack(radial_aevs, dim=1)  # shape (conformations, atoms, torchani.radial.length)
        angular_aevs = torch.stack(angular_aevs, dim=1)  # shape (conformations, atoms, torchani.angular.length)
        return radial_aevs, angular_aevs

    
    def shift_energy(self, energies, species):
        shift = 0
        for i in species:
            shift += self.self_energies[i]
        return energies - shift