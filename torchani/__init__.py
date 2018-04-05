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

    def compute_radial_term(self, distances):
        # use broadcasting semantics to combine constants
        # shape convension (conformations, eta, radius_shift)
        distances = distances.reshape(-1, 1, 1)
        fc = ani.cutoff_cosine(distances, self.constants['Rcr'])
        eta = torch.Tensor(self.constants['EtaR']).type(self.dtype).view(1,-1,1)
        radius_shift = torch.Tensor(self.constants['ShfR']).type(self.dtype).view(1,1,-1)
        ret = 0.25 * torch.exp(-eta * (distances - radius_shift)**2) * fc
        # end of shape convension
        # reshape to (conformations, ?)
        return ret.view(-1, self.radial_length())

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

    def compute_aev(self, coordinates, species):
        conformations = coordinates.shape[0]
        atoms = coordinates.shape[1]
        radial_aevs = []
        angular_aevs = []
        for i in range(atoms):
            xyz_i = coordinates[:,i,:]  # shape (conformations, 3)
            radial_sum_by_species = {}
            angular_sum_by_species = {}
            for j in range(atoms):
                if j == i:
                    continue
                xyz_j = coordinates[:,j,:]  # shape (conformations, 3)
                Rij_vec = xyz_j - xyz_i  # shape (conformations, 3)
                Rij_distance = torch.sqrt(torch.sum(Rij_vec ** 2, dim=1))  # shape (conformations,)
                radial_term = self.compute_radial_term(Rij_distance)  # shape (conformations, torchani.radial.length)
                if species[j] in radial_sum_by_species:
                    radial_sum_by_species[species[j]] += radial_term
                else:
                    radial_sum_by_species[species[j]] = radial_term
                for k in range(j+1,atoms):
                    if k == i:
                        continue
                    xyz_k = coordinates[:,k,:]
                    Rik_vec = xyz_k - xyz_i  # shape (conformations, 3)
                    Rik_distance = torch.sqrt(torch.sum(Rik_vec ** 2, dim=1))  # shape (conformations,)
                    cos_angle = 0.95 * torch.sum(Rij_vec * Rik_vec, dim=1) / (Rik_distance * Rij_distance)  # shape (conformations,)
                    angle = torch.acos(cos_angle)  # shape (conformations,)
                    angular_term = self.compute_angular_term(angle, Rij_distance, Rik_distance)  # shape (conformations, torchani.angular.length)
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