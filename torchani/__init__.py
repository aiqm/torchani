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
        # assign supported species indices
        self.species_indices = {}
        index = 0
        for i in self.species:
            self.species_indices[i] = index
            index += 1

    def radial_length(self):
        return len(self.constants['EtaR']) * len(self.constants['ShfR'])
    
    def angular_length(self):
        return len(self.constants['EtaA']) * len(self.constants['Zeta']) * len(self.constants['ShfA']) * len(self.constants['ShfZ'])

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

    def compute_angular_terms(self, angles, R_distances):
        # use broadcasting semantics to combine constants
        # shape convension (conformations, atoms, atoms, atoms, eta, zeta, radius_shift, angle_shift)
        atoms = angles.shape[1]
        angles = angles.view(-1, atoms, atoms, atoms, 1, 1, 1, 1)
        Rij = R_distances.view(-1, atoms, atoms, 1, 1, 1, 1, 1)
        Rik = R_distances.view(-1, atoms, 1, atoms, 1, 1, 1, 1)
        fcj = ani.cutoff_cosine(Rij, self.constants['Rca'])
        fck = ani.cutoff_cosine(Rik, self.constants['Rca'])
        eta = torch.Tensor(self.constants['EtaA']).type(self.dtype).view(1, 1, 1, 1, -1, 1, 1, 1)
        zeta = torch.Tensor(self.constants['Zeta']).type(self.dtype).view(1, 1, 1, 1, 1, -1, 1, 1)
        radius_shifts = torch.Tensor(self.constants['ShfA']).type(self.dtype).view(1, 1, 1, 1, 1, 1, -1, 1)
        angle_shifts = torch.Tensor(self.constants['ShfZ']).type(self.dtype).view(1, 1, 1, 1, 1, 1, 1, -1)
        ret = 2 * ((1 + torch.cos(angles - angle_shifts)) / 2) ** zeta * torch.exp(-eta * ((Rij + Rik) / 2 - radius_shifts) ** 2) * fcj * fck
        # end of shape convension
        # reshape to (conformations, ?)
        return ret.view(-1, atoms, atoms, atoms, self.angular_length())
    
    def fill_angular_sum_by_zero(self, angular_sum_by_species, conformations):
        possible_keys = set([frozenset([i,j]) for i,j in itertools.combinations_with_replacement(self.species, 2)])
        complementary = possible_keys - set(angular_sum_by_species.keys())
        for i in complementary:
            angular_sum_by_species[i] = torch.zeros(conformations, self.angular_length(), dtype=self.dtype)

    def radial_sum_indices(self, species):
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

    def compute_aev(self, coordinates, species):
        conformations = coordinates.shape[0]
        atoms = coordinates.shape[1]

        R_vecs = coordinates.unsqueeze(1) - coordinates.unsqueeze(2)  # shape (conformations, atoms, atoms, 3)
        R_distances = torch.sqrt(torch.sum(R_vecs ** 2, dim=-1))  # shape (conformations, atoms, atoms)
        radial_terms = self.compute_radial_terms(R_distances)  # shape (conformations, atoms, atoms, self.radial_length())
        radial_sum_indices = self.radial_sum_indices(species)
        radial_aevs = radial_terms.unsqueeze(-1) * radial_sum_indices
        radial_aevs = torch.sum(radial_aevs, dim=2).view(conformations, atoms, -1)

        Rijk_distance_prods = R_distances.unsqueeze(2) * R_distances.unsqueeze(3)  # shape (conformations, atoms, atoms, atoms)
        Rijk_inner_prods = torch.sum(R_vecs.unsqueeze(2) * R_vecs.unsqueeze(3), dim=-1)  # shape (conformations, atoms, atoms, atoms)
        cos_angles = 0.95 * Rijk_inner_prods / Rijk_distance_prods  # shape (conformations, atoms, atoms, atoms)
        angles = torch.acos(cos_angles)   # shape (conformations, atoms, atoms, atoms)
        angular_terms = self.compute_angular_terms(angles, R_distances)  # shape (conformations, atoms, atoms, atoms, self.angular_length())
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
            self.fill_angular_sum_by_zero(angular_sum_by_species, conformations)
            angular_aev = torch.cat(list(angular_sum_by_species.values()), dim=1)
            angular_aevs.append(angular_aev)
        angular_aevs = torch.stack(angular_aevs, dim=1)

        return radial_aevs, angular_aevs

    
    def shift_energy(self, energies, species):
        shift = 0
        for i in species:
            shift += self.self_energies[i]
        return energies - shift
