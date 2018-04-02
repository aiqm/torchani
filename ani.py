import torch
import numpy
import itertools

#### work around: for pytorch < 0.4, torch.where is not implemented...
#### TODO: remove this after pytorch releases 0.4
def where(cond, yes, no):
    cond = cond.float()
    return cond * yes + (1 - cond) * no

torch.where = where
####

class torchani:
    
    species = [ 'C', 'H', 'N', 'O' ]

    self_energies = {
        'H': -0.500607632585,
        'C': -37.8302333826,
        'N': -54.5680045287,
        'O': -75.0362229210,
    }

    class radial:

        cutoff = 4.6
        eta = [24]
        radius_shift = [ 1.000,1.225,1.450,1.675,1.900,2.125,2.350,2.575,2.800,3.025,3.250,3.475,3.700,3.925,4.150,4.375 ]
        length = len(eta) * len(radius_shift)

        def __iter__(self):
            return itertools.product(self.eta, self.radius_shift)


    class angular:

        cutoff = 3.1
        eta = [8]
        zeta = [32]
        radius_shift = [ 1.000,1.525,2.050,2.575 ]
        angle_shift = [ 0.0625*numpy.pi, 0.1875*numpy.pi, 0.3125*numpy.pi, 0.4375*numpy.pi, 0.5625*numpy.pi, 0.6875*numpy.pi, 0.8125*numpy.pi, 0.9375*numpy.pi ]
        length = len(eta) * len(zeta) * len(radius_shift) * len(angle_shift)

        def __iter__(self):
            return itertools.product(self.eta, self.zeta, self.radius_shift, self.angle_shift)

    def cutoff_cosine(distances, cutoff):
        return torch.where(distances <= cutoff, 0.5 * torch.cos(numpy.pi * distances / cutoff) + 0.5, torch.zeros_like(distances))

    def compute_radial_term(distances):
        fc = torchani.cutoff_cosine(distances, torchani.radial.cutoff)
        tensors = [ torch.exp(-eta * (distances - radius_shift)**2) * fc for eta, radius_shift in torchani.radial()]
        return torch.stack(tensors, dim=1)

    def compute_angular_term(angle, Rij, Rik):
        fcj = torchani.cutoff_cosine(Rij, torchani.angular.cutoff)
        fck = torchani.cutoff_cosine(Rik, torchani.angular.cutoff)
        tensors = [ 2 * ((1 + torch.cos(angle - angle_shift)) / 2) ** zeta * torch.exp(-eta * ((Rij + Rik)/2 - radius_shift)**2) * fcj * fck for eta, zeta, radius_shift, angle_shift in torchani.angular() ]
        return torch.stack(tensors, dim=1)

    def radial_zeros_by_species(conformations):
        zeros = {}
        for i in torchani.species:
            zeros[i] = torch.zeros(conformations, torchani.radial.length).cuda()
        return zeros

    def angular_zeros_by_species(conformations):
        zeros = {}
        for i,j in itertools.combinations(torchani.species, 2):
            zeros[frozenset([i,j])] = torch.zeros(conformations, torchani.angular.length).cuda()
        for i in torchani.species:
            zeros[frozenset([i])] = torch.zeros(conformations, torchani.angular.length).cuda()
        return zeros

    def compute_aev(coordinates, species):
        conformations = coordinates.shape[0]
        atoms = coordinates.shape[1]
        full_aevs = []
        for i in range(atoms):
            xyz_i = coordinates[:,i,:]  # shape (conformations, 3)
            radial_sum_by_species = torchani.radial_zeros_by_species(conformations)
            angular_sum_by_species = torchani.angular_zeros_by_species(conformations)
            for j in range(atoms):
                if j == i:
                    continue
                xyz_j = coordinates[:,j,:]  # shape (conformations, 3)
                Rij_vec = xyz_j - xyz_i  # shape (conformations, 3)
                Rij_distance = torch.sqrt(torch.sum(Rij_vec * Rij_vec, dim=1))  # shape (conformations,)
                radial_term = torchani.compute_radial_term(Rij_distance)  # shape (conformations, torchani.radial.length)
                radial_sum_by_species[species[i]] += radial_term
                for k in range(j+1,atoms):
                    if k == i:
                        continue
                    xyz_k = coordinates[:,k,:]
                    Rik_vec = xyz_k - xyz_i  # shape (conformations, 3)
                    Rik_distance = torch.sqrt(torch.sum(Rik_vec * Rik_vec, dim=1))  # shape (conformations,)
                    cos_angle = torch.sum(Rij_vec * Rik_vec, dim=1) / (Rik_distance * Rij_distance)  # shape (conformations,)
                    angle = torch.acos(cos_angle)  # shape (conformations,)
                    angular_term = torchani.compute_angular_term(angle, Rij_distance, Rik_distance)  # shape (conformations, torchani.angular.length)
                    angular_sum_by_species[frozenset(species[j]+species[k])] += angular_term
            radial_aev = torch.cat(list(radial_sum_by_species.values()), dim=1)  # shape (conformations, torchani.radial.length)
            angular_aev = torch.cat(list(angular_sum_by_species.values()), dim=1)  # shape (conformations, torchani.angular.length)
            full_aev = torch.cat([radial_aev, angular_aev], dim=1)  # shape (conformations, torchani.radial.length + torchani.angular.length)
            full_aevs.append(full_aev)
        return torch.stack(full_aevs)  # shape (atoms, conformations, torchani.radial.length + torchani.angular.length)

    
    def shift_energy(energies, species):
        shift = 0
        for i in species:
            shift += torchani.self_energies[i]
        return energies - shift