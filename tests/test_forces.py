import torch
import torchani
import unittest
import os
import pickle

path = os.path.dirname(os.path.realpath(__file__))
N = 97


class TestForce(unittest.TestCase):

    def setUp(self):
        self.tolerance = 1e-5
        ani1x = torchani.models.ANI1x()
        self.aev_computer = ani1x.aev_computer
        self.nnp = ani1x.neural_networks[0]
        self.model = torchani.nn.Sequential(self.aev_computer, self.nnp)

    def random_skip(self):
        return False

    def transform(self, x):
        return x

    def testIsomers(self):
        for i in range(N):
            datafile = os.path.join(path, 'test_data/ANI1_subset/{}'.format(i))
            with open(datafile, 'rb') as f:
                coordinates, species, _, _, _, forces = pickle.load(f)
                coordinates = torch.from_numpy(coordinates)
                species = torch.from_numpy(species)
                forces = torch.from_numpy(forces)
                coordinates = self.transform(coordinates)
                species = self.transform(species)
                forces = self.transform(forces)
                coordinates.requires_grad_(True)
                _, energies = self.model((species, coordinates))
                derivative = torch.autograd.grad(energies.sum(),
                                                 coordinates)[0]
                max_diff = (forces + derivative).abs().max().item()
                self.assertLess(max_diff, self.tolerance)

    def testPadding(self):
        species_coordinates = []
        coordinates_forces = []
        for i in range(N):
            datafile = os.path.join(path, 'test_data/ANI1_subset/{}'.format(i))
            with open(datafile, 'rb') as f:
                coordinates, species, _, _, _, forces = pickle.load(f)
                coordinates = torch.from_numpy(coordinates)
                species = torch.from_numpy(species)
                forces = torch.from_numpy(forces)
                coordinates = self.transform(coordinates)
                species = self.transform(species)
                forces = self.transform(forces)
                coordinates.requires_grad_(True)
                species_coordinates.append({'species': species, 'coordinates': coordinates})
        species_coordinates = torchani.utils.pad_atomic_properties(
            species_coordinates)
        _, energies = self.model((species_coordinates['species'], species_coordinates['coordinates']))
        energies = energies.sum()
        for coordinates, forces in coordinates_forces:
            derivative = torch.autograd.grad(energies, coordinates,
                                             retain_graph=True)[0]
            max_diff = (forces + derivative).abs().max().item()
            self.assertLess(max_diff, self.tolerance)


if __name__ == '__main__':
    unittest.main()
