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
        builtins = torchani.neurochem.Builtins()
        aev_computer = builtins.aev_computer
        nnp = builtins.models[0]
        self.model = torch.nn.Sequential(aev_computer, nnp)

    def testIsomers(self):
        for i in range(N):
            datafile = os.path.join(path, 'test_data/{}'.format(i))
            with open(datafile, 'rb') as f:
                coordinates, species, _, _, _, forces = pickle.load(f)
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
            datafile = os.path.join(path, 'test_data/{}'.format(i))
            with open(datafile, 'rb') as f:
                coordinates, species, _, _, _, forces = pickle.load(f)
                coordinates.requires_grad_(True)
                species_coordinates.append((species, coordinates))
                coordinates_forces.append((coordinates, forces))
        species, coordinates = torchani.utils.pad_coordinates(
            species_coordinates)
        _, energies = self.model((species, coordinates))
        energies = energies.sum()
        for coordinates, forces in coordinates_forces:
            derivative = torch.autograd.grad(energies, coordinates,
                                             retain_graph=True)[0]
            max_diff = (forces + derivative).abs().max().item()
            self.assertLess(max_diff, self.tolerance)


if __name__ == '__main__':
    unittest.main()
