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
        aev_computer = torchani.AEVComputer()
        self.prepare = torchani.PrepareInput(aev_computer.species)
        nnp = torchani.models.NeuroChemNNP(aev_computer.species)
        self.model = torch.nn.Sequential(self.prepare, aev_computer, nnp)
        self.prepared2e = torch.nn.Sequential(aev_computer, nnp)

    def testIsomers(self):
        for i in range(N):
            datafile = os.path.join(path, 'test_data/{}'.format(i))
            with open(datafile, 'rb') as f:
                coordinates, species, _, _, _, forces = pickle.load(f)
                coordinates = torch.tensor(coordinates, requires_grad=True)
                _, energies = self.model((species, coordinates))
                derivative = torch.autograd.grad(energies.sum(), coordinates)[0]
                max_diff = (forces + derivative).abs().max().item()
                self.assertLess(max_diff, self.tolerance)

    def testPadding(self):
        species_coordinates = []
        coordinates_forces = []
        for i in range(N):
            datafile = os.path.join(path, 'test_data/{}'.format(i))
            with open(datafile, 'rb') as f:
                coordinates, species, _, _, _, forces = pickle.load(f)
                species, coordinates = self.prepare((species, coordinates))
                coordinates = torch.tensor(coordinates, requires_grad=True)
                species_coordinates.append((species, coordinates))
                coordinates_forces.append((coordinates, forces))
        species, coordinates = torchani.padding.pad_and_batch(species_coordinates)
        _, energies = self.prepared2e((species, coordinates))
        energies = energies.sum()
        for coordinates, forces in coordinates_forces:
            derivative = torch.autograd.grad(energies, coordinates,
                                             retain_graph=True)[0]
            max_diff = (forces + derivative).abs().max().item()
            self.assertLess(max_diff, self.tolerance)


if __name__ == '__main__':
    unittest.main()
