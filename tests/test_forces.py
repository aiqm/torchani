import torch
import torchani
import unittest
import os
import pickle
import random

path = os.path.dirname(os.path.realpath(__file__))
N = 97


class TestForce(unittest.TestCase):

    def setUp(self):
        self.tolerance = 1e-5
        builtins = torchani.neurochem.Builtins()
        self.aev_computer = builtins.aev_computer
        nnp = builtins.models[0]
        self.model = torch.nn.Sequential(self.aev_computer, nnp)

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

    @unittest.skipIf(True, "WIP")
    def testBenzeneMD(self):
        tolerance = 1e-6
        for i in range(100):
            datafile = os.path.join(path, 'test_data/benzene-md/{}.dat'.format(i))
            with open(datafile, 'rb') as f:
                coordinates, species, _, _, _, forces, cell, pbc \
                    = pickle.load(f)
                coordinates = torch.from_numpy(coordinates).float().unsqueeze(0).requires_grad_(True)
                species = torch.from_numpy(species).unsqueeze(0)
                cell = torch.from_numpy(cell).float()
                pbc = torch.from_numpy(pbc)
                forces = torch.from_numpy(forces)
                coordinates = torchani.utils.map2central(cell, coordinates, pbc)
                coordinates = self.transform(coordinates)
                species = self.transform(species)
                forces = self.transform(forces)
                _, energies_ = self.model((species, coordinates, cell, pbc))
                derivative = torch.autograd.grad(energies_.sum(),
                                                 coordinates)[0]
                max_diff = (forces + derivative).abs().max().item()
                self.assertLess(max_diff, tolerance)

    def testTripeptideMD(self):
        tolerance = 2e-6
        for i in range(100):
            datafile = os.path.join(path, 'test_data/tripeptide-md/{}.dat'.format(i))
            with open(datafile, 'rb') as f:
                coordinates, species, _, _, _, forces, _, _ \
                    = pickle.load(f)
                coordinates = torch.from_numpy(coordinates).float().unsqueeze(0).requires_grad_(True)
                species = torch.from_numpy(species).unsqueeze(0)
                forces = torch.from_numpy(forces)
                coordinates = self.transform(coordinates)
                species = self.transform(species)
                forces = self.transform(forces)
                _, energies_ = self.model((species, coordinates))
                derivative = torch.autograd.grad(energies_.sum(),
                                                 coordinates)[0]
                max_diff = (forces + derivative).abs().max().item()
                self.assertLess(max_diff, tolerance)

    def testNIST(self):
        datafile = os.path.join(path, 'test_data/NIST/all')
        with open(datafile, 'rb') as f:
            data = pickle.load(f)
            for coordinates, species, _, _, _, forces in data:
                if self.random_skip():
                    continue
                coordinates = torch.from_numpy(coordinates).to(torch.float) \
                                   .requires_grad_(True)
                species = torch.from_numpy(species)
                forces = torch.from_numpy(forces).to(torch.float)
                _, energies = self.model((species, coordinates))
                derivative = torch.autograd.grad(energies.sum(),
                                                 coordinates)[0]
                max_diff = (forces + derivative).abs().max().item()
                self.assertLess(max_diff, self.tolerance)


class TestForceASEComputer(TestForce):

    def setUp(self):
        super(TestForceASEComputer, self).setUp()

    def transform(self, x):
        """To reduce the size of test cases for faster test speed"""
        return x[:2, ...]

    def random_skip(self):
        """To reduce the size of test cases for faster test speed"""
        return random.random() < 0.95


if __name__ == '__main__':
    unittest.main()
