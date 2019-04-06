import torch
import torchani
import unittest
import os
import pickle
import math
import random


path = os.path.dirname(os.path.realpath(__file__))
N = 97


class TestEnergies(unittest.TestCase):

    def setUp(self):
        self.tolerance = 5e-5
        builtins = torchani.neurochem.Builtins()
        self.aev_computer = builtins.aev_computer
        nnp = builtins.models[0]
        shift_energy = builtins.energy_shifter
        self.model = torch.nn.Sequential(self.aev_computer, nnp, shift_energy)

    def random_skip(self):
        return False

    def transform(self, x):
        return x

    def testIsomers(self):
        for i in range(N):
            datafile = os.path.join(path, 'test_data/ANI1_subset/{}'.format(i))
            with open(datafile, 'rb') as f:
                coordinates, species, _, _, energies, _ = pickle.load(f)
                coordinates = torch.from_numpy(coordinates).to(torch.float)
                species = torch.from_numpy(species)
                energies = torch.from_numpy(energies).to(torch.float)
                coordinates = self.transform(coordinates)
                species = self.transform(species)
                energies = self.transform(energies)
                _, energies_ = self.model((species, coordinates))
                max_diff = (energies - energies_).abs().max().item()
                self.assertLess(max_diff, self.tolerance)

    def testPadding(self):
        species_coordinates = []
        energies = []
        for i in range(N):
            datafile = os.path.join(path, 'test_data/ANI1_subset/{}'.format(i))
            with open(datafile, 'rb') as f:
                coordinates, species, _, _, e, _ = pickle.load(f)
                coordinates = torch.from_numpy(coordinates).to(torch.float)
                species = torch.from_numpy(species)
                e = torch.from_numpy(e).to(torch.float)
                coordinates = self.transform(coordinates)
                species = self.transform(species)
                e = self.transform(e)
                species_coordinates.append((species, coordinates))
                energies.append(e)
        species, coordinates = torchani.utils.pad_coordinates(
            species_coordinates)
        energies = torch.cat(energies)
        _, energies_ = self.model((species, coordinates))
        max_diff = (energies - energies_).abs().max().item()
        self.assertLess(max_diff, self.tolerance)

    def testNIST(self):
        datafile = os.path.join(path, 'test_data/NIST/all')
        with open(datafile, 'rb') as f:
            data = pickle.load(f)
            for coordinates, species, _, _, e, _ in data:
                if self.random_skip():
                    continue
                coordinates = torch.from_numpy(coordinates).to(torch.float)
                species = torch.from_numpy(species)
                energies = torch.from_numpy(e).to(torch.float)
                _, energies_ = self.model((species, coordinates))
                natoms = coordinates.shape[1]
                max_diff = (energies - energies_).abs().max().item()
                self.assertLess(max_diff / math.sqrt(natoms), self.tolerance)


class TestEnergiesASEComputer(TestEnergies):

    def setUp(self):
        super(TestEnergiesASEComputer, self).setUp()

    def transform(self, x):
        """To reduce the size of test cases for faster test speed"""
        return x[:2, ...]

    def random_skip(self):
        """To reduce the size of test cases for faster test speed"""
        return random.random() < 0.95


if __name__ == '__main__':
    unittest.main()
