import torch
import torchani
import unittest
import os
import pickle
import math


path = os.path.dirname(os.path.realpath(__file__))
N = 97


class TestEnergies(unittest.TestCase):

    def setUp(self):
        self.tolerance = 5e-5
        ani1x = torchani.models.ANI1x()
        self.aev_computer = ani1x.aev_computer
        nnp = ani1x.neural_networks[0]
        shift_energy = ani1x.energy_shifter
        self.nn = torch.nn.Sequential(nnp, shift_energy)
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

    def testBenzeneMD(self):
        tolerance = 1e-5
        for i in range(10):
            datafile = os.path.join(path, 'test_data/benzene-md/{}.dat'.format(i))
            with open(datafile, 'rb') as f:
                coordinates, species, _, _, energies, _, cell, pbc \
                    = pickle.load(f)
                coordinates = torch.from_numpy(coordinates).float().unsqueeze(0)
                species = torch.from_numpy(species).unsqueeze(0)
                cell = torch.from_numpy(cell).float()
                pbc = torch.from_numpy(pbc)
                coordinates = torchani.utils.map2central(cell, coordinates, pbc)
                coordinates = self.transform(coordinates)
                species = self.transform(species)
                energies = self.transform(energies)
                _, aev = self.aev_computer((species, coordinates), cell=cell, pbc=pbc)
                _, energies_ = self.nn((species, aev))
                max_diff = (energies - energies_).abs().max().item()
                self.assertLess(max_diff, tolerance)

    def testTripeptideMD(self):
        tolerance = 2e-4
        for i in range(100):
            datafile = os.path.join(path, 'test_data/tripeptide-md/{}.dat'.format(i))
            with open(datafile, 'rb') as f:
                coordinates, species, _, _, energies, _, _, _ \
                    = pickle.load(f)
                coordinates = torch.from_numpy(coordinates).float().unsqueeze(0)
                species = torch.from_numpy(species).unsqueeze(0)
                coordinates = self.transform(coordinates)
                species = self.transform(species)
                energies = self.transform(energies)
                _, energies_ = self.model((species, coordinates))
                max_diff = (energies - energies_).abs().max().item()
                self.assertLess(max_diff, tolerance)

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
                species_coordinates.append({'species': species, 'coordinates': coordinates})
                energies.append(e)
        species_coordinates = torchani.utils.pad_atomic_properties(
            species_coordinates)
        energies = torch.cat(energies)
        _, energies_ = self.model((species_coordinates['species'], species_coordinates['coordinates']))
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


if __name__ == '__main__':
    unittest.main()
