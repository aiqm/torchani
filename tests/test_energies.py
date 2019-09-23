import torch
import torchani
import unittest
import os
import pickle


path = os.path.dirname(os.path.realpath(__file__))
N = 97


class TestEnergies(unittest.TestCase):

    def setUp(self):
        self.tolerance = 5e-5
        ani1x = torchani.models.ANI1x()
        self.aev_computer = ani1x.aev_computer
        self.nnp = ani1x.neural_networks[0]
        self.energy_shifter = ani1x.energy_shifter
        self.nn = torchani.nn.Sequential(self.nnp, self.energy_shifter)
        self.model = torchani.nn.Sequential(self.aev_computer, self.nnp, self.energy_shifter)

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
                species_coordinates.append({'species': species, 'coordinates': coordinates})
                energies.append(e)
        species_coordinates = torchani.utils.pad_atomic_properties(
            species_coordinates)
        energies = torch.cat(energies)
        _, energies_ = self.model((species_coordinates['species'], species_coordinates['coordinates']))
        max_diff = (energies - energies_).abs().max().item()
        self.assertLess(max_diff, self.tolerance)


class TestEnergiesEnergyShifterJIT(TestEnergies):
    def setUp(self):
        super().setUp()
        self.energy_shifter = torch.jit.script(self.energy_shifter)
        self.nn = torchani.nn.Sequential(self.nnp, self.energy_shifter)
        self.model = torchani.nn.Sequential(self.aev_computer, self.nnp, self.energy_shifter)


if __name__ == '__main__':
    unittest.main()
