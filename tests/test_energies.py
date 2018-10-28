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
        builtins = torchani.neurochem.Builtins()
        self.aev_computer = builtins.aev_computer
        nnp = builtins.models[0]
        shift_energy = builtins.energy_shifter
        self.model = torch.nn.Sequential(self.aev_computer, nnp, shift_energy)

    def testIsomers(self):
        for i in range(N):
            datafile = os.path.join(path, 'test_data/{}'.format(i))
            with open(datafile, 'rb') as f:
                coordinates, species, _, _, energies, _ = pickle.load(f)
                _, energies_ = self.model((species, coordinates))
                max_diff = (energies - energies_).abs().max().item()
                self.assertLess(max_diff, self.tolerance)

    def testPadding(self):
        species_coordinates = []
        energies = []
        for i in range(N):
            datafile = os.path.join(path, 'test_data/{}'.format(i))
            with open(datafile, 'rb') as f:
                coordinates, species, _, _, e, _ = pickle.load(f)
                species_coordinates.append((species, coordinates))
                energies.append(e)
        species, coordinates = torchani.utils.pad_coordinates(
            species_coordinates)
        energies = torch.cat(energies)
        _, energies_ = self.model((species, coordinates))
        max_diff = (energies - energies_).abs().max().item()
        self.assertLess(max_diff, self.tolerance)


class TestEnergiesASEComputer(TestEnergies):

    def setUp(self):
        super(TestEnergiesASEComputer, self).setUp()
        self.aev_computer.neighborlist = torchani.ase.NeighborList()


if __name__ == '__main__':
    unittest.main()
