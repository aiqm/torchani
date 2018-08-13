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
        aev_computer = torchani.AEVComputer()
        nnp = torchani.models.NeuroChemNNP(aev_computer.species)
        shift_energy = torchani.EnergyShifter(aev_computer.species)
        self.model = torch.nn.Sequential(aev_computer, nnp, shift_energy)

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
        species, coordinates = torchani.padding.pad_and_batch(
            species_coordinates)
        energies = torch.cat(energies)
        _, energies_ = self.model((species, coordinates))
        max_diff = (energies - energies_).abs().max().item()
        self.assertLess(max_diff, self.tolerance)


if __name__ == '__main__':
    unittest.main()
