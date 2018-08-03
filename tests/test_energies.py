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
        aev_computer = torchani.SortedAEV()
        prepare = torchani.PrepareInput(aev_computer.species)
        nnp = torchani.models.NeuroChemNNP(aev_computer.species)
        self.model = torch.nn.Sequential(prepare, aev_computer, nnp)

    def _test_molecule(self, coordinates, species, energies):
        shift_energy = torchani.EnergyShifter()
        _, energies_ = self.model((species, coordinates))
        energies_ = shift_energy.add_sae(energies_.squeeze(), species)
        max_diff = (energies - energies_).abs().max().item()
        self.assertLess(max_diff, self.tolerance)

    def testGDB(self):
        for i in range(N):
            datafile = os.path.join(path, 'test_data/{}'.format(i))
            with open(datafile, 'rb') as f:
                coordinates, species, _, _, energies, _ = pickle.load(f)
                self._test_molecule(coordinates, species, energies)


if __name__ == '__main__':
    unittest.main()
