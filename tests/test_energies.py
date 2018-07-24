import torch
import torchani
import unittest
import os
import pickle


path = os.path.dirname(os.path.realpath(__file__))
N = 97


class TestEnergies(unittest.TestCase):

    def setUp(self, dtype=torchani.default_dtype,
              device=torchani.default_device):
        self.tolerance = 5e-5
        self.aev_computer = torchani.SortedAEV(
            dtype=dtype, device=torch.device('cpu'))
        self.nnp = torchani.models.NeuroChemNNP(self.aev_computer)

    def _test_molecule(self, coordinates, species, energies):
        shift_energy = torchani.EnergyShifter(torchani.buildin_sae_file)
        energies_ = self.nnp(coordinates, species).squeeze()
        energies_ = shift_energy.add_sae(energies_, species)
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
