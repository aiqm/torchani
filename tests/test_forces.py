import torch
import torchani
import unittest
import os
import pickle

path = os.path.dirname(os.path.realpath(__file__))
N = 97


class TestForce(unittest.TestCase):

    def setUp(self, dtype=torchani.default_dtype,
              device=torchani.default_device):
        self.tolerance = 1e-5
        self.aev_computer = torchani.SortedAEV(
            dtype=dtype, device=torch.device('cpu'))
        self.nnp = torchani.models.NeuroChemNNP(
            self.aev_computer, derivative=True)

    def _test_molecule(self, coordinates, species, forces):
        _, derivative = self.nnp(coordinates, species)
        max_diff = (forces + derivative).abs().max().item()
        self.assertLess(max_diff, self.tolerance)

    def testGDB(self):
        for i in range(N):
            datafile = os.path.join(path, 'test_data/{}'.format(i))
            with open(datafile, 'rb') as f:
                coordinates, species, _, _, _, forces = pickle.load(f)
                self._test_molecule(coordinates, species, forces)


if __name__ == '__main__':
    unittest.main()
