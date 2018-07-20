import torch
import numpy
import torchani
import unittest
import os
import pickle

path = os.path.dirname(os.path.realpath(__file__))
N = 97

class TestAEV(unittest.TestCase):

    def setUp(self, dtype=torchani.default_dtype):
        self.aev = torchani.SortedAEV(dtype=dtype, device=torch.device('cpu'))
        self.tolerance = 1e-5

    def _test_molecule(self, coordinates, species, expected_radial, expected_angular):
        radial, angular = self.aev(coordinates, species)
        radial_diff = expected_radial - radial
        radial_max_error = torch.max(torch.abs(radial_diff))
        angular_diff = expected_angular - angular
        angular_max_error = torch.max(torch.abs(angular_diff))
        self.assertLess(radial_max_error, self.tolerance)
        self.assertLess(angular_max_error, self.tolerance)

    def testGDB(self):
        for i in range(N):
            datafile = os.path.join(path, 'test_data/{}'.format(i))
            with open(datafile, 'rb') as f:
                coordinates, species, radial, angular, _, _ = pickle.load(f)
                self._test_molecule(coordinates, species, radial, angular)


if __name__ == '__main__':
    unittest.main()
