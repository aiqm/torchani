import os
import torch
import pickle
import unittest
from common_aev_test import _TestAEVBase

path = os.path.dirname(os.path.realpath(__file__))


class TestAEVNIST(_TestAEVBase):

    def testNIST(self):
        datafile = os.path.join(path, 'test_data/NIST/all')
        with open(datafile, 'rb') as f:
            data = pickle.load(f)
            for coordinates, species, radial, angular, _, _ in data:
                if self.random_skip():
                    continue
                coordinates = torch.from_numpy(coordinates).to(torch.float)
                species = torch.from_numpy(species)
                radial = torch.from_numpy(radial).to(torch.float)
                angular = torch.from_numpy(angular).to(torch.float)
                _, aev = self.aev_computer((species, coordinates))
                self.assertAEVEqual(radial, angular, aev)


if __name__ == '__main__':
    unittest.main()
