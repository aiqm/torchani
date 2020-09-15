import os
import torch
import pickle
import unittest
from common_aev_test import _TestAEVBase

path = os.path.dirname(os.path.realpath(__file__))


class TestAEVTripeptideMD(_TestAEVBase):

    def testTripeptideMD(self):
        tol = 5e-6
        for i in range(100):
            datafile = os.path.join(path, 'test_data/tripeptide-md/{}.dat'.format(i))
            with open(datafile, 'rb') as f:
                coordinates, species, expected_radial, expected_angular, _, _, _, _ \
                    = pickle.load(f)
                coordinates = torch.from_numpy(coordinates).float().unsqueeze(0)
                species = torch.from_numpy(species).unsqueeze(0)
                expected_radial = torch.from_numpy(expected_radial).float().unsqueeze(0)
                expected_angular = torch.from_numpy(expected_angular).float().unsqueeze(0)
                _, aev = self.aev_computer((species, coordinates))
                self.assertAEVEqual(expected_radial, expected_angular, aev, tol)


if __name__ == '__main__':
    unittest.main()
