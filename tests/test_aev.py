import torch
import torchani
import unittest
import os
import pickle

path = os.path.dirname(os.path.realpath(__file__))
N = 97


class TestAEV(unittest.TestCase):

    def setUp(self):
        builtins = torchani.neurochem.Builtins()
        self.aev_computer = builtins.aev_computer
        self.radial_length = self.aev_computer.radial_length()
        self.tolerance = 1e-5

    def _assertAEVEqual(self, expected_radial, expected_angular, aev):
        radial = aev[..., :self.radial_length]
        angular = aev[..., self.radial_length:]
        radial_diff = expected_radial - radial
        radial_max_error = torch.max(torch.abs(radial_diff)).item()
        angular_diff = expected_angular - angular
        angular_max_error = torch.max(torch.abs(angular_diff)).item()
        self.assertLess(radial_max_error, self.tolerance)
        self.assertLess(angular_max_error, self.tolerance)

    def testIsomers(self):
        for i in range(N):
            datafile = os.path.join(path, 'test_data/{}'.format(i))
            with open(datafile, 'rb') as f:
                coordinates, species, expected_radial, expected_angular, _, _ \
                    = pickle.load(f)
                _, aev = self.aev_computer((species, coordinates))
                self._assertAEVEqual(expected_radial, expected_angular, aev)

    def testPadding(self):
        species_coordinates = []
        radial_angular = []
        for i in range(N):
            datafile = os.path.join(path, 'test_data/{}'.format(i))
            with open(datafile, 'rb') as f:
                coordinates, species, radial, angular, _, _ = pickle.load(f)
                species_coordinates.append((species, coordinates))
                radial_angular.append((radial, angular))
        species, coordinates = torchani.utils.pad_coordinates(
            species_coordinates)
        _, aev = self.aev_computer((species, coordinates))
        start = 0
        for expected_radial, expected_angular in radial_angular:
            conformations = expected_radial.shape[0]
            atoms = expected_radial.shape[1]
            aev_ = aev[start:start+conformations, 0:atoms]
            start += conformations
            self._assertAEVEqual(expected_radial, expected_angular, aev_)


class TestAEVASENeighborList(TestAEV):

    def setUp(self):
        super(TestAEVASENeighborList, self).setUp()
        self.aev_computer.neighborlist = torchani.ase.NeighborList()


if __name__ == '__main__':
    unittest.main()
