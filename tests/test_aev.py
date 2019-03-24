import torch
import torchani
import unittest
import os
import pickle
import random

path = os.path.dirname(os.path.realpath(__file__))
N = 97


class TestAEV(unittest.TestCase):

    def setUp(self):
        builtins = torchani.neurochem.Builtins()
        self.aev_computer = builtins.aev_computer
        self.radial_length = self.aev_computer.radial_length
        self.tolerance = 1e-5

    def random_skip(self):
        return False

    def transform(self, x):
        return x

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
            datafile = os.path.join(path, 'test_data/ANI1_subset/{}'.format(i))
            with open(datafile, 'rb') as f:
                coordinates, species, expected_radial, expected_angular, _, _ \
                    = pickle.load(f)
                coordinates = torch.from_numpy(coordinates)
                species = torch.from_numpy(species)
                expected_radial = torch.from_numpy(expected_radial)
                expected_angular = torch.from_numpy(expected_angular)
                coordinates = self.transform(coordinates)
                species = self.transform(species)
                expected_radial = self.transform(expected_radial)
                expected_angular = self.transform(expected_angular)
                _, aev = self.aev_computer((species, coordinates))
                self._assertAEVEqual(expected_radial, expected_angular, aev)

    def testPadding(self):
        species_coordinates = []
        radial_angular = []
        for i in range(N):
            datafile = os.path.join(path, 'test_data/ANI1_subset/{}'.format(i))
            with open(datafile, 'rb') as f:
                coordinates, species, radial, angular, _, _ = pickle.load(f)
                coordinates = torch.from_numpy(coordinates)
                species = torch.from_numpy(species)
                radial = torch.from_numpy(radial)
                angular = torch.from_numpy(angular)
                coordinates = self.transform(coordinates)
                species = self.transform(species)
                radial = self.transform(radial)
                angular = self.transform(angular)
                species_coordinates.append((species, coordinates))
                radial_angular.append((radial, angular))
        species, coordinates = torchani.utils.pad_coordinates(
            species_coordinates)
        _, aev = self.aev_computer((species, coordinates))
        start = 0
        for expected_radial, expected_angular in radial_angular:
            conformations = expected_radial.shape[0]
            atoms = expected_radial.shape[1]
            aev_ = aev[start:(start + conformations), 0:atoms]
            start += conformations
            self._assertAEVEqual(expected_radial, expected_angular, aev_)

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
                self._assertAEVEqual(radial, angular, aev)


class TestPBC(unittest.TestCase):

    def setUp(self):
        self.builtin = torchani.neurochem.Builtins()
        self.aev_computer = self.builtin.aev_computer.to(torch.double)

    def testTranslationalInvariancePBC(self):
        coordinates = torch.tensor(
            [[[0, 0, 0],
              [1, 0, 0],
              [0, 1, 0],
              [0, 0, 1],
              [0, 1, 1]]],
            dtype=torch.double, requires_grad=True)
        cell = torch.eye(3, dtype=torch.double) * 2
        species = torch.tensor([[1, 0, 0, 0, 0]], dtype=torch.long)
        pbc = torch.ones(3, dtype=torch.uint8)

        _, aev = self.aev_computer((species, coordinates, cell, pbc))

        for _ in range(100):
            translation = torch.randn(3, dtype=torch.double)
            _, aev2 = self.aev_computer((species, coordinates + translation, cell, pbc))
            self.assertTrue(torch.allclose(aev, aev2))

    def testPBCConnersSeeEachOther(self):
        species = torch.tensor([[0, 0]])
        cell = torch.eye(3, dtype=torch.double) * 10
        pbc = torch.ones(3, dtype=torch.uint8)
        shifts = torchani.aev.compute_shifts(cell, pbc, 1)
        print(shifts)

        xyz1 = torch.tensor([0.1, 0.1, 0.1])
        xyz2s = [
            torch.tensor([9.9, 0.0, 0.0]),
            torch.tensor([0.0, 9.9, 0.0]),
            torch.tensor([0.0, 0.0, 9.9]),
            torch.tensor([9.9, 9.9, 0.0]),
            torch.tensor([0.0, 9.9, 9.9]),
            torch.tensor([9.9, 0.0, 9.9]),
            torch.tensor([9.9, 9.9, 9.9]),
        ]

        for xyz2 in xyz2s:
            coordinates = torch.stack([xyz1, xyz2]).to(torch.double).unsqueeze(0)
            molecule_index, atom_index1, atom_index2, shifts = torchani.aev.neighbor_pairs(species == -1, coordinates, cell, shifts, 1)
            print(molecule_index, atom_index1, atom_index2, shifts)
            self.assertEqual(molecule_index.tolist(), [0])
            self.assertEqual(atom_index1.tolist(), [0])
            self.assertEqual(atom_index2.tolist(), [1])
            #TODO assert shift


if __name__ == '__main__':
    unittest.main()
