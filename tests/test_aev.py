import torch
import torchani
import unittest
import os
import pickle
import random
import copy

path = os.path.dirname(os.path.realpath(__file__))
N = 97


class TestAEV(unittest.TestCase):

    def setUp(self):
        builtins = torchani.neurochem.Builtins()
        self.aev_computer = builtins.aev_computer
        self.radial_length = self.aev_computer.radial_length
        self.tolerance = 1e-5

    def random_skip(self, prob=0):
        return random.random() < prob

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

    def testGradient(self):
        """Test validity of autodiff by comparing analytical and numerical
        gradients.
        """
        datafile = os.path.join(path, 'test_data/NIST/all')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Create local copy of aev_computer to avoid interference with other
        # tests.
        aev_computer = copy.deepcopy(self.aev_computer).to(device).to(torch.float64)
        with open(datafile, 'rb') as f:
            data = pickle.load(f)
            for coordinates, species, _, _, _, _ in data:
                if self.random_skip(prob=0.99):
                    continue
                coordinates = torch.from_numpy(coordinates).to(device).to(torch.float64)
                coordinates.requires_grad_(True)
                species = torch.from_numpy(species).to(device)

                # PyTorch gradcheck expects to test a funtion with inputs and
                # outputs of type torch.Tensor. The numerical estimation of
                # the deriviate involves making small modifications to the
                # input and observing how it affects the output. The species
                # tensor needs to be removed from the input so that gradcheck
                # does not attempt to estimate the gradient with respect to
                # species and fail.
                # Create simple function wrapper to handle this.
                def aev_forward_wrapper(coords):
                    # Return only the aev portion of the output.
                    return aev_computer((species, coords))[1]
                # Sanity Check: Forward wrapper returns aev without error.
                aev_forward_wrapper(coordinates)
                torch.autograd.gradcheck(
                    aev_forward_wrapper,
                    coordinates
                )


class TestAEVASENeighborList(TestAEV):

    def setUp(self):
        super(TestAEVASENeighborList, self).setUp()
        self.aev_computer.neighborlist = torchani.ase.NeighborList()

    def transform(self, x):
        """To reduce the size of test cases for faster test speed"""
        return x[:2, ...]

    def random_skip(self):
        """To reduce the size of test cases for faster test speed"""
        return random.random() < 0.95


if __name__ == '__main__':
    unittest.main()
