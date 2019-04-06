import torch
import torchani
import unittest
import os
import pickle
import random
import copy
import itertools
import ase
import math

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


class TestPBCSeeEachOther(unittest.TestCase):

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
        allshifts = torchani.aev.compute_shifts(cell, pbc, 1)

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
            molecule_index, atom_index1, atom_index2, _ = torchani.aev.neighbor_pairs(species == -1, coordinates, cell, allshifts, 1)
            self.assertEqual(molecule_index.tolist(), [0])
            self.assertEqual(atom_index1.tolist(), [0])
            self.assertEqual(atom_index2.tolist(), [1])

    def testPBCSurfaceSeeEachOther(self):
        cell = torch.eye(3, dtype=torch.double) * 10
        pbc = torch.ones(3, dtype=torch.uint8)
        allshifts = torchani.aev.compute_shifts(cell, pbc, 1)
        species = torch.tensor([[0, 0]])

        for i in range(3):
            xyz1 = torch.tensor([5.0, 5.0, 5.0], dtype=torch.double)
            xyz1[i] = 0.1
            xyz2 = xyz1.clone()
            xyz2[i] = 9.9

            coordinates = torch.stack([xyz1, xyz2]).unsqueeze(0)
            molecule_index, atom_index1, atom_index2, _ = torchani.aev.neighbor_pairs(species == -1, coordinates, cell, allshifts, 1)
            self.assertEqual(molecule_index.tolist(), [0])
            self.assertEqual(atom_index1.tolist(), [0])
            self.assertEqual(atom_index2.tolist(), [1])

    def testPBCEdgesSeeEachOther(self):
        cell = torch.eye(3, dtype=torch.double) * 10
        pbc = torch.ones(3, dtype=torch.uint8)
        allshifts = torchani.aev.compute_shifts(cell, pbc, 1)
        species = torch.tensor([[0, 0]])

        for i, j in itertools.combinations(range(3), 2):
            xyz1 = torch.tensor([5.0, 5.0, 5.0], dtype=torch.double)
            xyz1[i] = 0.1
            xyz1[j] = 0.1
            for new_i, new_j in [[0.1, 9.9], [9.9, 0.1], [9.9, 9.9]]:
                xyz2 = xyz1.clone()
                xyz2[i] = new_i
                xyz2[j] = new_i

            coordinates = torch.stack([xyz1, xyz2]).unsqueeze(0)
            molecule_index, atom_index1, atom_index2, _ = torchani.aev.neighbor_pairs(species == -1, coordinates, cell, allshifts, 1)
            self.assertEqual(molecule_index.tolist(), [0])
            self.assertEqual(atom_index1.tolist(), [0])
            self.assertEqual(atom_index2.tolist(), [1])

    def testNonRectangularPBCConnersSeeEachOther(self):
        species = torch.tensor([[0, 0]])
        cell = ase.geometry.cellpar_to_cell([10, 10, 10 * math.sqrt(2), 90, 45, 90])
        cell = torch.tensor(ase.geometry.complete_cell(cell), dtype=torch.double)
        pbc = torch.ones(3, dtype=torch.uint8)
        allshifts = torchani.aev.compute_shifts(cell, pbc, 1)

        xyz1 = torch.tensor([0.1, 0.1, 0.05], dtype=torch.double)
        xyz2 = torch.tensor([10.0, 0.1, 0.1], dtype=torch.double)

        coordinates = torch.stack([xyz1, xyz2]).unsqueeze(0)
        molecule_index, atom_index1, atom_index2, _ = torchani.aev.neighbor_pairs(species == -1, coordinates, cell, allshifts, 1)
        self.assertEqual(molecule_index.tolist(), [0])
        self.assertEqual(atom_index1.tolist(), [0])
        self.assertEqual(atom_index2.tolist(), [1])


class TestAEVOnBoundary(unittest.TestCase):

    def setUp(self):
        self.eps = 1e-9
        cell = ase.geometry.cellpar_to_cell([100, 100, 100 * math.sqrt(2), 90, 45, 90])
        self.cell = torch.tensor(ase.geometry.complete_cell(cell), dtype=torch.double)
        self.inv_cell = torch.inverse(self.cell)
        self.coordinates = torch.tensor([[[0.0, 0.0, 0.0],
                                          [1.0, -0.1, -0.1],
                                          [-0.1, 1.0, -0.1],
                                          [-0.1, -0.1, 1.0],
                                          [-1.0, -1.0, -1.0]]], dtype=torch.double)
        self.species = torch.tensor([[1, 0, 0, 0]])
        self.pbc = torch.ones(3, dtype=torch.uint8)
        self.v1, self.v2, self.v3 = self.cell
        self.center_coordinates = self.coordinates + 0.5 * (self.v1 + self.v2 + self.v3)
        builtin = torchani.neurochem.Builtins()
        self.aev_computer = builtin.aev_computer.to(torch.double)
        _, self.aev = self.aev_computer((self.species, self.center_coordinates, self.cell, self.pbc))

    def assertInCell(self, coordinates):
        coordinates_cell = coordinates @ self.inv_cell
        self.assertTrue(torch.allclose(coordinates, coordinates_cell @ self.cell))
        in_cell = (coordinates_cell >= -self.eps) & (coordinates_cell <= 1 + self.eps)
        self.assertTrue(in_cell.all())

    def assertNotInCell(self, coordinates):
        coordinates_cell = coordinates @ self.inv_cell
        self.assertTrue(torch.allclose(coordinates, coordinates_cell @ self.cell))
        in_cell = (coordinates_cell >= -self.eps) & (coordinates_cell <= 1 + self.eps)
        self.assertFalse(in_cell.all())

    def testCornerSurfaceAndEdge(self):
        for i, j, k in itertools.product([0, 0.5, 1], repeat=3):
            if i == 0.5 and j == 0.5 and k == 0.5:
                continue
            coordinates = self.coordinates + i * self.v1 + j * self.v2 + k * self.v3
            self.assertNotInCell(coordinates)
            coordinates = torchani.utils.map2central(self.cell, coordinates, self.pbc)
            self.assertInCell(coordinates)
            _, aev = self.aev_computer((self.species, coordinates, self.cell, self.pbc))
            self.assertGreater(aev.abs().max().item(), 0)
            self.assertTrue(torch.allclose(aev, self.aev))


if __name__ == '__main__':
    unittest.main()
