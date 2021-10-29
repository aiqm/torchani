import unittest
import torch
import torchani
from torchani.testing import TestCase


class TestCalc(TestCase):

    def testComputeDiploes(self):
        species = torch.tensor([[1, 1, 8]], dtype=torch.long)
        coordinates = torch.tensor([[[-1, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=torch.float)
        charges = torch.tensor([[0.1, 0.1, -0.2]])

        dipoles = torchani.calc.compute_dipole(species, coordinates, charges, center_of_mass=True)
        self.assertEqual(dipoles, [[0.0000, -0.2000, 0.0000]])

        dipoles = torchani.calc.compute_dipole(species, coordinates, charges, center_of_mass=False)
        self.assertEqual(dipoles, [[0.0000, -0.2000, 0.0000]])

        charges = torch.tensor([[0.1, 0.1, -0.4]])
        dipoles = torchani.calc.compute_dipole(species, coordinates, charges, center_of_mass=True)
        self.assertEqual(dipoles, [[0.0000, -0.2223813533782959, 0.0000]])

        dipoles = torchani.calc.compute_dipole(species, coordinates, charges, center_of_mass=False)
        self.assertEqual(dipoles, [[0.0000, -0.4000, 0.0000]])


if __name__ == '__main__':
    unittest.main()
