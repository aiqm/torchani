import unittest

import torch

from torchani.electro import DipoleComputer
from torchani._testing import ANITestCase, expand


@expand()
class TestDipoles(ANITestCase):
    def setUp(self) -> None:
        self.species = torch.tensor([[1, 1, 8]], dtype=torch.long, device=self.device)
        self.coordinates = torch.tensor(
            [[[-1, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=torch.float, device=self.device
        )
        self.compute_dipole = self._setup(DipoleComputer())
        self.compute_dipole_from_origin = self._setup(
            DipoleComputer(reference="origin")
        )

    def testZeroNetCharge(self):
        species = self.species
        coordinates = self.coordinates
        charges = torch.tensor([[0.1, 0.1, -0.2]], device=self.device)
        dipoles = self.compute_dipole(
            species,
            coordinates,
            charges,
        )
        self.assertEqual(dipoles, [[0.0000, -0.2000, 0.0000]])

        dipoles = self.compute_dipole_from_origin(
            species,
            coordinates,
            charges,
        )
        self.assertEqual(dipoles, [[0.0000, -0.2000, 0.0000]])

    def testNonZeroNetCharge(self):
        species = self.species
        coordinates = self.coordinates
        charges = torch.tensor([[0.1, 0.1, -0.4]], device=self.device)
        dipoles = self.compute_dipole(
            species,
            coordinates,
            charges,
        )
        self.assertEqual(dipoles, [[0.0000, -0.2223813533782959, 0.0000]])

        dipoles = self.compute_dipole_from_origin(
            species,
            coordinates,
            charges,
        )
        self.assertEqual(dipoles, [[0.0000, -0.4000, 0.0000]])


if __name__ == "__main__":
    unittest.main(verbosity=2)
