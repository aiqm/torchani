import unittest
import torch
from torchani.testing import ANITest, expand
from torchani.geometry import Displacer
from torchani.units import bohr2angstrom


@expand()
class TestDisplaceToCom(ANITest):
    def setUp(self) -> None:
        self.fn = self._setup(Displacer())

    def testSimple(self):
        species = torch.tensor([[1, 1, 1, 1, 6]], dtype=torch.long, device=self.device)
        coordinates = torch.tensor(
            [[[0.0, 0.0, 0.0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [0.5, 0.5, 0.5]]],
            dtype=torch.float,
            device=self.device,
        )
        displaced_coordinates = self.fn(species, coordinates)
        self.assertEqual(
            displaced_coordinates,
            coordinates
            - torch.tensor(
                [[0.5, 0.5, 0.5]], device=self.device, dtype=torch.float
            ).unsqueeze(1),
        )

    def testDummyAtoms(self):
        species = torch.tensor(
            [[1, 1, 1, 1, 6, -1]], dtype=torch.long, device=self.device
        )
        coordinates = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [0, 1, 1],
                    [1, 0, 1],
                    [1, 1, 0],
                    [0.5, 0.5, 0.5],
                    [0, 0, 0],
                ]
            ],
            dtype=torch.float,
            device=self.device,
        )
        displaced_coordinates = self.fn(species, coordinates)
        expect_coordinates = coordinates - torch.tensor(
            [[0.5, 0.5, 0.5]], device=self.device, dtype=torch.float
        ).unsqueeze(1)
        expect_coordinates[(species == -1)] = 0
        self.assertEqual(displaced_coordinates, expect_coordinates)

    def testBatched(self):
        species = torch.tensor(
            [[1, 1, 1, 1, 6, -1], [6, 6, 6, 6, 8, -1]],
            dtype=torch.long,
            device=self.device,
        )
        coordinates = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [0, 1, 1],
                    [1, 0, 1],
                    [1, 1, 0],
                    [0.5, 0.5, 0.5],
                    [0, 0, 0],
                ]
            ],
            dtype=torch.float,
            device=self.device,
        )
        coordinates = torch.cat((coordinates, coordinates.clone()), dim=0)
        displaced_coordinates = self.fn(species, coordinates)
        expect_coordinates = coordinates - torch.tensor(
            [[0.5, 0.5, 0.5]], device=self.device, dtype=torch.float
        ).unsqueeze(1)
        expect_coordinates[(species == -1)] = 0
        self.assertEqual(displaced_coordinates, expect_coordinates)

    def testMatchOrcaResults(self):
        # this was taken from ORCA 4.2
        species = torch.tensor([[1, 1, 8]], dtype=torch.long, device=self.device)
        coordinates = torch.tensor(
            [
                [
                    [0.600000, 0.000000, 0.000000],
                    [-0.239517, 0.927144, 0.000000],
                    [0, 0, 0],
                ]
            ],
            dtype=torch.float,
            device=self.device,
        )
        displaced_coordinates = self.fn(species, coordinates)
        # com = coordinates + displaced_coordinates
        expect_com = torch.tensor(
            [[0.038116, 0.098033, 0.000000]], device=self.device, dtype=torch.float
        )
        expect_com = bohr2angstrom(expect_com)
        self.assertEqual(displaced_coordinates, coordinates - expect_com)


if __name__ == "__main__":
    unittest.main(verbosity=2)
