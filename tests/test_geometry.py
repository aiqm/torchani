import unittest
import torch
import torchani
from torchani.testing import TestCase


class TestGeometry(TestCase):

    def testCenterToComFrame(self):
        species = torch.tensor([[1, 1, 1, 1, 6]], dtype=torch.long)
        coordinates = torch.tensor([[[0.0, 0.0, 0.0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [0.5, 0.5, 0.5]]], dtype=torch.float)
        species, displaced_coordinates = torchani.geometry.displace_to_com_frame((species, coordinates))
        self.assertEqual(displaced_coordinates, coordinates - torch.tensor([[0.5, 0.5, 0.5]]).unsqueeze(1))

    def testCenterToComFrameDummy(self):
        species = torch.tensor([[1, 1, 1, 1, 6, -1]], dtype=torch.long)
        coordinates = torch.tensor([[[0.0, 0.0, 0.0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [0.5, 0.5, 0.5], [0, 0, 0]]], dtype=torch.float)
        species, displaced_coordinates = torchani.geometry.displace_to_com_frame((species, coordinates))
        expect_coordinates = coordinates - torch.tensor([[0.5, 0.5, 0.5]]).unsqueeze(1)
        expect_coordinates[(species == -1)] = 0
        self.assertEqual(displaced_coordinates, expect_coordinates)

    def testCenterToComFrameMany(self):
        species = torch.tensor([[1, 1, 1, 1, 6, -1], [6, 6, 6, 6, 8, -1]], dtype=torch.long)
        coordinates = torch.tensor([[[0.0, 0.0, 0.0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [0.5, 0.5, 0.5], [0, 0, 0]]], dtype=torch.float)
        coordinates = torch.cat((coordinates, coordinates.clone()), dim=0)
        species, displaced_coordinates = torchani.geometry.displace_to_com_frame((species, coordinates))
        expect_coordinates = coordinates - torch.tensor([[0.5, 0.5, 0.5]]).unsqueeze(1)
        expect_coordinates[(species == -1)] = 0
        self.assertEqual(displaced_coordinates, expect_coordinates)

    def testCenterOfMassWaterOrca(self):
        # this was taken from ORCA 4.2
        species = torch.tensor([[1, 1, 8]], dtype=torch.long)
        coordinates = torch.tensor([[[0.600000, 0.000000, 0.000000],
                                    [-0.239517, 0.927144, 0.000000],
                                    [0, 0, 0]]], dtype=torch.float)
        species, displaced_coordinates = torchani.geometry.displace_to_com_frame((species, coordinates))
        # com = coordinates + displaced_coordinates
        expect_com = torch.tensor([[0.038116, 0.098033, 0.000000]])
        expect_com = torchani.units.bohr2angstrom(expect_com)
        self.assertEqual(displaced_coordinates, coordinates - expect_com)


if __name__ == '__main__':
    unittest.main()
