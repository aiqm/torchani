import torch
import torchani
from torchani.testing import TestCase


class _TestAEVBase(TestCase):

    def setUp(self):
        self.aev_computer = torchani.AEVComputer.like_1x()
        self.radial_length = self.aev_computer.radial_length
        self.debug = False

    def assertAEVEqual(self, expected_radial, expected_angular, aev):
        radial = aev[..., :self.radial_length]
        angular = aev[..., self.radial_length:]
        if self.debug:
            aid = 1
            print(torch.stack([expected_radial[0, aid, :], radial[0, aid, :]]))
        self.assertEqual(expected_radial, radial)
        self.assertEqual(expected_angular, angular)
