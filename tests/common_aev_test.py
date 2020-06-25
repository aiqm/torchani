import unittest
import torch
import torchani

tolerance = 1e-5


class _TestAEVBase(unittest.TestCase):

    def setUp(self):
        ani1x = torchani.models.ANI1x()
        self.aev_computer = ani1x.aev_computer
        self.radial_length = self.aev_computer.radial_length
        self.debug = False

    def assertAEVEqual(self, expected_radial, expected_angular, aev, tolerance=tolerance):
        radial = aev[..., :self.radial_length]
        angular = aev[..., self.radial_length:]
        radial_diff = expected_radial - radial
        if self.debug:
            aid = 1
            print(torch.stack([expected_radial[0, aid, :], radial[0, aid, :], radial_diff.abs()[0, aid, :]], dim=1))
        radial_max_error = torch.max(torch.abs(radial_diff)).item()
        angular_diff = expected_angular - angular
        angular_max_error = torch.max(torch.abs(angular_diff)).item()
        self.assertLess(radial_max_error, tolerance)
        self.assertLess(angular_max_error, tolerance)
