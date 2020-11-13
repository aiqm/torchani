import unittest
import torch
import torchani
import os

tolerance = 1e-5


class _TestAEVBase(unittest.TestCase):

    def setUp(self):
        path = os.path.dirname(os.path.realpath(__file__))
        const_file = os.path.join(path, '../torchani/resources/ani-1x_8x/rHCNO-5.2R_16-3.5A_a4-8.params')  # noqa: E501
        consts = torchani.neurochem.Constants(const_file)
        self.aev_computer = torchani.AEVComputer(**consts)
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
