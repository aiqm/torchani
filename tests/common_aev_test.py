import torch
import torchani
import os
from torchani.testing import TestCase


class _TestAEVBase(TestCase):

    def setUp(self):
        path = os.path.dirname(os.path.realpath(__file__))
        const_file = os.path.join(path, '../torchani/resources/ani-1x_8x/rHCNO-5.2R_16-3.5A_a4-8.params')  # noqa: E501
        consts = torchani.neurochem.Constants(const_file)
        self.aev_computer = torchani.AEVComputer(**consts)
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
