import torch
import torchani


class _TestAEVBase(torchani.testing.TestCase):

    def setUp(self):
        ani1x = torchani.models.ANI1x()
        self.aev_computer = ani1x.aev_computer
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
