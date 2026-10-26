import torch
import unittest

from torchani._testing import ANITestCase, expand
from torchani.models import ANI1xnr
from torchani.grad import energies_and_forces


@expand()
class TestANI1xnr(ANITestCase):
    def setUp(self):
        model = ANI1xnr()
        self.ensemble = self._setup(model)
        self.model_list = [self._setup(model[j]) for j in range(len(model))]

    def testEnergyForce(self):
        species = torch.tensor(
            [[6, 8, 7, 1, 1, 1, 1, 1]], device=self.device, dtype=torch.long
        )
        coords = torch.tensor(
            [
                [
                    [0.8715798227, 0.1619958512, 0.0322465836],
                    [0.0824525507, -0.6745183984, -0.7728562897],
                    [-1.2366803437, -0.3917257278, -0.6407828312],
                    [1.3846786207, -0.5081762833, 0.7522303382],
                    [0.2593533005, 0.9062281629, 0.5764446979],
                    [1.6258014512, 0.7033308948, -0.5701975001],
                    [-1.6685247073, -0.8348361119, 0.1963280892],
                    [-1.3186606949, 0.6377016126, -0.6093561983],
                ]
            ],
            dtype=torch.float32,
            device=self.device,
        )

        out = energies_and_forces(self.ensemble, species, coords)
        expect_e = torch.tensor([-34.47121429])
        expect_f = torch.tensor(
            [
                [
                    [0.08440366, 0.06047969, 0.07886853],
                    [0.02933429, -0.07646689, -0.06630778],
                    [-0.11919171, 0.03896857, 0.02011552],
                    [-0.00627692, 0.02099152, -0.00762838],
                    [0.02202270, -0.00955067, -0.00464736],
                    [-0.01395069, -0.01102096, 0.00663728],
                    [0.01439927, 0.00772926, -0.02473310],
                    [-0.01074060, -0.03113052, -0.00230472],
                ]
            ]
        )
        self.assertEqual(out.energies, expect_e)
        self.assertEqual(out.forces, expect_f)


if __name__ == "__main__":
    unittest.main(verbosity=2)
