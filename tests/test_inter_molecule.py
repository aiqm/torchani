import unittest

import torch

from torchani._testing import ANITestCase, expand
from torchani.models import ANI1x


@expand()
class TestMoleculeIdxs(ANITestCase):
    def setUp(self):
        self.model = self._setup(ANI1x(model_index=0))

    def testInterMoleculeIdxs(self) -> None:
        znums = torch.tensor([[6, 1, 1, 1, 1]], device=self.device)
        coords = torch.tensor(
            [[
                [0.0009055189, 0.0040717406, -0.0202114333],
                [-0.6576767188, -0.8481438538, -0.3213789740],
                [-0.4585006357, 0.9751998880, -0.3060610353],
                [0.0853135728, -0.0253370864, 1.0803551453],
                [1.0299582628, -0.1057906884, -0.4327037027],
            ]], device=self.device,
        )
        coords2 = coords.clone().repeat(2, 1, 1)
        coords2[1, :, :] += coords.new_full((5, 3), fill_value=1.0, device=self.device)
        coords2 = coords2.view(1, -1, 3)
        znums2 = znums.clone().repeat(2, 1).view(1, -1)
        double = self.model((znums, coords)).energies * 2

        molecule_idxs = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], device=self.device)
        out = self.model((znums2, coords2), _molecule_idxs=molecule_idxs).energies
        # Excluding inter-molecule interactions
        self.assertEqual(out, double)

        # Including inter-molecule interactions
        out = self.model((znums2, coords2)).energies
        self.assertNotEqual(out, double)


if __name__ == "__main__":
    unittest.main(verbosity=2)
