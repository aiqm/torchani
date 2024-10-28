from pathlib import Path
import unittest

import torch
import numpy as np

from torchani.models import ANI1x
from torchani._testing import ANITestCase, expand
from torchani.grad import vibrational_analysis, energies_forces_and_hessians
from torchani.utils import get_atomic_masses


@expand(jit=False, device="cpu")
class TestVibrational(ANITestCase):
    def testWater(self):
        model = self._setup(ANI1x().double())
        # Expected results
        data_path = (Path(__file__).parent / "resources") / "water-vib-expect.npz"
        with np.load(data_path) as data:
            coordinates = torch.tensor(
                np.expand_dims(data["coordinates"], 0),
                dtype=torch.float,
                device=self.device,
            )
            species = torch.tensor(
                np.expand_dims(data["species"], 0), dtype=torch.long, device=self.device
            )
            modes_expect = torch.tensor(
                data["modes"], dtype=torch.float, device=self.device
            )
            freqs_expect = torch.tensor(
                data["freqs"], dtype=torch.float, device=self.device
            )
        masses = get_atomic_masses(species, dtype=torch.double)
        energies, forces, hessians = energies_forces_and_hessians(
            model, species, coordinates
        )
        freq2, modes2, _, _ = vibrational_analysis(masses, hessians)
        freq2 = freq2[6:].float()
        modes2 = modes2[6:]
        self.assertEqual(freqs_expect, freq2, atol=0, rtol=0.02, exact_dtype=False)

        diff1 = (modes_expect - modes2).abs().max(dim=-1).values.max(dim=-1).values
        diff2 = (modes_expect + modes2).abs().max(dim=-1).values.max(dim=-1).values
        diff = torch.where(diff1 < diff2, diff1, diff2)
        self.assertLess(float(diff.max().item()), 0.02)


if __name__ == "__main__":
    unittest.main(verbosity=2)
