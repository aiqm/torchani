from pathlib import Path
import unittest

import numpy as np

import torch
import torchani
from torchani.testing import TestCase


class TestVibrational(TestCase):
    def testWater(self):
        model = torchani.models.ANI1x().double()
        # Expected results
        data_path = (Path(__file__).parent / "test_data") / "water-vib-expect.npz"
        with np.load(data_path) as data:
            coordinates = torch.tensor(
                np.expand_dims(data["coordinates"], 0),
                dtype=torch.float,
                requires_grad=True,
            )
            species = torch.tensor(np.expand_dims(data["species"], 0), dtype=torch.long)
            modes_expect = torch.tensor(data["modes"], dtype=torch.float)
            freqs_expect = torch.tensor(data["freqs"], dtype=torch.float)
        masses = torchani.utils.get_atomic_masses(species, dtype=torch.double)
        _, energies = model((species, coordinates))
        hessian = torchani.utils.hessian(coordinates, energies=energies)
        freq2, modes2, _, _ = torchani.utils.vibrational_analysis(masses, hessian)
        freq2 = freq2[6:].float()
        modes2 = modes2[6:]
        self.assertEqual(freqs_expect, freq2, atol=0, rtol=0.02, exact_dtype=False)

        diff1 = (modes_expect - modes2).abs().max(dim=-1).values.max(dim=-1).values
        diff2 = (modes_expect + modes2).abs().max(dim=-1).values.max(dim=-1).values
        diff = torch.where(diff1 < diff2, diff1, diff2)
        self.assertLess(float(diff.max().item()), 0.02)


if __name__ == "__main__":
    unittest.main(verbosity=2)
