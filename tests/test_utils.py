import unittest

import torch

from torchani.testing import ANITest, expand
from torchani.utils import (
    # Padding
    pad_atomic_properties,
    # Converters
    ChemicalSymbolsToInts,
    ChemicalSymbolsToAtomicNumbers,
    # GSAES
    sorted_gsaes,
)


@expand()
class TestConverters(ANITest):
    def testSymbolsToIdxs(self):
        str2i = self._setup(ChemicalSymbolsToInts(["A", "B", "C", "D", "E", "F"]))

        # __len__ is not implemented in JIT
        if not self.jit:
            self.assertEqual(len(str2i), 6)
        self.assertListEqual(str2i("BACCC").tolist(), [1, 0, 2, 2, 2])

    def testSymbolsToAtomicNumbers(self):
        symbols_to_atomic_nums = self._setup(ChemicalSymbolsToAtomicNumbers())
        atomic_nums = symbols_to_atomic_nums(["H", "H", "C", "Cl", "N", "H"])
        self.assertEqual(
            atomic_nums, torch.tensor([1, 1, 6, 17, 7, 1], dtype=torch.long)
        )


@expand(device="cpu", jit=False)
class TestGSAES(ANITest):
    def testGSAES(self):
        gsaes = sorted_gsaes(("H", "C", "S"), "wB97X", "631Gd")
        self.assertEqual(gsaes, [-0.4993213, -37.8338334, -398.0814169])

        gsaes = sorted_gsaes(("H", "S", "C"), "wB97X", "631Gd")
        self.assertEqual(gsaes, [-0.4993213, -398.0814169, -37.8338334])

        # test case insensitivity
        gsaes = sorted_gsaes(("H", "S", "C"), "Wb97x", "631GD")
        self.assertEqual(gsaes, [-0.4993213, -398.0814169, -37.8338334])

        with self.assertRaises(KeyError):
            sorted_gsaes("wB97X", "631Gd", ("Pu"))


# TODO: For now this is non-jit only, it may need to be adapted
@expand(jit=False)
class TestPaddingUtils(ANITest):
    def testPadAtomicProperties(self):
        coordinates1 = torch.zeros(5, 4, 3, device=self.device)
        species1 = torch.tensor([[0, 2, 3, 1]], device=self.device).repeat(
            coordinates1.shape[0], 1
        )
        coordinates2 = torch.zeros(2, 5, 3, device=self.device)
        species2 = torch.tensor([[3, 2, 0, 1, 0]], device=self.device).repeat(
            coordinates2.shape[0], 1
        )
        atomic_properties = pad_atomic_properties(
            [
                {"species": species1, "coordinates": coordinates1},
                {"species": species2, "coordinates": coordinates2},
            ]
        )
        self.assertEqual(atomic_properties["species"].shape[0], 7)
        self.assertEqual(atomic_properties["species"].shape[1], 5)
        expected_species = torch.tensor(
            [
                [0, 2, 3, 1, -1],
                [0, 2, 3, 1, -1],
                [0, 2, 3, 1, -1],
                [0, 2, 3, 1, -1],
                [0, 2, 3, 1, -1],
                [3, 2, 0, 1, 0],
                [3, 2, 0, 1, 0],
            ],
            device=self.device,
        )
        self.assertEqual(atomic_properties["species"], expected_species)
        self.assertEqual(atomic_properties["coordinates"].abs().max().item(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
