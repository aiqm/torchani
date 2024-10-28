import unittest

import torch

from torchani._testing import ANITestCase, expand
from torchani.utils import (
    # Padding
    pad_atomic_properties,
    strip_redundant_padding,
    # Converters
    AtomicNumbersToChemicalSymbols,
    IntsToChemicalSymbols,
    ChemicalSymbolsToInts,
    ChemicalSymbolsToAtomicNumbers,
    # Torch utils
    nonzero_in_chunks,
)


@expand()
class TestTorchUtils(ANITestCase):
    def testNonzeroInChunks(self):
        tensor = torch.tensor([1, 0, 0, 1, 5], device=self.device)
        f = nonzero_in_chunks
        if self.jit:
            f = torch.jit.script(nonzero_in_chunks)
        result = f(tensor, chunk_size=2)
        expected = torch.tensor([0, 3, 4], device=self.device)
        self.assertEqual(result, expected)


@expand()
class TestConverters(ANITestCase):
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

    def testAtomicNumbersToSymbols(self):
        atomic_nums_to_symbols = self._setup(AtomicNumbersToChemicalSymbols())
        symbols = atomic_nums_to_symbols(torch.tensor([6, 1, 1, 1, -1, -1, -1]))
        self.assertEqual(symbols, ["C", "H", "H", "H"])

    def testIdxsToSymbols(self):
        i2str = self._setup(IntsToChemicalSymbols(["A", "B", "C", "D", "E", "F"]))
        symbols = i2str(torch.tensor([5, 1, 1, 1, -1, -1, -1]))
        self.assertEqual(symbols, ["F", "B", "B", "B"])


# TODO: For now this is non-jit only, it may need to be adapted
@expand(jit=False)
class TestPaddingUtils(ANITestCase):
    def testStripRedundantPadding(self):
        expect = torch.tensor(
            [
                [0, 2, 3, 1, -1],
                [0, 2, 3, 1, 0],
                [0, 2, 3, 1, -1],
                [0, 2, 3, 1, -1],
                [0, 2, 3, 1, 3],
            ],
            device=self.device,
        )
        species = torch.tensor(
            [
                [0, 2, 3, 1, -1, -1],
                [0, 2, 3, 1, 0, -1],
                [0, 2, 3, 1, -1, -1],
                [0, 2, 3, 1, -1, -1],
                [0, 2, 3, 1, 3, -1],
            ],
            device=self.device,
        )
        coords = torch.zeros(5, 6, 3, device=self.device)
        props = {"species": species, "coordinates": coords}
        props = strip_redundant_padding(props)
        self.assertEqual(props["species"], expect)
        self.assertEqual(props["coordinates"], torch.zeros(5, 5, 3, device=self.device))

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
