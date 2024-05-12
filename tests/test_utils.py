import unittest

import torch

from torchani.testing import ANITest, expand
from torchani.utils import (
    # Padding
    pad_atomic_properties,
    strip_redundant_padding,
    present_species,
    broadcast_first_dim,
    # Converters
    ChemicalSymbolsToInts,
    ChemicalSymbolsToAtomicNumbers,
    # Hessian
    hessian,
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


@expand(device="cpu", jit=True)
class TestHessian(ANITest):
    def testHessian(self):
        torch.jit.script(hessian)


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


# TODO: For now this test is cpu and non-jit only, it may need to be adapted
# TODO: Many of these utilities are not useful anymore
@expand(device="cpu", jit=False)
class TestPaddingUtils(ANITest):
    def testVectorSpecies(self):
        species1 = torch.tensor([[0, 2, 3, 1]])
        coordinates1 = torch.zeros(5, 4, 3)
        species2 = torch.tensor([[3, 2, 0, 1, 0]])
        coordinates2 = torch.zeros(2, 5, 3)
        atomic_properties = pad_atomic_properties(
            [
                broadcast_first_dim({"species": species1, "coordinates": coordinates1}),
                broadcast_first_dim({"species": species2, "coordinates": coordinates2}),
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
            ]
        )
        self.assertEqual(atomic_properties["species"], expected_species)
        self.assertEqual(atomic_properties["coordinates"].abs().max().item(), 0)

    def testTensorShape1NSpecies(self):
        species1 = torch.tensor([[0, 2, 3, 1]])
        coordinates1 = torch.zeros(5, 4, 3)
        species2 = torch.tensor([[3, 2, 0, 1, 0]])
        coordinates2 = torch.zeros(2, 5, 3)
        atomic_properties = pad_atomic_properties(
            [
                broadcast_first_dim({"species": species1, "coordinates": coordinates1}),
                broadcast_first_dim({"species": species2, "coordinates": coordinates2}),
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
            ]
        )
        self.assertEqual(atomic_properties["species"], expected_species)
        self.assertEqual(atomic_properties["coordinates"].abs().max().item(), 0)

    def testTensorSpecies(self):
        species1 = torch.tensor(
            [
                [0, 2, 3, 1],
                [0, 2, 3, 1],
                [0, 2, 3, 1],
                [0, 2, 3, 1],
                [0, 2, 3, 1],
            ]
        )
        coordinates1 = torch.zeros(5, 4, 3)
        species2 = torch.tensor([[3, 2, 0, 1, 0]])
        coordinates2 = torch.zeros(2, 5, 3)
        atomic_properties = pad_atomic_properties(
            [
                broadcast_first_dim({"species": species1, "coordinates": coordinates1}),
                broadcast_first_dim({"species": species2, "coordinates": coordinates2}),
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
            ]
        )
        self.assertEqual(atomic_properties["species"], expected_species)
        self.assertEqual(atomic_properties["coordinates"].abs().max().item(), 0)

    def testPresentSpecies(self):
        species = torch.tensor([0, 1, 1, 0, 3, 7, -1, -1])
        _present_species = present_species(species)
        expected = torch.tensor([0, 1, 3, 7])
        self.assertEqual(expected, _present_species)

    def testStripPaddingAndRestore(self):
        species1 = torch.randint(4, (5, 4), dtype=torch.long)
        coordinates1 = torch.randn(5, 4, 3)
        species2 = torch.randint(4, (2, 5), dtype=torch.long)
        coordinates2 = torch.randn(2, 5, 3)
        atomic_properties12 = pad_atomic_properties(
            [
                broadcast_first_dim({"species": species1, "coordinates": coordinates1}),
                broadcast_first_dim({"species": species2, "coordinates": coordinates2}),
            ]
        )
        species12 = atomic_properties12["species"]
        coordinates12 = atomic_properties12["coordinates"]
        species3 = torch.randint(4, (2, 10), dtype=torch.long)
        coordinates3 = torch.randn(2, 10, 3)
        atomic_properties123 = pad_atomic_properties(
            [
                broadcast_first_dim({"species": species1, "coordinates": coordinates1}),
                broadcast_first_dim({"species": species2, "coordinates": coordinates2}),
                broadcast_first_dim({"species": species3, "coordinates": coordinates3}),
            ]
        )
        species123 = atomic_properties123["species"]
        coordinates123 = atomic_properties123["coordinates"]
        species_coordinates1_ = strip_redundant_padding(
            broadcast_first_dim(
                {"species": species123[:5, ...], "coordinates": coordinates123[:5, ...]}
            )
        )
        species1_ = species_coordinates1_["species"]
        coordinates1_ = species_coordinates1_["coordinates"]
        self.assertEqual(species1_, species1)
        self.assertEqual(coordinates1_, coordinates1)
        species_coordinates12_ = strip_redundant_padding(
            broadcast_first_dim(
                {"species": species123[:7, ...], "coordinates": coordinates123[:7, ...]}
            )
        )
        species12_ = species_coordinates12_["species"]
        coordinates12_ = species_coordinates12_["coordinates"]
        self.assertEqual(species12_, species12)
        self.assertEqual(coordinates12_, coordinates12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
