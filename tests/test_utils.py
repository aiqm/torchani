import unittest
import torch
import torchani
from torchani.testing import ANITest, expand


@expand()
class TestConverters(ANITest):
    def testSymbolsToIdxs(self):
        str2i = self._setup(
            torchani.utils.ChemicalSymbolsToInts(["A", "B", "C", "D", "E", "F"])
        )

        # __len__ is not implemented in JIT
        if not self.jit:
            self.assertEqual(len(str2i), 6)
        self.assertListEqual(str2i("BACCC").tolist(), [1, 0, 2, 2, 2])

    def testSymbolsToAtomicNumbers(self):
        symbols_to_atomic_nums = self._setup(
            torchani.utils.ChemicalSymbolsToAtomicNumbers()
        )
        atomic_nums = symbols_to_atomic_nums(["H", "H", "C", "Cl", "N", "H"])
        self.assertEqual(
            atomic_nums, torch.tensor([1, 1, 6, 17, 7, 1], dtype=torch.long)
        )


@expand(device="cpu", jit=True)
class TestHessian(ANITest):
    def testHessian(self):
        torch.jit.script(torchani.utils.hessian)


@expand(device="cpu", jit=False)
class TestGSAES(ANITest):
    def testGSAES(self):
        gsaes = torchani.utils.sorted_gsaes(("H", "C", "S"), "wB97X", "631Gd")
        self.assertEqual(gsaes, [-0.4993213, -37.8338334, -398.0814169])

        gsaes = torchani.utils.sorted_gsaes(("H", "S", "C"), "wB97X", "631Gd")
        self.assertEqual(gsaes, [-0.4993213, -398.0814169, -37.8338334])

        # test case insensitivity
        gsaes = torchani.utils.sorted_gsaes(("H", "S", "C"), "Wb97x", "631GD")
        self.assertEqual(gsaes, [-0.4993213, -398.0814169, -37.8338334])

        with self.assertRaises(KeyError):
            torchani.utils.sorted_gsaes("wB97X", "631Gd", ("Pu"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
