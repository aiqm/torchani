import unittest
import torch
import torchani
from torchani.testing import TestCase


class TestUtils(TestCase):

    def testChemicalSymbolsToInts(self):
        str2i = torchani.utils.ChemicalSymbolsToInts(['A', 'B', 'C', 'D', 'E', 'F'])
        self.assertEqual(len(str2i), 6)
        self.assertListEqual(str2i('BACCC').tolist(), [1, 0, 2, 2, 2])

    def testChemicalSymbolsToAtomicNumbers(self):
        symbols_to_atomic_nums = torchani.utils.ChemicalSymbolsToAtomicNumbers()
        atomic_nums = symbols_to_atomic_nums(['H', 'H', 'C', 'Cl', 'N', 'H'])
        self.assertEqual(atomic_nums, torch.tensor([1, 1, 6, 17, 7, 1], dtype=torch.long))

    def testChemicalSymbolsToAtomicNumbersJIT(self):
        symbols_to_atomic_nums = torchani.utils.ChemicalSymbolsToAtomicNumbers()
        symbols_to_atomic_nums = torch.jit.script(symbols_to_atomic_nums)
        atomic_nums = symbols_to_atomic_nums(['H', 'H', 'C', 'Cl', 'N', 'H'])
        self.assertEqual(atomic_nums, torch.tensor([1, 1, 6, 17, 7, 1], dtype=torch.long))

    def testChemicalSymbolsToIntsJIT(self):
        str2i = torchani.utils.ChemicalSymbolsToInts(['A', 'B', 'C', 'D', 'E', 'F'])
        str2i = torch.jit.script(str2i)
        self.assertListEqual(str2i('BACCC').tolist(), [1, 0, 2, 2, 2])

    def testHessianJIT(self):
        torch.jit.script(torchani.utils.hessian)

    def testGSAES(self):
        gsaes = torchani.utils.sorted_gsaes(('H', 'C', 'S'), 'wB97X', '631Gd')
        self.assertEqual(gsaes, [-0.4993213, -37.8338334, -398.0814169])

        gsaes = torchani.utils.sorted_gsaes(('H', 'S', 'C'), 'wB97X', '631Gd')
        self.assertEqual(gsaes, [-0.4993213, -398.0814169, -37.8338334])

        # test case insensitivity
        gsaes = torchani.utils.sorted_gsaes(('H', 'S', 'C'), 'Wb97x', '631GD')
        self.assertEqual(gsaes, [-0.4993213, -398.0814169, -37.8338334])

        with self.assertRaises(KeyError):
            torchani.utils.sorted_gsaes('wB97X', '631Gd', ('Pu'))


if __name__ == '__main__':
    unittest.main()
