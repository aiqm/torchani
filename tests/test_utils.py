import unittest
import torch
import torchani


class TestUtils(unittest.TestCase):

    def testChemicalSymbolsToInts(self):
        str2i = torchani.utils.ChemicalSymbolsToInts(['A', 'B', 'C', 'D', 'E', 'F'])
        self.assertEqual(len(str2i), 6)
        self.assertListEqual(str2i('BACCC').tolist(), [1, 0, 2, 2, 2])

    def testHessianJIT(self):
        torch.jit.script(torchani.utils.hessian)


if __name__ == '__main__':
    unittest.main()
