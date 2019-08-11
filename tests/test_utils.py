import unittest
import torchani


class TestUtils(unittest.TestCase):

    def testChemicalSymbolsToInts(self):
        str2i = torchani.utils.ChemicalSymbolsToInts('ABCDEF')
        self.assertEqual(len(str2i), 6)
        self.assertListEqual(str2i('BACCC').tolist(), [1, 0, 2, 2, 2])


if __name__ == '__main__':
    unittest.main()
