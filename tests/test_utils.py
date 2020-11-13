import unittest
import torchani


class TestUtils(torchani.testing.TestCase):

    def testChemicalSymbolsToInts(self):
        str2i = torchani.utils.ChemicalSymbolsToInts(['A', 'B', 'C', 'D', 'E', 'F'])
        self.assertEqual(len(str2i), 6)
        self.assertListEqual(str2i('BACCC').tolist(), [1, 0, 2, 2, 2])


if __name__ == '__main__':
    unittest.main()
