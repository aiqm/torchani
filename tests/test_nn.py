import unittest
import torch
import torchani


class TestSpeciesConverter(unittest.TestCase):

    def setUp(self):
        self.c = torchani.SpeciesConverter(['H', 'C', 'N', 'O'])

    def testSpeciesConverter(self):
        input_ = torch.tensor([
            [1, 6, 7, 8, -1],
            [1, 1, -1, 8, 1],
        ], dtype=torch.long)
        expect = torch.tensor([
            [0, 1, 2, 3, -1],
            [0, 0, -1, 3, 0],
        ], dtype=torch.long)
        dummy_coordinates = torch.empty(2, 5, 3)
        output = self.c((input_, dummy_coordinates)).species
        self.assertTrue(torch.allclose(output, expect))


class TestSpeciesConverterJIT(TestSpeciesConverter):

    def setUp(self):
        super().setUp()
        self.c = torch.jit.script(self.c)


if __name__ == '__main__':
    unittest.main()
