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


class TestBuiltinNetPeriodicTableIndex(unittest.TestCase):

    def testCH4(self):
        model1 = torchani.models.ANI1x()
        model2 = torchani.models.ANI1x(periodic_table_index=True)
        coordinates = torch.tensor([[[0.03192167, 0.00638559, 0.01301679],
                             [-0.83140486, 0.39370209, -0.26395324],
                             [-0.66518241, -0.84461308, 0.20759389],
                             [0.45554739, 0.54289633, 0.81170881],
                             [0.66091919, -0.16799635, -0.91037834]]],
                           requires_grad=True)
        species1 = model1.species_to_tensor('CHHHH').unsqueeze(0)
        species2 = torch.tensor([[6, 1, 1, 1, 1]])

        ###############################################################################
        # Now let's compute energy and force:
        energy1 = model1((species1, coordinates)).energies
        energy2 = model2((species2, coordinates)).energies
        derivative1 = torch.autograd.grad(energy1.sum(), coordinates)[0]
        derivative2 = torch.autograd.grad(energy2.sum(), coordinates)[0]
        self.assertTrue(torch.allclose(energy1, energy2))
        self.assertTrue(torch.allclose(derivative1, derivative2))


if __name__ == '__main__':
    unittest.main()
