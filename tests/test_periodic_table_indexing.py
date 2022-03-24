import unittest
import torch
import torchani
from torchani.testing import TestCase


class TestSpeciesConverter(TestCase):

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
        self.assertEqual(output, expect)


class TestSpeciesConverterJIT(TestSpeciesConverter):

    def setUp(self):
        super().setUp()
        self.c = torch.jit.script(self.c)


class TestBuiltinEnsemblePeriodicTableIndex(TestCase):

    def setUp(self):
        self.model1 = torchani.models.ANI1x()
        self.model2 = torchani.models.ANI1x(periodic_table_index=True)
        self.coordinates = torch.tensor([[[0.03192167, 0.00638559, 0.01301679],
                                          [-0.83140486, 0.39370209, -0.26395324],
                                          [-0.66518241, -0.84461308, 0.20759389],
                                          [0.45554739, 0.54289633, 0.81170881],
                                          [0.66091919, -0.16799635, -0.91037834]]],
                                        requires_grad=True)
        self.species1 = self.model1.species_to_tensor(['C', 'H', 'H', 'H', 'H']).unsqueeze(0)
        self.species2 = torch.tensor([[6, 1, 1, 1, 1]])

    def testCH4Ensemble(self):
        energy1 = self.model1((self.species1, self.coordinates)).energies
        energy2 = self.model2((self.species2, self.coordinates)).energies
        derivative1 = torch.autograd.grad(energy1.sum(), self.coordinates)[0]
        derivative2 = torch.autograd.grad(energy2.sum(), self.coordinates)[0]
        self.assertEqual(energy1, energy2)
        self.assertEqual(derivative1, derivative2)

    def testCH4Single(self):
        energy1 = self.model1[0]((self.species1, self.coordinates)).energies
        energy2 = self.model2[0]((self.species2, self.coordinates)).energies
        derivative1 = torch.autograd.grad(energy1.sum(), self.coordinates)[0]
        derivative2 = torch.autograd.grad(energy2.sum(), self.coordinates)[0]
        self.assertEqual(energy1, energy2)
        self.assertEqual(derivative1, derivative2)


if __name__ == '__main__':
    unittest.main()
