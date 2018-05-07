import torch
import numpy
import torchani
import unittest


class TestForceNeuroChem(unittest.TestCase):

    def setUp(self, dtype=torchani.default_dtype, device=torchani.default_device):
        self.tolerance = 1e-5
        self.ncaev = torchani.NeuroChemAEV(dtype=dtype, device=device)
        self.aev_computer1 = torchani.NeighborAEV(dtype=dtype, device=device)
        self.model1 = torchani.ModelOnAEV(
            self.aev_computer1, derivative=True, device=device, from_pync=None)
        self.aev_computer2 = torchani.AEV(dtype=dtype, device=device)
        self.model2 = torchani.ModelOnAEV(
            self.aev_computer2, derivative=True, device=device, from_pync=None)

    def _test_molecule(self, coordinates, species):
        _, force = self.model1(coordinates, species)
        _, force = self.model2(coordinates, species)

    def testCH4(self):
        coordinates = torch.tensor([[[0.03192167,  0.00638559,  0.01301679],
                                     [-0.83140486,  0.39370209, -0.26395324],
                                     [-0.66518241, -0.84461308,  0.20759389],
                                     [0.45554739,   0.54289633,  0.81170881],
                                     [0.66091919,  -0.16799635, -0.91037834]],
                                    [[0,  0,  0],
                                     [0,  0,  1],
                                     [1,  0,  0],
                                     [0,  1,  0],
                                     [-1, -1, -1]],
                                    ], dtype=self.aev_computer1.dtype, device=self.aev_computer1.device)
        species = ['C', 'H', 'H', 'H', 'H']
        self._test_molecule(coordinates, species)


if __name__ == '__main__':
    unittest.main()
