import unittest
import pickle
import os
import torch
import torchani
from torchani.testing import TestCase

path = os.path.dirname(os.path.realpath(__file__))
N = 10


class TestEnsemble(TestCase):

    def setUp(self):
        self.conformations = 20
        ani1x = torchani.models.ANI1x()
        self.aev_computer = ani1x.aev_computer
        self.model_iterator = ani1x.neural_networks
        self.ensemble = torchani.nn.Sequential(self.aev_computer, self.model_iterator)

    def _test_molecule(self, coordinates, species):
        model_list = [torchani.nn.Sequential(self.aev_computer, m) for m in self.model_iterator]
        coordinates.requires_grad_(True)
        _, energy1 = self.ensemble((species, coordinates))
        force1 = torch.autograd.grad(energy1.sum(), coordinates)[0]
        energy2 = [m((species, coordinates))[1] for m in model_list]
        energy2 = sum(energy2) / len(model_list)
        force2 = torch.autograd.grad(energy2.sum(), coordinates)[0]
        self.assertEqual(energy1, energy2)
        self.assertEqual(force1, force2)

    def testGDB(self):
        for i in range(N):
            datafile = os.path.join(path, 'test_data/ANI1_subset/{}'.format(i))
            with open(datafile, 'rb') as f:
                coordinates, species, _, _, _, _ = pickle.load(f)
                coordinates = torch.from_numpy(coordinates)
                species = torch.from_numpy(species)
                self._test_molecule(coordinates, species)


class TestEnsembleJIT(TestEnsemble):

    def setUp(self):
        super().setUp()
        self.ensemble = torchani.nn.Sequential(self.aev_computer, self.model_iterator)
        self.ensemble = torch.jit.script(self.ensemble)


if __name__ == '__main__':
    unittest.main()
