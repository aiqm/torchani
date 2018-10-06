import unittest
import pickle
import os
import torch
import torchani

path = os.path.dirname(os.path.realpath(__file__))
N = 10


class TestEnsemble(unittest.TestCase):

    def setUp(self):
        self.tol = 1e-5
        self.conformations = 20

    def _test_molecule(self, coordinates, species):
        builtins = torchani.neurochem.Builtins()
        coordinates.requires_grad_(True)
        aev = builtins.aev_computer
        ensemble = builtins.models
        models = [torch.nn.Sequential(aev, m) for m in ensemble]
        ensemble = torch.nn.Sequential(aev, ensemble)

        _, energy1 = ensemble((species, coordinates))
        force1 = torch.autograd.grad(energy1.sum(), coordinates)[0]
        energy2 = [m((species, coordinates))[1] for m in models]
        energy2 = sum(energy2) / len(models)
        force2 = torch.autograd.grad(energy2.sum(), coordinates)[0]
        energy_diff = (energy1 - energy2).abs().max().item()
        force_diff = (force1 - force2).abs().max().item()
        self.assertLess(energy_diff, self.tol)
        self.assertLess(force_diff, self.tol)

    def testGDB(self):
        for i in range(N):
            datafile = os.path.join(path, 'test_data/{}'.format(i))
            with open(datafile, 'rb') as f:
                coordinates, species, _, _, _, _ = pickle.load(f)
                self._test_molecule(coordinates, species)


if __name__ == '__main__':
    unittest.main()
