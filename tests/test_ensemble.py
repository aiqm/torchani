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
        coordinates = torch.tensor(coordinates, requires_grad=True)
        n = torchani.buildin_ensemble
        prefix = torchani.buildin_model_prefix
        aev = torchani.SortedAEV(device=torch.device('cpu'))
        ensemble = torchani.models.NeuroChemNNP(aev, ensemble=True)
        models = [torchani.models.
                  NeuroChemNNP(aev, ensemble=False,
                               from_=prefix + '{}/networks/'.format(i))
                  for i in range(n)]

        energy1 = ensemble(coordinates, species)
        force1 = torch.autograd.grad(energy1.sum(), coordinates)[0]
        energy2 = [m(coordinates, species) for m in models]
        energy2 = sum(energy2) / n
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
