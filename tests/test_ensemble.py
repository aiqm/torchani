import unittest
import pickle
import os
import torch
import torchani

path = os.path.dirname(os.path.realpath(__file__))
N = 97


class TestEnsemble(unittest.TestCase):

    def setUp(self):
        self.tol = 1e-5
        self.conformations = 20

    def _test_molecule(self, coordinates, species):
        n = torchani.buildin_ensemble
        prefix = torchani.buildin_model_prefix
        aev = torchani.SortedAEV(device=torch.device('cpu'))
        coordinates, species = aev.sort_by_species(coordinates, species)
        ensemble = torchani.models.NeuroChemNNP(aev, derivative=True,
                                                ensemble=True)
        models = [torchani.models.
                  NeuroChemNNP(aev, derivative=True,
                               ensemble=False,
                               from_=prefix + '{}/networks/'.format(i))
                  for i in range(n)]

        energy1, force1 = ensemble(coordinates, species)
        energy2, force2 = zip(*[m(coordinates, species) for m in models])
        energy2 = sum(energy2) / n
        force2 = sum(force2) / n
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
