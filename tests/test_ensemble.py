import unittest
import torch
import torchani


class TestEnsemble(unittest.TestCase):

    def setUp(self):
        self.tol = 1e-5
        self.conformations = 20

    def _test_molecule(self, coordinates, species):
        prefix = torchani.buildin_model_prefix
        n = torchani.buildin_ensembles
        aev = torchani.SortedAEV(device=torch.device('cpu'))
        coordinates, species = aev.sort_by_species(coordinates, species)
        ensemble = torchani.ModelOnAEV(aev, derivative=True,
                                       from_nc=prefix,
                                       ensemble=n)
        models = [torchani.ModelOnAEV(aev, derivative=True,
                  from_nc=prefix + '{}/networks/'.format(i)) for i in range(n)]

        energy1, force1 = ensemble(coordinates, species)
        energy2, force2 = zip(*[m(coordinates, species) for m in models])
        energy2 = sum(energy2) / n
        force2 = sum(force2) / n
        energy_diff = (energy1 - energy2).abs().max().item()
        force_diff = (force1 - force2).abs().max().item()
        self.assertLess(energy_diff, self.tol)
        self.assertLess(force_diff, self.tol)

    def testRandomHH(self):
        coordinates = torch.rand(self.conformations, 1, 3)
        self._test_molecule(coordinates, ['H', 'H'])

    def testRandomHC(self):
        coordinates = torch.rand(self.conformations, 2, 3)
        self._test_molecule(coordinates, ['H', 'C'])

    def testRandomHNCONCHO(self):
        coordinates = torch.rand(self.conformations, 8, 3)
        self._test_molecule(coordinates, list('HNCONCHO'))


if __name__ == '__main__':
    unittest.main()
