import torch
import torchani
import unittest
import os
import pickle

path = os.path.dirname(os.path.realpath(__file__))
N = 97


class TestForce(unittest.TestCase):

    def setUp(self):
        self.tolerance = 1e-5
        aev_computer = torchani.AEVComputer()
        prepare = torchani.PrepareInput(aev_computer.species)
        nnp = torchani.models.NeuroChemNNP(aev_computer.species)
        self.model = torch.nn.Sequential(prepare, aev_computer, nnp)

    def _test_molecule(self, coordinates, species, forces):
        # generate a random permute
        atoms = len(species)
        randperm = torch.randperm(atoms)
        coordinates = coordinates.index_select(1, randperm)
        forces = forces.index_select(1, randperm)
        species = [species[i] for i in randperm.tolist()]

        coordinates = torch.tensor(coordinates, requires_grad=True)
        _, energies = self.model((species, coordinates))
        derivative = torch.autograd.grad(energies.sum(), coordinates)[0]
        max_diff = (forces + derivative).abs().max().item()
        self.assertLess(max_diff, self.tolerance)

    def testGDB(self):
        for i in range(N):
            datafile = os.path.join(path, 'test_data/{}'.format(i))
            with open(datafile, 'rb') as f:
                coordinates, species, _, _, _, forces = pickle.load(f)
                self._test_molecule(coordinates, species, forces)


if __name__ == '__main__':
    unittest.main()
