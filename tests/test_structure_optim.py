import os
import unittest
import torch
import torchani
import copy
import pickle
from ase.optimize import BFGS


path = os.path.dirname(os.path.realpath(__file__))


class TestStructureOptimization(unittest.TestCase):

    def setUp(self):
        self.tolerance = 1e-6
        self.builtin = torchani.neurochem.Builtins()
        self.calculator = torchani.ase.Calculator(
            self.builtin.species, self.builtin.aev_computer,
            self.builtin.models[0], self.builtin.energy_shifter)

    def testRMSE(self):
        datafile = os.path.join(path, 'test_data/NeuroChemOptimized/all')
        with open(datafile, 'rb') as f:
            all_atoms = pickle.load(f)
            for atoms in all_atoms:
                old_coordinates = copy.deepcopy(atoms.get_positions())
                old_coordinates = torch.from_numpy(old_coordinates)
                atoms.set_calculator(self.calculator)
                opt = BFGS(atoms)
                opt.run()
                coordinates = atoms.get_positions()
                coordinates = torch.from_numpy(coordinates)
                distances = (old_coordinates - coordinates).norm(dim=1)
                rmse = distances.mean()
                self.assertLess(rmse, self.tolerance)


if __name__ == '__main__':
    unittest.main()
