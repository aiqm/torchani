import os
import unittest
import torch
import torchani
import copy
import pickle
from ase.optimize import BFGS
from ase import Atoms
from torchani.testing import TestCase


path = os.path.dirname(os.path.realpath(__file__))


class TestStructureOptimization(TestCase):

    def setUp(self):
        self.tolerance = 1e-6
        self.calculator = torchani.models.ANI1x(model_index=0).ase()

    def testRMSE(self):
        datafile = os.path.join(path, 'test_data/NeuroChemOptimized/all')
        with open(datafile, 'rb') as f:
            all_atoms = pickle.load(f)
            for atoms in all_atoms:
                # reconstructing Atoms object.
                # ASE does not support loading pickled object from older version
                atoms = Atoms(atoms.get_chemical_symbols(), positions=atoms.get_positions())
                old_coordinates = copy.deepcopy(atoms.get_positions())
                old_coordinates = torch.from_numpy(old_coordinates)
                atoms.calc = self.calculator
                opt = BFGS(atoms)
                opt.run()
                coordinates = atoms.get_positions()
                self.assertEqual(old_coordinates, coordinates)


if __name__ == '__main__':
    unittest.main()
