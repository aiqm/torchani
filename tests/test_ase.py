from ase.lattice.cubic import Diamond
from ase.md.langevin import Langevin
from ase import units
from ase.calculators.test import numeric_force
import torch
import torchani
import unittest
import os

path = os.path.dirname(os.path.realpath(__file__))
N = 97
tol = 5e-5


def get_numeric_force(atoms, eps):
    fn = torch.zeros((len(atoms), 3), dtype=torch.double)
    for i in range(len(atoms)):
        for j in range(3):
            fn[i, j] = numeric_force(atoms, i, j, eps)
    return fn


class TestASE(unittest.TestCase):

    def testWithNumericalForceWithPBCEnabled(self):
        atoms = Diamond(symbol="C", pbc=True)
        builtin = torchani.neurochem.Builtins()
        calculator = torchani.ase.Calculator(
            builtin.species, builtin.aev_computer,
            builtin.models, builtin.energy_shifter)
        atoms.set_calculator(calculator)
        dyn = Langevin(atoms, 5 * units.fs, 30000000 * units.kB, 0.002)
        dyn.run(100)
        f = torch.from_numpy(atoms.get_forces())
        fn = get_numeric_force(atoms, 0.001)
        df = (f - fn).abs().max()
        avgf = f.abs().mean()
        if avgf > 0:
            self.assertLess(df / avgf, 0.1)


if __name__ == '__main__':
    unittest.main()
