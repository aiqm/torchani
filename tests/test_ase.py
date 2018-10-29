from ase.lattice.cubic import Diamond
from ase.md.langevin import Langevin
from ase import units
from ase.calculators.test import numeric_force
import torch
import torchani
import unittest


def get_numeric_force(atoms, eps):
    fn = torch.zeros((len(atoms), 3))
    for i in range(len(atoms)):
        for j in range(3):
            fn[i, j] = numeric_force(atoms, i, j, eps)
    return fn


class TestASE(unittest.TestCase):

    def _testForce(self, pbc):
        atoms = Diamond(symbol="C", pbc=pbc)
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

    def testForceWithPBCEnabled(self):
        self._testForce(True)

    def testForceWithPBCDisabled(self):
        self._testForce(False)


if __name__ == '__main__':
    unittest.main()
