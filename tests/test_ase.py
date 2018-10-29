from ase.lattice.cubic import Diamond
from ase.md.langevin import Langevin
from ase import units, Atoms
from ase.calculators.test import numeric_force
import torch
import torchani
import unittest
import numpy


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

    def testForceAgainstDefaultNeighborList(self):
        atoms = Diamond(symbol="C", pbc=False)
        builtin = torchani.neurochem.Builtins()

        calculator = torchani.ase.Calculator(
            builtin.species, builtin.aev_computer,
            builtin.models, builtin.energy_shifter)
        default_neighborlist_calculator = torchani.ase.Calculator(
            builtin.species, builtin.aev_computer,
            builtin.models, builtin.energy_shifter, True)

        atoms.set_calculator(calculator)
        dyn = Langevin(atoms, 5 * units.fs, 50 * units.kB, 0.002)

        def test_energy(a=atoms):
            a = a.copy()
            a.set_calculator(calculator)
            e1 = a.get_potential_energy()
            a.set_calculator(default_neighborlist_calculator)
            e2 = a.get_potential_energy()
            self.assertEqual(e1, e2)

        dyn.attach(test_energy, interval=1)
        dyn.run(500)

    def testTranslationalInvariancePBC(self):
        atoms = Atoms('CH4', [[0, 0, 0],
                              [1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1],
                              [0, 1, 1],],
                      cell=[2,2,2], pbc=True)

        builtin = torchani.neurochem.Builtins()
        calculator = torchani.ase.Calculator(
            builtin.species, builtin.aev_computer,
            builtin.models, builtin.energy_shifter)
        atoms.set_calculator(calculator)
        e = atoms.get_potential_energy()

        for _ in range(100):
            positions = atoms.get_positions()
            translation = (numpy.random.rand(3) - 0.5) * 2
            atoms.set_positions(positions + translation)
            self.assertEqual(e, atoms.get_potential_energy())


if __name__ == '__main__':
    unittest.main()
