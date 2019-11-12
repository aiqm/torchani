from ase.lattice.cubic import Diamond
from ase.md.langevin import Langevin
from ase.md.nptberendsen import NPTBerendsen
from ase import units
from ase.io import read
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

    def setUp(self):
        self.model = torchani.models.ANI1x().double()[0]

    def testWithNumericalForceWithPBCEnabled(self):
        atoms = Diamond(symbol="C", pbc=True)
        calculator = self.model.ase()
        atoms.set_calculator(calculator)
        dyn = Langevin(atoms, 5 * units.fs, 30000000 * units.kB, 0.002)
        dyn.run(100)
        f = torch.from_numpy(atoms.get_forces())
        fn = get_numeric_force(atoms, 0.001)
        df = (f - fn).abs().max()
        avgf = f.abs().mean()
        if avgf > 0:
            self.assertLess(df / avgf, 0.1)

    def testWithNumericalStressWithPBCEnabled(self):
        filename = os.path.join(path, '../tools/generate-unit-test-expect/others/Benzene.cif')
        benzene = read(filename)
        calculator = self.model.ase()
        benzene.set_calculator(calculator)
        dyn = NPTBerendsen(benzene, timestep=0.1 * units.fs,
                           temperature=300 * units.kB,
                           taut=0.1 * 1000 * units.fs, pressure=1.01325,
                           taup=1.0 * 1000 * units.fs, compressibility=4.57e-5)

        def test_stress():
            stress = benzene.get_stress()
            numerical_stress = calculator.calculate_numerical_stress(benzene)
            diff = torch.from_numpy(stress - numerical_stress).abs().max().item()
            self.assertLess(diff, tol)
        dyn.attach(test_stress, interval=30)
        dyn.run(120)


if __name__ == '__main__':
    unittest.main()
