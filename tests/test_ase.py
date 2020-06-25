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
tol = 5e-5


def get_numeric_force(atoms, eps):
    fn = torch.zeros((len(atoms), 3), dtype=torch.double)
    for i in range(len(atoms)):
        for j in range(3):
            fn[i, j] = numeric_force(atoms, i, j, eps)
    return fn


class TestASE(unittest.TestCase):

    def setUp(self):
        self.model = torchani.models.ANI1x(model_index=0).double()

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


class TestASEWithPTI(unittest.TestCase):

    def setUp(self):
        self.model_pti = torchani.models.ANI1x(periodic_table_index=True).double()
        self.model = torchani.models.ANI1x().double()

    def testEqualEnsemblePTI(self):
        calculator_pti = self.model_pti.ase()
        calculator = self.model.ase()
        atoms = Diamond(symbol="C", pbc=True)
        atoms_pti = Diamond(symbol="C", pbc=True)
        atoms.set_calculator(calculator)
        atoms_pti.set_calculator(calculator_pti)
        self.assertEqual(atoms.get_potential_energy(), atoms_pti.get_potential_energy())

    def testEqualOneModelPTI(self):
        calculator_pti = self.model_pti[0].ase()
        calculator = self.model[0].ase()
        atoms = Diamond(symbol="C", pbc=True)
        atoms_pti = Diamond(symbol="C", pbc=True)
        atoms.set_calculator(calculator)
        atoms_pti.set_calculator(calculator_pti)
        self.assertEqual(atoms.get_potential_energy(), atoms_pti.get_potential_energy())


if __name__ == '__main__':
    unittest.main()
