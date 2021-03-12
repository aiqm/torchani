from ase.lattice.cubic import Diamond
from ase.md.langevin import Langevin
from ase.md.nptberendsen import NPTBerendsen
from ase import units
from ase.io import read
from ase.calculators.test import numeric_force
from torchani.testing import TestCase
import numpy as np
import torch
import torchani
import unittest
import os

path = os.path.dirname(os.path.realpath(__file__))


def get_numeric_force(atoms, eps):
    fn = torch.zeros((len(atoms), 3), dtype=torch.double)
    for i in range(len(atoms)):
        for j in range(3):
            fn[i, j] = numeric_force(atoms, i, j, eps)
    return fn


class TestASE(TestCase):

    def setUp(self):
        self.model = torchani.models.ANI1x(model_index=0).double()

    def testWithNumericalForceWithPBCEnabled(self):
        # Run a Langevin thermostat dynamic for 100 steps and after the dynamic
        # check once that the numerical and analytical force agree to a given
        # relative tolerance
        atoms = Diamond(symbol="C", pbc=True)
        calculator = self.model.ase()
        atoms.set_calculator(calculator)
        dyn = Langevin(atoms, 5 * units.fs, 30000000 * units.kB, 0.002)
        dyn.run(100)
        f = atoms.get_forces()
        fn = get_numeric_force(atoms, 0.001)
        self.assertEqual(f, fn, rtol=0.1, atol=0.1)

    def testWithNumericalStressWithPBCEnabled(self):
        # Run NPT dynamics for some steps and periodically check that the
        # numerical and analytical stresses agree up to a given
        # absolute difference
        filename = os.path.join(path, '../tools/generate-unit-test-expect/others/Benzene.json')
        benzene = read(filename)
        # set velocities to a very small value to avoid division by zero
        # warning due to initial zero temperature.
        #
        # Note that there are 4 benzene molecules, thus, 48 atoms in
        # Benzene.json
        benzene.set_velocities(np.full((48, 3), 1e-15))
        calculator = self.model.ase()
        benzene.set_calculator(calculator)
        dyn = NPTBerendsen(benzene, timestep=0.1 * units.fs,
                           temperature=300 * units.kB,
                           taut=0.1 * 1000 * units.fs, pressure=1.01325,
                           taup=1.0 * 1000 * units.fs, compressibility=4.57e-5)

        def test_stress():
            stress = benzene.get_stress()
            numerical_stress = calculator.calculate_numerical_stress(benzene)
            self.assertEqual(stress, numerical_stress)
        dyn.attach(test_stress, interval=30)
        dyn.run(120)


class TestASEWithPTI(unittest.TestCase):
    # Tests that the values obtained by wrapping a BuiltinModel or
    # BuiltinEnsemble with a calculator are the same with and without
    # periodic_table_index

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
