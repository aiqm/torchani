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
from torchani.aev import CellList
import unittest
import os

path = os.path.dirname(os.path.realpath(__file__))


class TestASE(TestCase):

    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def testConsistentForcesCellWithPairwise(self):
        # Run a Langevin thermostat dynamic for 100 steps and after the dynamic
        # check once that the numerical and analytical force agree to a given
        # relative tolerance
        model_cell = torchani.models.ANI1x(model_index=0, cell_list=True)
        model_cell = model_cell.to(dtype=torch.double, device=self.device)
        model = torchani.models.ANI1x(model_index=0)
        model = model.to(dtype=torch.double, device=self.device)

        f_cell = self._testForcesPBC(model_cell, only_get_forces=True)
        f = self._testForcesPBC(model, only_get_forces=True)
        self.assertEqual(f, f_cell, rtol=0.1, atol=0.1)

    def testConsistentForcesCellListVerlet(self):
        # Run a Langevin thermostat dynamic for 100 steps and after the dynamic
        # check once that the numerical and analytical force agree to a given
        # relative tolerance
        model_cell = torchani.models.ANI1x(model_index=0, cell_list=True)
        model_cell = model_cell.to(dtype=torch.double, device=self.device)
        model_dyn = torchani.models.ANI1x(model_index=0, cell_list=True)
        model_dyn.aev_computer.neighborlist = CellList(model_dyn.aev_computer.radial_terms.cutoff, verlet=True)
        model_dyn = model_cell.to(dtype=torch.double, device=self.device)

        f_cell = self._testForcesPBC(model_cell, only_get_forces=True)
        f_dyn = self._testForcesPBC(model_dyn, only_get_forces=True)
        self.assertEqual(f_dyn, f_cell, rtol=0.1, atol=0.1)

    def testNumericalForcesFullPairwise(self):
        model = torchani.models.ANI1x(model_index=0)
        model = model.to(dtype=torch.double, device=self.device)
        self._testForcesPBC(model, repeats=1)

    def testNumericalForcesCellList(self):
        model = torchani.models.ANI1x(model_index=0, cell_list=True)
        model = model.to(dtype=torch.double, device=self.device)
        self._testForcesPBC(model)

    def testNumericalForcesCellListVerlet(self):
        model = torchani.models.ANI1x(model_index=0, cell_list=True)
        model.aev_computer.neighborlist = CellList(model.aev_computer.radial_terms.cutoff, verlet=True)
        model = model.to(dtype=torch.double, device=self.device)
        self._testForcesPBC(model, steps=100)

    def testNumericalForcesCellListConstantV(self):
        model = torchani.models.ANI1x(model_index=0, cell_list=True)
        model.aev_computer.neighborlist = CellList(model.aev_computer.radial_terms.cutoff, constant_volume=True)
        model = model.to(dtype=torch.double, device=self.device)
        self._testForcesPBC(model)

    def _testForcesPBC(self, model, only_get_forces=False, repeats=2, steps=10):
        # Run a Langevin thermostat dynamic for 100 steps and after the dynamic
        # check once that the numerical and analytical force agree to a given
        # relative tolerance
        prng = np.random.RandomState(seed=1234)
        atoms = Diamond(symbol="C", pbc=True, size=(repeats, repeats, repeats))
        calculator = model.ase()
        atoms.calc = calculator
        dyn = Langevin(atoms, timestep=0.5 * units.fs, temperature_K=3820, friction=0.002, rng=prng)
        dyn.run(steps)
        f = atoms.get_forces()
        if only_get_forces:
            return f
        # only test 10 atoms maximum, since this is very slow
        if len(atoms) > 10:
            num_atoms = 10
        else:
            num_atoms = len(atoms)

        fn = self._get_numeric_force(atoms, 0.001, num_atoms)
        self.assertEqual(f[:num_atoms, :], fn, rtol=0.1, atol=0.1)

    @staticmethod
    def _get_numeric_force(atoms, eps, num_atoms):
        fn = torch.zeros((num_atoms, 3), dtype=torch.double)
        for i in range(num_atoms):
            for j in range(3):
                fn[i, j] = numeric_force(atoms, i, j, eps)
        return fn

    def testWithNumericalStressFullPairwise(self):
        model = torchani.models.ANI1x(model_index=0)
        model = model.to(dtype=torch.double, device=self.device)
        self._testWithNumericalStressPBC(model)

    def testWithNumericalStressCellList(self):
        model = torchani.models.ANI1x(model_index=0, cell_list=True)
        model = model.to(dtype=torch.double, device=self.device)
        self._testWithNumericalStressPBC(model)

    def _testWithNumericalStressPBC(self, model):
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
        calculator = model.ase()
        benzene.calc = calculator
        dyn = NPTBerendsen(benzene, timestep=0.1 * units.fs,
                           temperature_K=300,
                           taut=0.1 * 1000 * units.fs, pressure_au=1.0 * units.bar,
                           taup=1.0 * 1000 * units.fs, compressibility_au=4.57e-5 / units.bar)

        def test_stress():
            stress = benzene.get_stress()
            numerical_stress = calculator.calculate_numerical_stress(benzene)
            self.assertEqual(stress, numerical_stress)

        dyn.attach(test_stress, interval=2)
        dyn.run(10)


class TestASEWithPeriodicTableIndex(unittest.TestCase):
    # Tests that the values obtained by wrapping a BuiltinModel or
    # BuiltinEnsemble with a calculator are the same with and without
    # periodic_table_index

    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_pti = torchani.models.ANI1x(periodic_table_index=True)
        self.model = torchani.models.ANI1x()
        self.model = self.model.to(dtype=torch.double, device=self.device)
        self.model_pti = self.model_pti.to(dtype=torch.double, device=self.device)

    def testEqualEnsemblePeriodicTableIndex(self):
        calculator_pti = self.model_pti.ase()
        calculator = self.model.ase()
        atoms = Diamond(symbol="C", pbc=True)
        atoms_pti = Diamond(symbol="C", pbc=True)
        atoms.calc = calculator
        atoms_pti.calc = calculator_pti
        self.assertEqual(atoms.get_potential_energy(), atoms_pti.get_potential_energy())

    def testEqualOneModelPeriodicTableIndex(self):
        calculator_pti = self.model_pti[0].ase()
        calculator = self.model[0].ase()
        atoms = Diamond(symbol="C", pbc=True)
        atoms_pti = Diamond(symbol="C", pbc=True)
        atoms.calc = calculator
        atoms_pti.calc = calculator_pti
        self.assertEqual(atoms.get_potential_energy(), atoms_pti.get_potential_energy())


if __name__ == '__main__':
    unittest.main()
