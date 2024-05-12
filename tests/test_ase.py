import typing as tp
from itertools import product
import os
import unittest
import pickle

import torch
import numpy as np
from parameterized import parameterized
from ase import units, Atoms
from ase.io import read
from ase.optimize import BFGS
from ase.lattice.cubic import Diamond
from ase.md.langevin import Langevin
from ase.md.nptberendsen import NPTBerendsen
from ase.calculators.test import numeric_force

from torchani.neighbors import CellList
from torchani.testing import ANITest, expand
from torchani.models import ANI1x, ANIdr, PairPotentialsModel
from torchani.potentials import PairPotential


path = os.path.dirname(os.path.realpath(__file__))


def _stress_test_name(fn: tp.Any, idx: int, param: tp.Any) -> str:
    nl = ""
    if param.args[2] == "cell_list":
        nl = "cell"
    elif param.args[2] == "full_pairwise":
        nl = "allpairs"
    return f"{fn.__name__}_fdotr_{param.args[0]}_repdisp_{param.args[1]}_{nl}"


@expand(jit=False)
class TestASE(ANITest):
    def testConsistentForcesCellWithAllPairs(self):
        model = self._setup(ANI1x(model_index=0).double())
        model_cell = self._setup(
            ANI1x(model_index=0, neighborlist="cell_list").double()
        )

        f_cell = self._testForcesPBC(model_cell, only_get_forces=True)
        f = self._testForcesPBC(model, only_get_forces=True)
        self.assertEqual(f, f_cell, rtol=0.1, atol=0.1)

    def testConsistentForcesWithPairPotentialModel(self):
        model = self._setup(ANI1x(model_index=0).double())
        model_cell = self._setup(
            ANI1x(model_index=0, neighborlist="cell_list").double()
        )
        model_pair = self._setup(
            PairPotentialsModel(
                aev_computer=model_cell.aev_computer,
                neural_networks=model_cell.neural_networks,
                energy_shifter=model_cell.energy_shifter,
                elements=model_cell.get_chemical_symbols(),
                pairwise_potentials=[
                    PairPotential(cutoff=6.4),
                    PairPotential(cutoff=5.2),
                    PairPotential(cutoff=3.0),
                ],
            ).double()
        )

        f_cell = self._testForcesPBC(model_cell, only_get_forces=True)
        f_pair = self._testForcesPBC(model_pair, only_get_forces=True)
        f = self._testForcesPBC(model, only_get_forces=True)
        self.assertEqual(f_pair, f_cell, rtol=0.1, atol=0.1)
        self.assertEqual(f_pair, f, rtol=0.1, atol=0.1)

    def testNumericalForcesAllPairs(self):
        model = self._setup(ANI1x(model_index=0).double())
        self._testForcesPBC(model, repeats=1)

    def testNumericalForcesCellList(self):
        model = self._setup(ANI1x(model_index=0, neighborlist="cell_list").double())
        self._testForcesPBC(model)

    def testNumericalForcesCellListConstantV(self):
        model = self._setup(
            ANI1x(model_index=0, neighborlist=CellList(constant_volume=True)).double()
        )
        self._testForcesPBC(model)

    def _testForcesPBC(self, model, only_get_forces=False, repeats=2, steps=10):
        # Run a Langevin thermostat dynamic for 100 steps and after the dynamic
        # check once that the numerical and analytical force agree to a given
        # relative tolerance
        prng = np.random.RandomState(seed=1234)
        atoms = Diamond(symbol="C", pbc=True, size=(repeats, repeats, repeats))
        calculator = model.ase()
        atoms.calc = calculator
        dyn = Langevin(
            atoms, timestep=0.5 * units.fs, temperature_K=3820, friction=0.002, rng=prng
        )
        dyn.run(steps)
        f = atoms.get_forces()
        if only_get_forces:
            return f
        # only test 10 atoms maximum, since this is very slow
        if len(atoms) > 10:
            num_atoms = 10
        else:
            num_atoms = len(atoms)

        def _get_numeric_force(atoms, eps, num_atoms):
            fn = torch.zeros((num_atoms, 3), dtype=torch.double)
            for i in range(num_atoms):
                for j in range(3):
                    fn[i, j] = numeric_force(atoms, i, j, eps)
            return fn

        fn = _get_numeric_force(atoms, 0.001, num_atoms)
        self.assertEqual(f[:num_atoms, :], fn, rtol=0.1, atol=0.1)

    @parameterized.expand(
        product((True, False), (True, False), ("full_pairwise", "cell_list")),
        name_func=_stress_test_name,
    )
    def testAnalyticalStressMatchNumerical(
        self,
        stress_partial_fdotr,
        repdisp,
        neighborlist,
    ):
        if repdisp:
            if neighborlist == "cell_list":
                self.skipTest(
                    "Cell used in this test is too small for dispersion potential"
                )
            model = self._setup(
                ANIdr(model_index=0, neighborlist=neighborlist).double()
            )
        else:
            model = self._setup(
                ANI1x(model_index=0, neighborlist=neighborlist).double()
            )
        # Run NPT dynamics for some steps and periodically check that the
        # numerical and analytical stresses agree up to a given
        # absolute difference
        filename = os.path.join(
            path, "../tools/generate-unit-test-expect/others/Benzene.json"
        )
        benzene = read(filename)
        # set velocities to a very small value to avoid division by zero
        # warning due to initial zero temperature.
        #
        # Note that there are 4 benzene molecules, thus, 48 atoms in
        # Benzene.json
        benzene.set_velocities(np.full((48, 3), 1e-15))
        calculator = model.ase(stress_partial_fdotr=stress_partial_fdotr)
        benzene.calc = calculator
        dyn = NPTBerendsen(
            benzene,
            timestep=0.1 * units.fs,
            temperature_K=300,
            taut=0.1 * 1000 * units.fs,
            pressure_au=1.0 * units.bar,
            taup=1.0 * 1000 * units.fs,
            compressibility_au=4.57e-5 / units.bar,
        )

        def test_stress():
            stress = benzene.get_stress()
            numerical_stress = calculator.calculate_numerical_stress(benzene)
            self.assertEqual(stress, numerical_stress)

        dyn.attach(test_stress, interval=2)
        dyn.run(10)


@expand(jit=False)
class TestOptimizationASE(ANITest):
    def setUp(self):
        self.tolerance = 1e-6
        self.calculator = self._setup(ANI1x(model_index=0)).ase()

    def testCoordsRMSE(self):
        datafile = os.path.join(path, "test_data/NeuroChemOptimized/all")
        with open(datafile, "rb") as f:
            systems = pickle.load(f)
            for system in systems:
                # reconstructing Atoms object.
                # ASE does not support loading pickled object from older version
                positions = system.get_positions()
                symbols = system.get_chemical_symbols()
                atoms = Atoms(symbols, positions=positions)
                old_coordinates = torch.tensor(positions.copy(), device=self.device)
                atoms.calc = self.calculator
                opt = BFGS(atoms)
                opt.run()
                coordinates = atoms.get_positions()
                self.assertEqual(old_coordinates, coordinates)


if __name__ == "__main__":
    unittest.main(verbosity=2)
