import torch
import shutil
import typing as tp
from pathlib import Path
from itertools import product
import warnings
import unittest
import pickle

import numpy as np
from numpy.typing import NDArray
from parameterized import parameterized

from torchani import ASE_IS_AVAILABLE

if not ASE_IS_AVAILABLE:
    warnings.warn("Skipping all ASE tests, install ase to run them")
    raise unittest.SkipTest("ASE is not available, skipping all ASE tests.")

from ase import units, Atoms
from ase.vibrations import Vibrations
from ase.optimize import BFGS
from ase.lattice.cubic import Diamond
from ase.md.langevin import Langevin
from ase.md.nptberendsen import NPTBerendsen
from ase.calculators.test import numeric_force

from torchani.io import read_xyz
from torchani._testing import ANITestCase, expand
from torchani.arch import ANI
from torchani.models import ANI1x, ANIdr
from torchani.potentials import DummyPotential


def _stress_test_name(fn: tp.Any, idx: int, param: tp.Any) -> str:
    nl = ""
    if param.args[2] == "cell_list":
        nl = "cell"
    elif param.args[2] == "all_pairs":
        nl = "allpairs"
    return f"{fn.__name__}_fdotr_{param.args[0]}_repdisp_{param.args[1]}_{nl}"


@expand(jit=False)
class TestASE(ANITestCase):
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
        symbols = model_cell.symbols
        model_pair = self._setup(
            ANI(
                aev_computer=model_cell.aev_computer,
                neural_networks=model_cell.neural_networks,
                energy_shifter=model_cell.energy_shifter,
                symbols=model_cell.symbols,
                potentials={
                    "dummy-0": DummyPotential(symbols, cutoff=6.4),
                    "dummy-1": DummyPotential(symbols, cutoff=5.2),
                    "dummy-2": DummyPotential(symbols, cutoff=3.0),
                },
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
            fn = np.zeros((num_atoms, 3), dtype=np.float64)
            for i in range(num_atoms):
                for j in range(3):
                    fn[i, j] = numeric_force(atoms, i, j, eps)
            return fn

        fn = _get_numeric_force(atoms, 0.001, num_atoms)
        self.assertEqual(f[:num_atoms, :], fn, rtol=0.1, atol=0.1)

    @parameterized.expand(
        product((True, False), (True, False), ("all_pairs", "cell_list")),
        name_func=_stress_test_name,
    )
    def testAnalyticalStressMatchNumerical(
        self,
        use_fdotr,
        repdisp,
        neighborlist,
    ):
        if repdisp:
            if neighborlist == "cell_list":
                self.skipTest(
                    "Cell used in this test is too small for dispersion potential"
                )
            model = self._setup(
                ANIdr(model_index=0, neighborlist=neighborlist, dtype=torch.double)
            )
        else:
            model = self._setup(
                ANI1x(model_index=0, neighborlist=neighborlist, dtype=torch.double)
            )
        # Run NPT dynamics for some steps and periodically check that the
        # numerical and analytical stresses agree up to a given
        # absolute difference
        # Note that there are 4 benzene molecules, thus, 48 atoms in
        # Benzene.xyz
        species, coordinates, cell, pbc = read_xyz(
            (Path(__file__).parent / "resources") / "benzene.xyz"
        )
        assert pbc is not None
        assert cell is not None
        benzene = Atoms(
            numbers=species.squeeze(0).numpy(),
            positions=coordinates.squeeze(0).numpy(),
            cell=cell.numpy(),
            velocities=np.full((species.shape[1], 3), 1e-15),  # Set vels to small value
            pbc=pbc.numpy(),
        )
        calculator = model.ase(stress_kind="fdotr" if use_fdotr else "scaling")
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
class TestVibrationsASE(ANITestCase):
    def tearDown(self) -> None:
        vib_path = Path(Path(__file__).parent, "vib")
        if vib_path.is_dir():
            shutil.rmtree(vib_path)

    def testWater(self) -> None:
        model = ANI1x().double()
        data_path = Path(Path(__file__).parent, "resources", "water-vib-expect.npz")
        with np.load(data_path) as data:
            coordinates = data["coordinates"]
            species = data["species"]
            modes_expect = data["modes"]
            freqs_expect = data["freqs"]
        molecule = Atoms(numbers=species, positions=coordinates, calculator=model.ase())
        # Compute vibrational frequencies with ASE
        vib = Vibrations(molecule)
        vib.run()
        array_freqs = np.array([np.real(x) for x in vib.get_frequencies()[6:]])
        _modes: tp.List[NDArray[np.float64]] = []
        for j in range(6, 6 + len(array_freqs)):
            _modes.append(np.expand_dims(vib.get_mode(j), axis=0))
        vib.clean()
        modes = np.concatenate(_modes, axis=0)
        self.assertEqual(
            freqs_expect,
            freqs_expect,
            atol=0,
            rtol=0.02,
            exact_dtype=False,
        )
        diff1 = np.abs(modes_expect - modes).max(axis=-1).max(axis=-1)
        diff2 = np.abs(modes_expect + modes).max(axis=-1).max(axis=-1)
        diff = np.where(diff1 < diff2, diff1, diff2)
        self.assertLess(float(diff.max().item()), 0.02)


@expand(jit=False)
class TestOptimizationASE(ANITestCase):
    def setUp(self):
        self.tolerance = 1e-6
        self.calculator = self._setup(ANI1x(model_index=0)).ase()

    def testCoordsRMSE(self):
        path_parts = ["resources", "NeuroChemOptimized", "all"]
        data_file = Path(Path(__file__).parent, *path_parts).resolve()
        with open(data_file, "rb") as f:
            systems = pickle.load(f)
            for system in systems:
                # reconstructing Atoms object.
                # ASE does not support loading pickled object from older version
                positions = system.get_positions()
                symbols = system.get_chemical_symbols()
                atoms = Atoms(symbols, positions=positions)
                old_coordinates = positions.copy()
                atoms.calc = self.calculator
                opt = BFGS(atoms)
                opt.run()
                coordinates = atoms.get_positions()
                self.assertEqual(old_coordinates, coordinates)


if __name__ == "__main__":
    unittest.main(verbosity=2)
