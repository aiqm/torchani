import math
import typing as tp
import unittest

import torch

from torchani._testing import ANITestCase, expand, make_molecs
from torchani.arch import simple_aniq
from torchani.potentials import Coulomb, FixedCoulomb
from torchani.units import ANGSTROM_TO_BOHR

SYMBOLS = ("H", "C", "N", "O")
ELEM_CHARGES = (0.1, -0.2, 0.3, -0.4)


@expand()
class TestCoulomb(ANITestCase):
    def setUp(self):
        self.coords = torch.tensor(
            [[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]], device=self.device
        )
        self.atomic_nums = torch.tensor([[1, 1]], device=self.device)
        self.qs = torch.tensor([[0.5, -0.5]], device=self.device)
        self.dist = 2.0 * ANGSTROM_TO_BOHR

    def testTwoCharges(self):
        pot = self._setup(Coulomb(SYMBOLS))
        energies = pot(self.atomic_nums, self.coords, scalars=self.qs)
        expect = torch.tensor([-0.25 / self.dist], device=self.device)
        self.assertEqual(energies, expect)

    def testDielectric(self):
        pot = self._setup(Coulomb(SYMBOLS, dielectric=4.0))
        energies = pot(self.atomic_nums, self.coords, scalars=self.qs)
        expect = torch.tensor([-0.25 / self.dist / 4.0], device=self.device)
        self.assertEqual(energies, expect)

    def testErfDamp(self):
        pot = self._setup(Coulomb(SYMBOLS, damp_fn="erf"))
        energies = pot(self.atomic_nums, self.coords, scalars=self.qs)
        damp = math.erf(0.5 * self.dist)
        expect = torch.tensor([-0.25 / self.dist * damp], device=self.device)
        self.assertEqual(energies, expect)

    def testTanhDamp(self):
        pot = self._setup(Coulomb(SYMBOLS, damp_fn="tanh"))
        energies = pot(self.atomic_nums, self.coords, scalars=self.qs)
        damp = math.tanh(0.5 * self.dist)
        expect = torch.tensor([-0.25 / self.dist * damp], device=self.device)
        self.assertEqual(energies, expect)

    def testAtomic(self):
        pot = self._setup(Coulomb(SYMBOLS))
        energies = pot(self.atomic_nums, self.coords, scalars=self.qs, atomic=True)
        half = -0.125 / self.dist
        expect = torch.tensor([[half, half]], device=self.device)
        self.assertEqual(energies, expect)

    def testBadDampFn(self):
        with self.assertRaises(ValueError):
            Coulomb(SYMBOLS, damp_fn="gauss")

    def testMatchesFixedCoulomb(self):
        molecs = make_molecs(5, 8, seed=1234, device=self.device)
        charge_table = torch.zeros(9, device=self.device)
        charge_table[[1, 6, 7, 8]] = torch.tensor(ELEM_CHARGES, device=self.device)
        qs = charge_table[molecs.atomic_nums]

        pot = self._setup(Coulomb(SYMBOLS))
        fixed_pot = self._setup(FixedCoulomb(SYMBOLS, charges=ELEM_CHARGES))
        energies = pot(molecs.atomic_nums, molecs.coords, scalars=qs)
        expect = fixed_pot(molecs.atomic_nums, molecs.coords)
        self.assertEqual(energies, expect)


@expand()
class TestANIqCoulomb(ANITestCase):
    def setUp(self):
        self.molecs = make_molecs(3, 6, seed=1234, device=self.device)

    def _model(self, coulomb: tp.Any):
        return simple_aniq(
            SYMBOLS,
            "wb97x-631gd",
            repulsion=False,
            strategy="pyaev",
            coulomb=coulomb,
        ).to(self.device)

    def testBadOptions(self):
        with self.assertRaises(ValueError):
            self._model("erff")
        with self.assertRaises(ValueError):
            simple_aniq(SYMBOLS, "wb97x-631gd", coulomb="full", coulomb_cutoff=12.0)

    def testFullInfiniteCutoff(self):
        model = simple_aniq(
            SYMBOLS, "wb97x-631gd", coulomb="full", coulomb_cutoff=math.inf
        )
        self.assertEqual(model.potentials["coulomb"].cutoff, math.inf)

    def testCoulombIsLast(self):
        for coulomb in ("full", "erf", "tanh"):
            model = self._model(coulomb)
            self.assertEqual(list(model.potentials.keys())[-1], "coulomb")

    def testEnergiesIncludeCoulomb(self):
        for coulomb in ("full", "erf", "tanh"):
            model = self._model(coulomb)
            pot = self._setup(model.potentials["coulomb"])
            species_coords = (self.molecs.atomic_nums, self.molecs.coords)

            _, energies, qs = self._setup(model)(species_coords)
            model.set_enabled("coulomb", False)
            _, no_coulomb_energies, no_coulomb_qs = self._setup(model)(species_coords)

            coulomb_energies = pot(
                self.molecs.atomic_nums, self.molecs.coords, scalars=qs
            )
            self.assertEqual(qs, no_coulomb_qs)
            self.assertEqual(energies, no_coulomb_energies + coulomb_energies)

    def testEnsembleValues(self):
        model = self._setup(self._model("full"))
        species_coords = (self.molecs.atomic_nums, self.molecs.coords)
        _, energies, qs = model(species_coords)
        _, ensemble_energies, ensemble_qs = model(species_coords, ensemble_values=True)
        self.assertEqual(ensemble_qs.shape, self.molecs.atomic_nums.shape)
        self.assertEqual(ensemble_qs, qs)
        self.assertEqual(ensemble_energies.mean(0), energies)


if __name__ == "__main__":
    unittest.main()
