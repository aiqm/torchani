import pickle
from pathlib import Path
import torch
import typing as tp
import unittest

from torchani._testing import make_molecs
from torchani._testing import ANITestCase, expand
from torchani.potentials import (
    RepulsionZBL,
    RepulsionLJ,
    DispersionLJ,
    LennardJones,
    FixedCoulomb,
    FixedMNOK,
    BasePairPotential,
)
from torchani.utils import SYMBOLS_2X

this_dir = Path(__file__).parent


@expand()
class TestAcceptEnergies(ANITestCase):
    def setUp(self):
        self.sym = SYMBOLS_2X

    def testZBL(self):
        self._testPotential("zbl", self._setup(RepulsionZBL(self.sym)))

    def testLJ(self):
        self._testPotential("lj", self._setup(LennardJones(self.sym)))
        self._testPotential("rep-lj", self._setup(RepulsionLJ(self.sym)))
        self._testPotential("disp-lj", self._setup(DispersionLJ(self.sym)))

    def testFixedCoulomb(self):
        self._testPotential(
            "fixed-coulomb",
            self._setup(FixedCoulomb(self.sym, charges=[0.1] * len(self.sym))),
        )

    def testFixedMNOK(self):
        self._testPotential(
            "fixed-mnok",
            self._setup(
                FixedMNOK(
                    self.sym, charges=[0.1] * len(self.sym), eta=[0.01] * len(self.sym)
                )
            ),
        )

    def _testPotential(self, name: str, pot: BasePairPotential):
        device = tp.cast(tp.Literal["cpu", "cuda"], self.device.type)
        molec = make_molecs(10, 10, seed=1234, device=device)
        energies = pot(molec.atomic_nums, molec.coords)
        with open(
            Path(this_dir, "resources", "potentials", f"{name}-energies.pkl"), mode="rb"
        ) as f:
            expect_energies = pickle.load(f)
        self.assertEqual(energies, torch.tensor(expect_energies, device=self.device))


# TODO: Not sure why subclassing makes unittest not detect the test here
@expand()
class TestAcceptForces(ANITestCase):
    def _testPotential(self, name: str, pot: BasePairPotential):
        device = tp.cast(tp.Literal["cpu", "cuda"], self.device.type)
        molec = make_molecs(10, 10, seed=1234, symbols=self.sym, device=device)
        molec.coords.requires_grad_(True)
        energies = pot(
            molec.atomic_nums,
            molec.coords,
        )
        forces = -torch.autograd.grad(energies.sum(), molec.coords)[0]
        with open(
            Path(this_dir, "resources", "potentials", f"{name}-energies-forces.pkl"),
            mode="rb",
        ) as f:
            expect_energies, expect_forces = pickle.load(f)
        self.assertEqual(energies, torch.tensor(expect_energies, device=self.device))
        self.assertEqual(forces, torch.tensor(expect_forces, device=self.device))

    def setUp(self):
        self.sym = SYMBOLS_2X

    def testZBL(self):
        self._testPotential("zbl", self._setup(RepulsionZBL(self.sym)))

    def testLJ(self):
        self._testPotential("lj", self._setup(LennardJones(self.sym)))
        self._testPotential("rep-lj", self._setup(RepulsionLJ(self.sym)))
        self._testPotential("disp-lj", self._setup(DispersionLJ(self.sym)))

    def testFixedCoulomb(self):
        self._testPotential(
            "fixed-coulomb",
            self._setup(FixedCoulomb(self.sym, charges=[0.1] * len(self.sym))),
        )

    def testFixedMNOK(self):
        self._testPotential(
            "fixed-mnok",
            self._setup(
                FixedMNOK(
                    self.sym, charges=[0.1] * len(self.sym), eta=[0.01] * len(self.sym)
                )
            ),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
