import math
import unittest

import torch

from torchani import units
from torchani.utils import SYMBOLS_1X
from torchani.aev import AEVComputer
from torchani.potentials import TwoBodyDispersionD3
from torchani.potentials.dftd3 import _load_c6_constants
from torchani._testing import ANITestCase, expand
from torchani.grad import energies_and_forces


@expand()
class TestDispersion(ANITestCase):
    def setUp(self):
        # Use the exact same conversion factors as the original DFTD3 code
        self.old_angstrom_to_bohr = units.ANGSTROM_TO_BOHR
        self.old_hartree_to_kcalpermol = units.HARTREE_TO_KCALPERMOL
        units.ANGSTROM_TO_BOHR = 1 / 0.52917726
        units.HARTREE_TO_KCALPERMOL = 627.509541

        self.aev_computer = self._setup(AEVComputer.like_1x().double())
        # fully symmetric methane
        self.coordinates = torch.tensor(
            [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [0.5, 0.5, 0.5]],
            dtype=torch.double,
            device=self.device,
        ).unsqueeze(0)
        self.species = torch.tensor(
            [[0, 0, 0, 0, 1]], dtype=torch.long, device=self.device
        )
        self.atomic_numbers = torch.tensor(
            [[1, 1, 1, 1, 6]], dtype=torch.long, device=self.device
        )

    def tearDown(self):
        # Reset conversion factors
        units.ANGSTROM_TO_BOHR = self.old_angstrom_to_bohr
        units.HARTREE_TO_KCALPERMOL = self.old_hartree_to_kcalpermol

    def testConstructor(self):
        disp = self._setup(
            TwoBodyDispersionD3.from_functional(functional="wB97X", symbols=SYMBOLS_1X)
        )
        self.assertTrue(disp._s6 == torch.tensor(1.0, device=self.device))

    def testMethaneCoordinationNums(self):
        disp = self._setup(
            TwoBodyDispersionD3.from_functional(functional="wB97X", symbols=SYMBOLS_1X)
        )
        neighbors = self.aev_computer.neighborlist(
            disp.cutoff, self.species, self.coordinates
        )

        distances = neighbors.distances
        atom_index12 = neighbors.indices

        distances = units.angstrom2bohr(distances)
        coordnums = disp._coordnums(
            self.coordinates.shape[0],
            self.coordinates.shape[1],
            self.species.flatten()[atom_index12],
            atom_index12,
            distances,
        )
        # coordination numbers taken directly from DFTD3 Grimme et. al. code
        self.assertEqual(
            coordnums,
            torch.tensor(
                [1.0052222, 1.0052222, 1.0052222, 1.0052222, 3.999873048],
                device=self.device,
            ).double(),
        )

    def testPrecomputedC6(self):
        disp = self._setup(
            TwoBodyDispersionD3.from_functional(functional="wB97X", symbols=SYMBOLS_1X)
        )
        diags = {
            "H": torch.diag(disp.precalc_coeff6[0, 0]),
            "C": torch.diag(disp.precalc_coeff6[1, 1]),
            "O": torch.diag(disp.precalc_coeff6[2, 2]),
            "N": torch.diag(disp.precalc_coeff6[3, 3]),
        }
        # values from DFTD3 Grimme et. al. code
        expect_diags = {
            "H": torch.tensor(
                [
                    3.0267000198,
                    7.5915999413,
                    -1.0000000000,
                    -1.0000000000,
                    -1.0000000000,
                ],
                device=self.device,
            ),
            "C": torch.tensor(
                [
                    49.1129989624,
                    43.2452011108,
                    29.3602008820,
                    25.7808990479,
                    18.2066993713,
                ],
                device=self.device,
            ),
            "O": torch.tensor(
                [
                    25.2684993744,
                    22.1240997314,
                    19.6767997742,
                    15.5817003250,
                    -1.0000000000,
                ],
                device=self.device,
            ),
            "N": torch.tensor(
                [
                    15.5059003830,
                    12.8161001205,
                    10.3708000183,
                    -1.0000000000,
                    -1.0000000000,
                ],
                device=self.device,
            ),
        }
        for k in diags.keys():
            self.assertEqual(diags[k], expect_diags[k])

    def testAllCarbonConstants(self):
        c6_constants, _, _ = _load_c6_constants()
        c6_constants = c6_constants.to(self.device)
        expect_c6_carbon = torch.tensor(
            [
                [49.1130, 46.0681, 37.8419, 35.4129, 29.2830],
                [46.0681, 43.2452, 35.5219, 33.2540, 27.5206],
                [37.8419, 35.5219, 29.3602, 27.5063, 22.9517],
                [35.4129, 33.2540, 27.5063, 25.7809, 21.5377],
                [29.2830, 27.5206, 22.9517, 21.5377, 18.2067],
            ],
            device=self.device,
        )
        self.assertEqual(expect_c6_carbon, c6_constants[6, 6])

    def testMethaneC6(self):
        disp = self._setup(
            TwoBodyDispersionD3.from_functional(functional="wB97X", symbols=SYMBOLS_1X)
        )
        neighbors = self.aev_computer.neighborlist(
            disp.cutoff, self.species, self.coordinates
        )

        distances = neighbors.distances
        atom_index12 = neighbors.indices

        distances = units.angstrom2bohr(distances)
        species12 = self.species.flatten()[atom_index12]
        coordnums = disp._coordnums(
            self.coordinates.shape[0],
            self.coordinates.shape[1],
            species12,
            atom_index12,
            distances,
        )
        order6_coeffs = disp._interpolate_coeff6(species12, coordnums, atom_index12)
        # C6 coefficients taken directly from DFTD3 Grimme et. al. code
        expect_order6 = torch.tensor(
            [
                3.0882003,
                3.0882003,
                3.0882003,
                7.4632792,
                3.0882003,
                3.0882003,
                7.4632792,
                3.0882003,
                7.4632792,
                7.4632792,
            ],
            device=self.device,
            dtype=torch.double,
        )
        self.assertEqual(order6_coeffs, expect_order6)

    def testMethaneEnergy(self):
        disp = self._setup(
            TwoBodyDispersionD3.from_functional(functional="wB97X", symbols=SYMBOLS_1X)
        )
        neighbors = self.aev_computer.neighborlist(
            disp.cutoff, self.species, self.coordinates
        )
        energy = disp.compute_from_neighbors(
            self.species, self.coordinates, neighbors
        ).energies
        energy = units.hartree2kcalpermol(energy)
        self.assertEqual(
            energy,
            torch.tensor([-1.251336], device=self.device, dtype=torch.double),
            rtol=1e-6,
            atol=1e-6,
        )

    def testMethaneStandalone(self):
        if self.jit:
            self.skipTest("calc is non-jittable")
        disp = self._setup(
            TwoBodyDispersionD3.from_functional(
                SYMBOLS_1X,
                functional="wB97X",
                cutoff=8.0,
                cutoff_fn="dummy",
            )
        )
        energy = disp(self.atomic_numbers, self.coordinates)
        energy = units.hartree2kcalpermol(energy)
        self.assertEqual(
            energy,
            torch.tensor([-1.251336], device=self.device, dtype=torch.double),
            rtol=1e-6,
            atol=1e-6,
        )

    def testMethaneStandaloneBatch(self):
        if self.jit:
            self.skipTest("calc is non-jittable")
        disp = self._setup(
            TwoBodyDispersionD3.from_functional(
                SYMBOLS_1X,
                functional="wB97X",
                cutoff=8.0,
                cutoff_fn="dummy",
            )
        )
        r = 2
        coordinates = self.coordinates.repeat(r, 1, 1)
        species = self.atomic_numbers.repeat(r, 1)
        energy = disp(species, coordinates)
        energy = units.hartree2kcalpermol(energy)
        self.assertEqual(
            energy,
            torch.tensor(
                [-1.251336, -1.251336], dtype=torch.double, device=self.device
            ),
            rtol=1e-6,
            atol=1e-6,
        )

    def testDispersionBatches(self):
        if self.jit:
            self.skipTest("calc is non-jittable")
        disp = self._setup(
            TwoBodyDispersionD3.from_functional(
                symbols=SYMBOLS_1X,
                cutoff=8.0,
                functional="wB97X",
            )
        )
        coordinates1 = torch.tensor(
            [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]],
            device=self.device,
        ).unsqueeze(0)
        coordinates2 = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.5, 0.0, 0.0]],
            device=self.device,
        ).unsqueeze(0)
        coordinates3 = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [3.5, 0.0, 0.0]],
            device=self.device,
        ).unsqueeze(0)
        species1 = torch.tensor([[1, 6, 7]], device=self.device)
        species2 = torch.tensor([[-1, 1, 6]], device=self.device)
        species3 = torch.tensor([[-1, 1, 1]], device=self.device)
        coordinates_cat = torch.cat((coordinates1, coordinates2, coordinates3), dim=0)
        species_cat = torch.cat((species1, species2, species3), dim=0)

        energy1 = disp(species1, coordinates1)
        # avoid first atom since it isdummy
        energy2 = disp(species2[:, 1:], coordinates2[:, 1:, :])
        energy3 = disp(species3[:, 1:], coordinates3[:, 1:, :])
        energies_cat = torch.cat((energy1, energy2, energy3))
        energies = disp(species_cat, coordinates_cat)
        self.assertEqual(energies, energies_cat)

    def testForce(self):
        if self.jit:
            self.skipTest("calc is non-jittable")
        disp = self._setup(
            TwoBodyDispersionD3.from_functional(
                symbols=SYMBOLS_1X,
                cutoff=math.inf,
                functional="wB97X",
            )
        )
        energies, forces = energies_and_forces(
            disp, self.atomic_numbers, self.coordinates
        )
        grad = -forces / units.ANGSTROM_TO_BOHR
        self.coordinates.requires_grad_(False)
        # compare with analytical gradient from Grimme's DFTD3 (DFTD3 gives
        # gradient in Bohr)
        expect_grad = torch.tensor(
            [
                [-0.42656701194940e-05, -0.42656701194940e-05, -0.42656701194940e-05],
                [-0.42656701194940e-05, 0.42656701194940e-05, 0.42656701194940e-05],
                [0.42656701194940e-05, -0.42656701194940e-05, 0.42656701194940e-05],
                [0.42656701194940e-05, 0.42656701194940e-05, -0.42656701194940e-05],
                [0.00000000000000e00, 0.00000000000000e00, 0.00000000000000e00],
            ],
            device=self.device,
            dtype=torch.double,
        )
        self.assertEqual(expect_grad.unsqueeze(0), grad)


if __name__ == "__main__":
    unittest.main(verbosity=2)
