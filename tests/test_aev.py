import typing as tp
from pathlib import Path
import unittest
import os
import pickle
import itertools
import traceback

import torch

from torchani._testing import TestCase
from torchani.neighbors import AllPairs, compute_bounding_cell
from torchani.nn import SpeciesConverter
from torchani.utils import ChemicalSymbolsToInts, pad_atomic_properties, map_to_central
from torchani.aev import AEVComputer, ANIAngular, ANIRadial
from torchani.io import read_xyz
from torchani.neighbors import CellList


path = os.path.dirname(os.path.realpath(__file__))
N = 97


class _TestAEVBase(TestCase):
    def setUp(self):
        self.aev_computer = AEVComputer.like_1x()
        self.radial_len = self.aev_computer.radial_len
        self._debug_aev = False

    def assertAEVEqual(self, expected_radial, expected_angular, aev):
        radial = aev[..., : self.radial_len]
        angular = aev[..., self.radial_len:]
        if self._debug_aev:
            aid = 1
            print(torch.stack([expected_radial[0, aid, :], radial[0, aid, :]]))
        self.assertEqual(expected_radial, radial)
        self.assertEqual(expected_angular, angular)


class TestAEVConstructor(TestCase):
    def testTerms2x(self):
        exact_angular = ANIAngular.like_2x()
        exact_radial = ANIRadial.like_2x()
        computer = AEVComputer(exact_radial, exact_angular, num_species=7)
        computer_alt = AEVComputer.like_2x()
        self._compare_constants(computer, computer_alt)

    def testTerms1x(self):
        exact_angular = ANIAngular.like_1x()
        exact_radial = ANIRadial.like_1x()
        computer = AEVComputer(exact_radial, exact_angular, num_species=4)
        computer_alt = AEVComputer.like_1x()
        self._compare_constants(computer, computer_alt)

    def _compare_constants(self, aev_computer, aev_computer_alt, rtol=1e-7, atol=1e-7):
        alt_state_dict = aev_computer_alt.state_dict()
        for k, v in aev_computer.state_dict().items():
            self.assertEqual(alt_state_dict[k], v, rtol=rtol, atol=atol)


class TestIsolated(TestCase):
    # Tests that there is no error when atoms are separated
    # a distance greater than the cutoff radius from all other atoms
    # this can throw an IndexError for large distances or lone atoms
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.aev_computer = AEVComputer.like_1x().to(self.device)
        self.symbols_to_idxs = ChemicalSymbolsToInts(["H", "C", "N", "O"])
        self.rcr = self.aev_computer.radial.cutoff
        self.rca = self.aev_computer.angular.cutoff

    def testCO2(self):
        species = self.symbols_to_idxs(["O", "C", "O"]).to(self.device).unsqueeze(0)
        distances = [
            1.0,
            self.rca,
            self.rca + 1e-4,
            self.rcr,
            self.rcr + 1e-4,
            2 * self.rcr,
        ]
        for dist in distances:
            coords = torch.tensor(
                [[[-dist, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, dist]]],
                requires_grad=True,
                device=self.device,
            )
            try:
                _ = self.aev_computer(species, coords)
            except IndexError:
                self.fail(
                    f"\n\n{traceback.format_exc()}\nFailure at distance: {dist}\n"
                    f"Radial r_cut of aev_computer: {self.rcr}\n"
                    f"Angular r_cut of aev_computer: {self.rca}"
                )

    def testH2(self):
        species = self.symbols_to_idxs(["H", "H"]).to(self.device).unsqueeze(0)
        distances = [
            1.0,
            self.rca,
            self.rca + 1e-4,
            self.rcr,
            self.rcr + 1e-4,
            2 * self.rcr,
        ]
        for dist in distances:
            coords = torch.tensor(
                [[[0.0, 0.0, 0.0], [0.0, 0.0, dist]]],
                requires_grad=True,
                device=self.device,
            )
            try:
                _ = self.aev_computer(species, coords)
            except IndexError:
                self.fail(
                    f"\n\n{traceback.format_exc()}\nFailure at distance: {dist}\n"
                    f"Radial r_cut of aev_computer: {self.rcr}\n"
                    f"Angular r_cut of aev_computer: {self.rca}"
                )

    def testH(self):
        # Tests for failure on a single atom
        species = self.symbols_to_idxs(["H"]).to(self.device).unsqueeze(0)
        coords = torch.tensor(
            [[[0.0, 0.0, 0.0]]], requires_grad=True, device=self.device
        )
        try:
            _ = self.aev_computer(species, coords)
        except IndexError:
            self.fail(f"\n\n{traceback.format_exc()}\nFailure on lone atom\n")


class TestAEV(_TestAEVBase):
    def testGradsBatches(self):
        # test if gradients are the same for single molecules and for batches
        # with dummy atoms
        N = 25
        assert N % 5 == 0, "N must be a multiple of 5"
        coords_list = []
        species_list = []
        grads_expect_list = []
        for j in range(N):
            c = torch.randn((1, 3, 3), dtype=torch.float, requires_grad=True)
            s = torch.randint(low=0, high=4, size=(1, 3), dtype=torch.long)
            if j % 5 == 0:
                s[0, 0] = -1
            aev = self.aev_computer(s, c)
            aev.backward(torch.ones_like(aev))
            if c.grad is None:
                self.fail("Got a None gradient")
            grads_expect_list.append(c.grad)
            coords_list.append(c)
            species_list.append(s)

        coords_cat = torch.cat(coords_list, dim=0).detach()
        coords_cat.requires_grad_(True)
        species_cat = torch.cat(species_list, dim=0)
        grads_expect = torch.cat(grads_expect_list, dim=0)

        aev = self.aev_computer(species_cat, coords_cat)
        aev.backward(torch.ones_like(aev))
        self.assertEqual(grads_expect, coords_cat.grad)

    def testIsomers(self):
        for i in range(N):
            datafile = os.path.join(path, f"resources/ANI1_subset/{i}")
            with open(datafile, "rb") as f:
                (
                    coords,
                    species,
                    expected_radial,
                    expected_angular,
                    _,
                    _,
                ) = pickle.load(f)
                coords = torch.from_numpy(coords)
                species = torch.from_numpy(species)
                expected_radial = torch.from_numpy(expected_radial)
                expected_angular = torch.from_numpy(expected_angular)
                aev = self.aev_computer(species, coords)
                self.assertAEVEqual(expected_radial, expected_angular, aev)

    def testNoNan(self):
        # AEV should not output NaN even when coords are superimposed
        coords = torch.ones(1, 3, 3, dtype=torch.float)
        species = torch.zeros(1, 3, dtype=torch.long)
        aev = self.aev_computer(species, coords)
        self.assertFalse(torch.isnan(aev).any())

    def testBoundingCell(self):
        datafile = os.path.join(path, "resources/ANI1_subset/10")
        with open(datafile, "rb") as f:
            coords, species, _, _, _, _ = pickle.load(f)
            coords = torch.from_numpy(coords)
            species = torch.from_numpy(species)

        coords, cell = compute_bounding_cell(coords, 1e-5)
        self.assertTrue((coords > 0.0).all())
        self.assertTrue((coords < torch.norm(cell, dim=1)).all())

    def testPadding(self):
        species_coords = []
        radial_angular = []
        for i in range(N):
            datafile = os.path.join(path, f"resources/ANI1_subset/{i}")
            with open(datafile, "rb") as f:
                coords, species, radial, angular, _, _ = pickle.load(f)
                coords = torch.from_numpy(coords)
                species = torch.from_numpy(species)
                radial = torch.from_numpy(radial)
                angular = torch.from_numpy(angular)
                species_coords.append({"species": species, "coords": coords})
                radial_angular.append((radial, angular))
        species_coords_dict = pad_atomic_properties(species_coords)
        aev = self.aev_computer(
            species_coords_dict["species"], species_coords_dict["coords"]
        )
        start = 0
        for expected_radial, expected_angular in radial_angular:
            conformations = expected_radial.shape[0]
            atoms = expected_radial.shape[1]
            aev_ = aev[start:(start + conformations), 0:atoms]
            start += conformations
            self.assertAEVEqual(expected_radial, expected_angular, aev_)


class TestAEVJIT(TestAEV):
    def setUp(self):
        super().setUp()
        self.aev_computer = tp.cast(AEVComputer, torch.jit.script(self.aev_computer))


class TestPBCSeeEachOther(TestCase):
    def setUp(self):
        self.aev_computer = AEVComputer.like_1x().to(torch.double)
        self.neighborlist = AllPairs()

    def testTranslationalInvariancePBC(self):
        coords = torch.tensor(
            [[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1]]],
            dtype=torch.double,
            requires_grad=True,
        )
        cell = torch.eye(3, dtype=torch.double) * 2
        species = torch.tensor([[1, 0, 0, 0, 0]], dtype=torch.long)
        pbc = torch.ones(3, dtype=torch.bool)

        aev = self.aev_computer(species, coords, cell=cell, pbc=pbc)

        for _ in range(100):
            translation = torch.randn(3, dtype=torch.double)
            aev2 = self.aev_computer(species, coords + translation, cell=cell, pbc=pbc)
            self.assertEqual(aev, aev2)

    def testPBCConnersSeeEachOther(self):
        species = torch.tensor([[0, 0]])
        cell = torch.eye(3, dtype=torch.double) * 10
        pbc = torch.ones(3, dtype=torch.bool)

        xyz1 = torch.tensor([0.1, 0.1, 0.1])
        xyz2s = [
            torch.tensor([9.9, 0.0, 0.0]),
            torch.tensor([0.0, 9.9, 0.0]),
            torch.tensor([0.0, 0.0, 9.9]),
            torch.tensor([9.9, 9.9, 0.0]),
            torch.tensor([0.0, 9.9, 9.9]),
            torch.tensor([9.9, 0.0, 9.9]),
            torch.tensor([9.9, 9.9, 9.9]),
        ]

        for xyz2 in xyz2s:
            coords = torch.stack([xyz1, xyz2]).to(torch.double).unsqueeze(0)
            atom_index12, _, _ = self.neighborlist(1.0, species, coords, cell, pbc)
            atom_index1, atom_index2 = atom_index12.unbind(0)
            self.assertEqual(atom_index1.tolist(), [0])
            self.assertEqual(atom_index2.tolist(), [1])

    def testPBCSurfaceSeeEachOther(self):
        cell = torch.eye(3, dtype=torch.double) * 10
        pbc = torch.ones(3, dtype=torch.bool)
        species = torch.tensor([[0, 0]])

        for i in range(3):
            xyz1 = torch.tensor([5.0, 5.0, 5.0], dtype=torch.double)
            xyz1[i] = 0.1
            xyz2 = xyz1.clone()
            xyz2[i] = 9.9

            coords = torch.stack([xyz1, xyz2]).unsqueeze(0)
            atom_index12, _, _ = self.neighborlist(1.0, species, coords, cell, pbc)
            atom_index1, atom_index2 = atom_index12.unbind(0)
            self.assertEqual(atom_index1.tolist(), [0])
            self.assertEqual(atom_index2.tolist(), [1])

    def testPBCEdgesSeeEachOther(self):
        cell = torch.eye(3, dtype=torch.double) * 10
        pbc = torch.ones(3, dtype=torch.bool)
        species = torch.tensor([[0, 0]])

        for i, j in itertools.combinations(range(3), 2):
            xyz1 = torch.tensor([5.0, 5.0, 5.0], dtype=torch.double)
            xyz1[i] = 0.1
            xyz1[j] = 0.1
            for new_i, new_j in [[0.1, 9.9], [9.9, 0.1], [9.9, 9.9]]:
                xyz2 = xyz1.clone()
                xyz2[i] = new_i
                xyz2[j] = new_j

            coords = torch.stack([xyz1, xyz2]).unsqueeze(0)
            atom_index12, _, _ = self.neighborlist(1.0, species, coords, cell, pbc)
            atom_index1, atom_index2 = atom_index12.unbind(0)
            self.assertEqual(atom_index1.tolist(), [0])
            self.assertEqual(atom_index2.tolist(), [1])

    def testNonRectangularPBCConnersSeeEachOther(self):
        species = torch.tensor([[0, 0]])
        cell = torch.tensor(
            [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [10.0, 0.0, 10.0]],
            dtype=torch.double,
        )

        pbc = torch.ones(3, dtype=torch.bool)

        xyz1 = torch.tensor([0.1, 0.1, 0.05], dtype=torch.double)
        xyz2 = torch.tensor([10.0, 0.1, 0.1], dtype=torch.double)

        coords = torch.stack([xyz1, xyz2]).unsqueeze(0)
        atom_index12, _, _ = self.neighborlist(1.0, species, coords, cell, pbc)
        atom_index1, atom_index2 = atom_index12.unbind(0)
        self.assertEqual(atom_index1.tolist(), [0])
        self.assertEqual(atom_index2.tolist(), [1])


class TestAEVOnBoundary(TestCase):
    def setUp(self):
        self.eps = 1e-9
        self.cell = torch.tensor(
            [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [100.0, 0.0, 100.0]],
            dtype=torch.double,
        )
        self.inv_cell = torch.inverse(self.cell)
        self.coords = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [1.0, -0.1, -0.1],
                    [-0.1, 1.0, -0.1],
                    [-0.1, -0.1, 1.0],
                    [-1.0, -1.0, -1.0],
                ]
            ],
            dtype=torch.double,
        )
        self.species = torch.tensor([[1, 0, 0, 0, 0]])
        self.pbc = torch.ones(3, dtype=torch.bool)
        self.v1, self.v2, self.v3 = self.cell
        self.center_coords = self.coords + 0.5 * (self.v1 + self.v2 + self.v3)
        self.aev_computer = AEVComputer.like_1x().to(torch.double)

        self.aev = self.aev_computer(
            self.species, self.center_coords, cell=self.cell, pbc=self.pbc
        )

    def assertInCell(self, coords):
        coords_cell = coords @ self.inv_cell
        self.assertEqual(coords, coords_cell @ self.cell)
        in_cell = (coords_cell >= -self.eps) & (coords_cell <= 1 + self.eps)
        self.assertTrue(in_cell.all())

    def assertNotInCell(self, coords):
        coords_cell = coords @ self.inv_cell
        self.assertEqual(coords, coords_cell @ self.cell)
        in_cell = (coords_cell >= -self.eps) & (coords_cell <= 1 + self.eps)
        self.assertFalse(in_cell.all())

    def testCornerSurfaceAndEdge(self):
        for i, j, k in itertools.product([0, 0.5, 1], repeat=3):
            if i == 0.5 and j == 0.5 and k == 0.5:
                continue
            coords = self.coords + i * self.v1 + j * self.v2 + k * self.v3
            self.assertNotInCell(coords)
            coords = map_to_central(coords, self.cell, self.pbc)
            self.assertInCell(coords)
            aev = self.aev_computer(self.species, coords, self.cell, self.pbc)
            self.assertGreater(aev.abs().max().item(), 0)
            self.assertEqual(aev, self.aev)


class TestAEVOnBenzenePBC(TestCase):
    def setUp(self):
        self.aev_computer = AEVComputer.like_1x()
        self.species, self.coords, cell, pbc = read_xyz(
            (Path(__file__).parent / "resources") / "benzene.xyz"
        )
        assert cell is not None
        assert pbc is not None
        self.cell = cell
        self.pbc = pbc
        self.species = SpeciesConverter(["H", "C", "N", "O"])(self.species)
        self.aev = self.aev_computer(self.species, self.coords, self.cell, self.pbc)
        self.natoms = self.aev.shape[1]

    def testRepeat(self):
        c1, c2, c3 = self.cell
        species2 = self.species.repeat(1, 4)
        coords2 = torch.cat(
            [
                self.coords,
                self.coords + c1,
                self.coords + 2 * c1,
                self.coords + 3 * c1,
            ],
            dim=1,
        )
        cell2 = torch.stack([4 * c1, c2, c3])
        aev2 = self.aev_computer(species2, coords2, cell=cell2, pbc=self.pbc)
        for i in range(3):
            _start = i * self.natoms
            _end = (i + 1) * self.natoms
            aev3 = aev2[:, _start:_end, :]
            self.assertEqual(self.aev, aev3)

    def testManualMirror(self):
        c1, c2, c3 = self.cell
        species2 = self.species.repeat(1, 3**3)
        coords2 = torch.cat(
            [
                self.coords + i * c1 + j * c2 + k * c3
                for i, j, k in itertools.product([0, -1, 1], repeat=3)
            ],
            dim=1,
        )
        aev2 = self.aev_computer(species2, coords2)
        aev2 = aev2[:, : self.natoms, :]
        self.assertEqual(self.aev, aev2)


class TestAEVNIST(_TestAEVBase):
    def testNIST(self):
        datafile = os.path.join(path, "resources/NIST/all")
        with open(datafile, "rb") as f:
            data = pickle.load(f)
            # only use first 100 data points to make test take an
            # acceptable time
            for coords, species, radial, angular, _, _ in data[:100]:
                coords = torch.from_numpy(coords).to(torch.float)
                species = torch.from_numpy(species)
                radial = torch.from_numpy(radial).to(torch.float)
                angular = torch.from_numpy(angular).to(torch.float)
                aev = self.aev_computer(species, coords)
                self.assertAEVEqual(radial, angular, aev)


class TestDynamicsAEV(_TestAEVBase):
    def testBenzene(self):
        for i in [2, 8]:
            datafile = os.path.join(path, f"resources/benzene-md/{i}.dat")
            with open(datafile, "rb") as f:
                (
                    coords,
                    species,
                    expected_radial,
                    expected_angular,
                    _,
                    _,
                    cell,
                    pbc,
                ) = pickle.load(f)
                coords = torch.from_numpy(coords).float().unsqueeze(0)
                species = torch.from_numpy(species).unsqueeze(0)
                expected_radial = torch.from_numpy(expected_radial).float().unsqueeze(0)
                expected_angular = (
                    torch.from_numpy(expected_angular).float().unsqueeze(0)
                )
                cell = torch.from_numpy(cell).float()
                pbc = torch.from_numpy(pbc).bool()
                aev = self.aev_computer(species, coords, cell=cell, pbc=pbc)
                self.assertAEVEqual(expected_radial, expected_angular, aev)

    def testBenzeneCellList(self):
        for i in [2, 8]:
            datafile = os.path.join(path, f"resources/benzene-md/{i}.dat")
            self.aev_computer.neighborlist = CellList()
            with open(datafile, "rb") as f:
                (
                    coords,
                    species,
                    expected_radial,
                    expected_angular,
                    _,
                    _,
                    cell,
                    pbc,
                ) = pickle.load(f)
                coords = torch.from_numpy(coords).float().unsqueeze(0)
                species = torch.from_numpy(species).unsqueeze(0)
                expected_radial = torch.from_numpy(expected_radial).float().unsqueeze(0)
                expected_angular = (
                    torch.from_numpy(expected_angular).float().unsqueeze(0)
                )
                cell = torch.from_numpy(cell).float()
                pbc = torch.from_numpy(pbc).bool()
                aev = self.aev_computer(species, coords, cell=cell, pbc=pbc)
                self.assertAEVEqual(expected_radial, expected_angular, aev)

    def testTripeptide(self):
        for i in range(100):
            datafile = os.path.join(path, f"resources/tripeptide-md/{i}.dat")
            with open(datafile, "rb") as f:
                (
                    coords,
                    species,
                    expected_radial,
                    expected_angular,
                    _,
                    _,
                    _,
                    _,
                ) = pickle.load(f)
                coords = torch.from_numpy(coords).float().unsqueeze(0)
                species = torch.from_numpy(species).unsqueeze(0)
                expected_radial = torch.from_numpy(expected_radial).float().unsqueeze(0)
                expected_angular = (
                    torch.from_numpy(expected_angular).float().unsqueeze(0)
                )
                aev = self.aev_computer(species, coords)
                self.assertAEVEqual(expected_radial, expected_angular, aev)


if __name__ == "__main__":
    unittest.main(verbosity=2)
