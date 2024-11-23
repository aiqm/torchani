from pathlib import Path
import unittest

import torch
from torch import Tensor

import torchani
from torchani.io import read_xyz
from torchani._testing import TestCase, expand, ANITestCase
from torchani.aev import AEVComputer
from torchani.neighbors import (
    CellList,
    setup_grid,
    coords_to_fractional,
    coords_to_grid_idx3,
    flatten_idx3,
    image_pairs_within,
    count_atoms_in_buckets,
    _offset_idx3,
)


class TestCellList(TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.cutoff = 5.2
        self.cell_size = self.cutoff * 3 + 0.1
        # The length of the box is ~ (3 * 5.2 + 0.1) this so that
        # 3 buckets in each direction are needed to cover it, if one uses
        # a cutoff of 5.2 and a bucket length of 5.200001
        # first bucket is 0 - 5.2, in 3 directions, and subsequent buckets
        # are on top of that
        _, self.coordinates, cell, pbc = read_xyz(
            (Path(__file__).resolve().parent / "resources") / "tight_cell.xyz"
        )
        self.pbc = pbc
        self.cell = cell

    def testInit(self):
        self.assertTrue(_offset_idx3().shape == (13, 3))

    def testSetupGrid(self):
        assert self.cell is not None
        grid_shape = setup_grid(
            self.cell,
            self.cutoff,
        )
        # this creates a unit cell with 27 buckets
        # and a grid of 3 x 3 x 3 buckets (3 in each direction, GX == GY == GZ == 3)
        expect_shape = torch.tensor([3, 3, 3], dtype=torch.long, device=self.device)
        self.assertEqual(grid_shape, expect_shape)

    def testFractionalize(self):
        # Coordinate fractionalization
        assert self.cell is not None
        frac = coords_to_fractional(self.coordinates, self.cell)
        self.assertTrue(~torch.isnan(frac).any())
        self.assertTrue(~torch.isinf(frac).any())
        self.assertTrue((frac < 1.0).all())
        self.assertTrue((frac >= 0.0).all())

    def testGridIdx3(self):
        assert self.cell is not None
        grid_shape = setup_grid(
            self.cell,
            self.cutoff,
        )
        atom_grid_idx3 = coords_to_grid_idx3(self.coordinates, self.cell, grid_shape)
        self.assertTrue(atom_grid_idx3.shape == (1, 54, 3))
        self.assertEqual(atom_grid_idx3, atom_grid_idx3_expect)

    def testGridIdx(self):
        assert self.cell is not None
        grid_shape = setup_grid(
            self.cell,
            self.cutoff,
        )
        grid_idx = flatten_idx3(atom_grid_idx3_expect, grid_shape)
        # All flat grid indices are present in this test
        grid_idx_compare = torch.repeat_interleave(
            torch.arange(0, 27, dtype=torch.long), 2
        )
        self.assertEqual(grid_idx, grid_idx_compare.view(1, -1))

    def testCounts(self):
        assert self.cell is not None
        grid_numel_expect = 27
        grid_shape = setup_grid(
            self.cell,
            self.cutoff,
        )
        atom_grid_idx = flatten_idx3(atom_grid_idx3_expect, grid_shape)
        self.assertEqual(int(grid_shape.prod()), grid_numel_expect)
        grid_count, grid_cumcount = count_atoms_in_buckets(
            atom_grid_idx,
            grid_shape,
        )
        # these are all 2
        self.assertEqual(grid_count.shape, (grid_numel_expect,))
        self.assertTrue((grid_count == 2).all())
        # these are all 0 2 4 6 ...
        self.assertEqual(grid_cumcount, torch.arange(0, 54, 2))

    def testImagePairsWithinBuckets(self):
        assert self.cell is not None
        grid_shape = setup_grid(
            self.cell,
            self.cutoff,
        )
        atom_grid_idx3 = coords_to_grid_idx3(self.coordinates, self.cell, grid_shape)
        atom_grid_idx = flatten_idx3(atom_grid_idx3, grid_shape)
        grid_count, grid_cumcount = count_atoms_in_buckets(
            atom_grid_idx,
            grid_shape,
        )
        within = image_pairs_within(
            grid_count,
            grid_cumcount,
            int(grid_count.max()),
        )
        # some hand comparisons
        self.assertEqual(within.shape, (2, 27))
        self.assertEqual(within[0], torch.arange(1, 55, 2))
        self.assertEqual(within[1], torch.arange(0, 54, 2))

    def testCellListInit(self):
        AEVComputer.like_1x(neighborlist="cell_list")


@expand()
class TestCellListComparison(ANITestCase):
    def setUp(self):
        self.cutoff = 5.2
        self.cell_size = self.cutoff * 3 + 0.1
        # The length of the box is ~ (3 * 5.2 + 0.1) this so that
        # 3 buckets in each direction are needed to cover it, if one uses
        # a cutoff of 5.2 and a bucket length of 5.200001
        # first bucket is 0 - 5.2, in 3 directions, and subsequent buckets
        # are on top of that
        self.species, self.coordinates, cell, pbc = read_xyz(
            Path(Path(__file__).parent, "resources", "tight_cell.xyz"),
            device=self.device,
        )
        self.pbc = pbc
        self.cell = cell
        self.clist = self._setup(CellList())
        self.num_to_test = 10

    def _check_neighborlists_match(self, coords: Tensor):
        species = torch.ones(coords.shape[:-1], dtype=torch.long, device=self.device)
        aev_cl = self._setup(AEVComputer.like_1x(neighborlist="cell_list"))
        aev_fp = self._setup(AEVComputer.like_1x(neighborlist="all_pairs"))
        aevs_cl = aev_cl(species, coords, cell=self.cell, pbc=self.pbc)
        aevs_fp = aev_fp(species, coords, cell=self.cell, pbc=self.pbc)
        self.assertEqual(aevs_cl, aevs_fp)

    def testCellListMatchesAllPairs(self):
        cut = self.cutoff
        d = 0.5
        batch = [
            [
                [cut / 2 - d, cut / 2 - d, cut / 2 - d],
                [cut / 2 + 0.1, cut / 2 + 0.1, cut / 2 + 0.1],
                [cut / 2, cut / 2 + 2.4 * cut, cut / 2 + 2.4 * cut],
            ],
            [
                [cut / 2 - d, cut / 2 - d, cut / 2 - d],
                [cut / 2 + 0.1, cut / 2 + 0.1, cut / 2 + 0.1],
                [cut / 2 + 2.4 * cut, cut / 2 + 2.4 * cut, cut / 2 + 2.4 * cut],
            ],
            [
                [1.0000e-03, 6.5207e00, 1.0000e-03],
                [1.0000e-03, 1.5299e01, 1.3643e01],
                [1.0000e-03, 2.1652e00, 1.5299e01],
            ],
            [
                [1.5299e01, 1.0000e-03, 5.5613e00],
                [1.0000e-03, 1.0000e-03, 3.8310e00],
                [1.5299e01, 1.1295e01, 1.5299e01],
            ],
            [
                [1.0000e-03, 1.0000e-03, 1.0000e-03],
                [1.0389e01, 1.0000e-03, 1.5299e01],
            ],
        ]
        for coordinates in batch:
            self._check_neighborlists_match(
                torch.tensor(coordinates, dtype=torch.float, device=self.device).view(
                    1, -1, 3
                )
            )
        self._check_neighborlists_match(self.coordinates)

    def testCellListMatchesAllPairsRandomNoise(self):
        for j in range(self.num_to_test):
            noise = 0.1
            coordinates = self.coordinates + torch.empty(
                self.coordinates.shape, device=self.device
            ).uniform_(-noise, noise)
            self._check_neighborlists_match(coordinates)

    def testCellListMatchesAllPairsRandomNormal(self):
        for j in range(self.num_to_test):
            coordinates = (
                torch.randn((1, 10, 3), device=self.device, dtype=torch.float)
                * 3
                * self.cell_size
            )
            coordinates = torch.clamp(
                coordinates, min=0.0001, max=self.cell_size - 0.0001
            )
            self._check_neighborlists_match(coordinates)


@expand()
class TestCellListComparisonNoPBC(TestCellListComparison):
    def setUp(self):
        self.cutoff = 5.2
        self.cell_size = self.cutoff * 3 + 0.1
        # The length of the box is ~ (3 * 5.2 + 0.1) this so that
        # 3 buckets in each direction are needed to cover it, if one uses
        # a cutoff of 5.2 and a bucket length of 5.200001
        # first bucket is 0 - 5.2, in 3 directions, and subsequent buckets
        # are on top of that
        self.species, self.coordinates, _, _ = read_xyz(
            Path(Path(__file__).parent, "resources", "tight_cell.xyz"),
            device=self.device,
        )
        self.cell = None
        self.pbc = None
        self.clist = self._setup(CellList())
        self.num_to_test = 10


class TestCellListEnergies(TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        cut = 5.2
        self.cut = cut
        self.cell_size = cut * 3 + 0.1
        # The length of the box is ~ 3 * cutoff this so that
        # 3 buckets in each direction are needed to cover it, if one uses
        # a cutoff of 5.2 and a bucket length of 5.200001
        # first bucket is 0 - 5.2, in 3 directions, and subsequent buckets
        # are on top of that
        self.pbc = torch.tensor([True, True, True], dtype=torch.bool)
        self.cell = torch.diag(
            torch.tensor([self.cell_size, self.cell_size, self.cell_size])
        ).float()
        self.aev_cl = AEVComputer.like_1x(neighborlist="cell_list")
        self.aev_fp = AEVComputer.like_1x(neighborlist="all_pairs")
        self.model_cl = torchani.models.ANI1x(model_index=0, device=self.device)
        self.model_fp = torchani.models.ANI1x(model_index=0, device=self.device)
        self.model_cl.potentials["nnp"].aev_computer = self.aev_cl
        self.model_fp.potentials["nnp"].aev_computer = self.aev_fp
        self.num_to_test = 10

    def testCellListEnergiesRandom(self):
        self.model_cl = self.model_cl.to(self.device, torch.double)
        self.model_fp = self.model_fp.to(self.device, torch.double)
        species = torch.ones((1, 100), dtype=torch.long, device=self.device)
        for j in range(self.num_to_test):
            coordinates = (
                torch.randn((1, 100, 3), device=self.device, dtype=torch.double)
                * 3
                * self.cell_size
            )
            coordinates = torch.clamp(
                coordinates, min=0.0001, max=self.cell_size - 0.0001
            )
            _, e_cl = self.model_cl(
                (species, coordinates),
                cell=self.cell.to(self.device, torch.double),
                pbc=self.pbc.to(self.device),
            )
            _, e_fp = self.model_fp(
                (species, coordinates),
                cell=self.cell.to(self.device, torch.double),
                pbc=self.pbc.to(self.device),
            )
            self.assertEqual(e_cl, e_fp)

    def testCellListEnergiesRandomFloat(self):
        # The tolerance of this test is slightly modified because otherwise the
        # test also fails with AEVComputer AllPairs ** against itself **.
        # This is becausse non determinancy in order of operations in cuda
        # creates small floating point errors that may be larger than the
        # default threshold (I hope)
        self.model_cl = self.model_cl.to(self.device, torch.float)
        self.model_fp = self.model_fp.to(self.device, torch.float)
        species = torch.ones((1, 100), dtype=torch.long, device=self.device)
        for j in range(self.num_to_test):
            coordinates = (
                torch.randn((1, 100, 3), device=self.device, dtype=torch.float)
                * 3
                * self.cell_size
            )
            coordinates = torch.clamp(
                coordinates, min=0.0001, max=self.cell_size - 0.0001
            ).float()

            _, e_c = self.model_cl(
                (species, coordinates),
                cell=self.cell.to(self.device, torch.float),
                pbc=self.pbc.to(self.device),
            )
            _, e_j = self.model_fp(
                (species, coordinates),
                cell=self.cell.to(self.device, torch.float),
                pbc=self.pbc.to(self.device),
            )
            self.assertEqual(e_c, e_j, rtol=1e-4, atol=1e-4)

    def testCellListRandomFloat(self):
        aev_cl = self.aev_cl.to(self.device, torch.float)
        aev_fp = self.aev_fp.to(self.device, torch.float)
        idxs = torch.randint(
            self.aev_cl.num_species, (1, 100), device=self.device, dtype=torch.long
        )
        for j in range(self.num_to_test):
            coordinates = (
                torch.randn((1, 100, 3), device=self.device, dtype=torch.float)
                * 3
                * self.cell_size
            )
            coordinates = torch.clamp(
                coordinates, min=0.0001, max=self.cell_size - 0.0001
            )

            aevs_cl = aev_cl(
                idxs,
                coordinates,
                cell=self.cell.to(self.device),
                pbc=self.pbc.to(self.device),
            )
            aevs_fp = aev_fp(
                idxs,
                coordinates,
                cell=self.cell.to(self.device),
                pbc=self.pbc.to(self.device),
            )
            self.assertEqual(aevs_cl, aevs_fp, rtol=1e-4, atol=1e-4)


@unittest.skipIf(not torch.cuda.is_available(), "No cuda device found")
class TestCellListEnergiesCuda(TestCellListEnergies):
    def setUp(self):
        super().setUp()
        self.device = torch.device("cuda")
        self.num_to_test = 100


@expand()
class TestCellListLargeSystem(ANITestCase):
    def setUp(self):
        cut = 5.2
        cell_size = cut * 3 + 0.1

        self.cut = cut
        self.cell_size = cell_size
        self.num_to_test = 10
        self.pbc = torch.tensor(
            [True, True, True],
            dtype=torch.bool,
            device=self.device,
        )
        self.cell = torch.diag(
            torch.tensor([cell_size, cell_size, cell_size], device=self.device),
        ).float()
        self.aev_cl = self._setup(AEVComputer.like_1x(neighborlist="cell_list"))
        self.aev_fp = self._setup(AEVComputer.like_1x(neighborlist="all_pairs"))
        if self.jit and self.device == "cuda":
            # JIT + CUDA can have slightly different answers
            self.rtol = 1.0e-6
            self.atol = 1.0e-6
        else:
            self.rtol = 1.0e-7
            self.atol = 1.0e-7

    def testRandomNoPBC(self):
        idxs = torch.randint(
            self.aev_cl.num_species, (1, 100), device=self.device, dtype=torch.long
        )
        for j in range(self.num_to_test):
            coordinates = (
                torch.randn((1, 100, 3), device=self.device, dtype=torch.double)
                * 3
                * self.cell_size
            )
            coordinates = torch.clamp(
                coordinates, min=0.0001, max=self.cell_size - 0.0001
            )

            aevs_cl = self.aev_cl(idxs, coordinates)
            aevs_fp = self.aev_fp(idxs, coordinates)
            self.assertEqual(aevs_cl, aevs_fp, rtol=self.rtol, atol=self.atol)

    def testRandom(self):
        idxs = torch.randint(
            self.aev_cl.num_species, (1, 100), device=self.device, dtype=torch.long
        )
        for j in range(self.num_to_test):
            coordinates = (
                torch.randn((1, 100, 3), device=self.device, dtype=torch.double)
                * 3
                * self.cell_size
            )
            coordinates = torch.clamp(
                coordinates, min=0.0001, max=self.cell_size - 0.0001
            )

            aevs_cl = self.aev_cl(
                idxs,
                coordinates,
                cell=self.cell.to(self.device, torch.double),
                pbc=self.pbc.to(self.device),
            )
            aevs_fp = self.aev_fp(
                idxs,
                coordinates,
                cell=self.cell.to(self.device, torch.double),
                pbc=self.pbc.to(self.device),
            )
            self.assertEqual(aevs_cl, aevs_fp, rtol=self.rtol, atol=self.atol)


atom_grid_idx3_expect = torch.tensor(
    [
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 2],
            [0, 0, 2],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 1],
            [0, 1, 1],
            [0, 1, 2],
            [0, 1, 2],
            [0, 2, 0],
            [0, 2, 0],
            [0, 2, 1],
            [0, 2, 1],
            [0, 2, 2],
            [0, 2, 2],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 1],
            [1, 0, 1],
            [1, 0, 2],
            [1, 0, 2],
            [1, 1, 0],
            [1, 1, 0],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 2],
            [1, 1, 2],
            [1, 2, 0],
            [1, 2, 0],
            [1, 2, 1],
            [1, 2, 1],
            [1, 2, 2],
            [1, 2, 2],
            [2, 0, 0],
            [2, 0, 0],
            [2, 0, 1],
            [2, 0, 1],
            [2, 0, 2],
            [2, 0, 2],
            [2, 1, 0],
            [2, 1, 0],
            [2, 1, 1],
            [2, 1, 1],
            [2, 1, 2],
            [2, 1, 2],
            [2, 2, 0],
            [2, 2, 0],
            [2, 2, 1],
            [2, 2, 1],
            [2, 2, 2],
            [2, 2, 2],
        ]
    ],
    dtype=torch.long,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
