import unittest

import torch
from torch import Tensor

import torchani
from torchani.testing import TestCase, expand, ANITest
from torchani.aev import AEVComputer
from torchani.neighbors import (
    CellList,
    fractionalize_coords,
    flatten_grid_idx3,
    coords_to_grid_idx3,
    image_pairs_within_grid_elements,
    count_atoms_in_grid,
)
from torchani.geometry import tile_into_tight_cell


class TestCellList(TestCase):
    def setUp(self):
        self.device = torch.device("cpu")

        self.cutoff = 5.2
        self.cell_size = self.cutoff * 3 + 0.1
        # The length of the box is ~ 3 * cutoff this so that
        # 3 buckets in each direction are needed to cover it, if one uses
        # a cutoff of 5.2 and a bucket length of 5.200001
        coordinates = torch.full(
            (1, 2, 3), self.cutoff / 2, dtype=torch.float, device=self.device
        )
        coordinates[:, 1, :] += 0.1
        species = torch.ones((1, 2), dtype=torch.long, device=self.device)
        self.species, self.coordinates, _ = tile_into_tight_cell(
            (species, coordinates),
            fixed_displacement_size=self.cutoff,
            make_coordinates_positive=False,
        )
        assert self.species.shape == (1, 54)
        assert self.coordinates.shape == (1, 54, 3)

        # first bucket is 0 - 5.2, in 3 directions, and subsequent buckets
        # are on top of that
        self.pbc = torch.tensor(
            [True, True, True], dtype=torch.bool, device=self.device
        )
        self.cell = torch.eye(3, dtype=torch.float, device=self.device) * self.cell_size
        self.clist = CellList()

    def testInitDefault(self):
        self.assertTrue(self.clist.buckets_per_cutoff == 1)
        self.assertTrue(self.clist.num_neighbors == 13)

    def testSetupCell(self):
        clist = self.clist
        clist._setup_variables(self.cell, self.cutoff)
        # this creates a unit cell with 27 buckets
        # and a grid of 3 x 3 x 3 buckets (3 in each direction, GX == GY == GZ == 3)
        expect_shape = torch.tensor([3, 3, 3], dtype=torch.long, device=self.device)
        self.assertEqual(clist.grid_numel, 27)
        self.assertEqual(clist.grid_shape, expect_shape)
        # since it is circularly padded, shape should be 5 5 5
        self.assertEqual(clist.vector_idx_to_flat.shape, (5, 5, 5))

    def testVectorIndexToFlat(self):
        clist = self.clist
        clist._setup_variables(self.cell, self.cutoff)
        # check some specific values of the tensor, it should be in row major
        # order so for instance the values in the z axis are 0 1 2 in the y
        # axis 0 3 6 and in the x axis 0 9 18
        self.assertTrue(clist.vector_idx_to_flat[1, 1, 1] == 0)
        self.assertTrue(clist.vector_idx_to_flat[2, 1, 1] == 9)
        self.assertTrue(clist.vector_idx_to_flat[1, 2, 1] == 3)
        self.assertTrue(clist.vector_idx_to_flat[1, 1, 2] == 1)
        self.assertTrue(clist.vector_idx_to_flat[0, 1, 1] == 18)
        self.assertTrue(clist.vector_idx_to_flat[1, 0, 1] == 6)
        self.assertTrue(clist.vector_idx_to_flat[1, 1, 0] == 2)

    def testFractionalize(self):
        # Coordinate fractionalization
        frac = fractionalize_coords(self.coordinates, self.cell)
        self.assertTrue(~torch.isnan(frac).any())
        self.assertTrue(~torch.isinf(frac).any())
        self.assertTrue((frac < 1.0).all())
        self.assertTrue((frac >= 0.0).all())

    def testGridIdx3(self):
        clist = self.clist
        clist._setup_variables(self.cell, self.cutoff)
        atom_grid_idx3 = coords_to_grid_idx3(
            self.coordinates, self.cell, clist.grid_shape
        )
        self.assertTrue(atom_grid_idx3.shape == (1, 54, 3))
        self.assertEqual(atom_grid_idx3, atom_grid_idx3_expect)

    def testGridIdx(self):
        clist = self.clist
        clist._setup_variables(self.cell, self.cutoff)
        grid_idx = flatten_grid_idx3(atom_grid_idx3_expect, clist.grid_shape)
        # All flat grid indices are present in this test
        grid_idx_compare = torch.repeat_interleave(
            torch.arange(0, 27, dtype=torch.long), 2
        )
        self.assertEqual(grid_idx, grid_idx_compare.view(1, -1))

    def testGridIndexAlt(self):
        # Alternative test
        clist = self.clist
        clist._setup_variables(self.cell, self.cutoff)
        grid_idx = clist.vector_idx_to_flat[
            (atom_grid_idx3_expect + torch.ones(1, dtype=torch.long))
            .reshape(-1, 3)
            .unbind(1)
        ].reshape(1, -1)
        grid_idx_compare = torch.repeat_interleave(
            torch.arange(0, 27, dtype=torch.long), 2
        )
        self.assertEqual(clist.grid_numel, 27)
        self.assertEqual(grid_idx, grid_idx_compare.view(1, -1))

    def testCounts(self):
        grid_numel_expect = 27
        clist = self.clist
        clist._setup_variables(self.cell, self.cutoff)
        atom_grid_idx = flatten_grid_idx3(atom_grid_idx3_expect, clist.grid_shape)
        self.assertEqual(clist.grid_numel, grid_numel_expect)
        grid_count, grid_cumcount = count_atoms_in_grid(atom_grid_idx, clist.grid_numel)
        # these are all 2
        self.assertEqual(grid_count.shape, (grid_numel_expect,))
        self.assertTrue((grid_count == 2).all())
        # these are all 0 2 4 6 ...
        self.assertEqual(grid_cumcount, torch.arange(0, 54, 2))

    def testWithinBetween(self):
        clist = self.clist
        clist._setup_variables(self.cell, self.cutoff)
        atom_grid_idx3 = coords_to_grid_idx3(
            self.coordinates, self.cell, clist.grid_shape
        )
        atom_grid_idx = flatten_grid_idx3(atom_grid_idx3, clist.grid_shape)
        grid_count, grid_cumcount = count_atoms_in_grid(atom_grid_idx, clist.grid_numel)
        within = image_pairs_within_grid_elements(
            grid_count,
            grid_cumcount,
            int(grid_count.max()),
        )
        # some hand comparisons with the within indices
        self.assertEqual(within.shape, (2, 27))
        self.assertEqual(within[0], torch.arange(0, 54, 2))
        self.assertEqual(within[1], torch.arange(1, 55, 2))

    def testCellListInit(self):
        AEVComputer.like_1x(neighborlist="cell_list")

    def testCellListMatchesFullPairwise(self):
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

    def testCellListMatchesFullPairwiseRandomNoise(self):
        cut = self.cutoff
        for j in range(100):
            coordinates = torch.tensor(
                [
                    [cut / 2, cut / 2, cut / 2],
                    [cut / 2 + 0.1, cut / 2 + 0.1, cut / 2 + 0.1],
                ]
            ).unsqueeze(0)
            species = torch.tensor([[1, 1]], dtype=torch.long)
            _, coordinates, _ = tile_into_tight_cell(
                (species, coordinates),
                fixed_displacement_size=cut,
                noise=0.1,
                make_coordinates_positive=False,
            )
            self._check_neighborlists_match(coordinates)

    def testCellListMatchesFullPairwiseRandomNormal(self):
        for j in range(100):
            coordinates = (
                torch.randn((1, 10, 3), device=self.device, dtype=torch.float)
                * 3
                * self.cell_size
            )
            coordinates = torch.clamp(
                coordinates, min=0.0001, max=self.cell_size - 0.0001
            )
            self._check_neighborlists_match(coordinates)

    def _check_neighborlists_match(self, coordinates: Tensor):
        species = torch.ones(
            coordinates.shape[:-1], dtype=torch.long, device=coordinates.device
        )
        aev_cl = AEVComputer.like_1x(neighborlist="cell_list")
        aev_fp = AEVComputer.like_1x(neighborlist="full_pairwise")
        _, aevs_cl = aev_cl((species, coordinates), cell=self.cell, pbc=self.pbc)
        _, aevs_fp = aev_fp((species, coordinates), cell=self.cell, pbc=self.pbc)
        self.assertEqual(aevs_cl, aevs_fp)


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
        self.aev_fp = AEVComputer.like_1x(neighborlist="full_pairwise")
        self.model_cl = torchani.models.ANI1x(model_index=0).to(self.device)
        self.model_fp = torchani.models.ANI1x(model_index=0).to(self.device)
        self.model_cl.aev_computer = self.aev_cl
        self.model_fp.aev_computer = self.aev_fp
        self.num_to_test = 10

    def testCellListEnergiesRandom(self):
        self.model_cl = self.model_cl.to(self.device).double()
        self.model_fp = self.model_fp.to(self.device).double()
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
                cell=self.cell.to(self.device).double(),
                pbc=self.pbc.to(self.device),
            )
            _, e_fp = self.model_fp(
                (species, coordinates),
                cell=self.cell.to(self.device).double(),
                pbc=self.pbc.to(self.device),
            )
            self.assertEqual(e_cl, e_fp)

    def testCellListEnergiesRandomFloat(self):
        # The tolerance of this test is slightly modified because otherwise the
        # test also fails with AEVComputer FullPairwise ** against itself **.
        # This is becausse non determinancy in order of operations in cuda
        # creates small floating point errors that may be larger than the
        # default threshold (I hope)
        self.model_cl = self.model_cl.to(self.device).float()
        self.model_fp = self.model_fp.to(self.device).float()
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
                cell=self.cell.to(self.device).float(),
                pbc=self.pbc.to(self.device),
            )
            _, e_j = self.model_fp(
                (species, coordinates),
                cell=self.cell.to(self.device).float(),
                pbc=self.pbc.to(self.device),
            )
            self.assertEqual(e_c, e_j, rtol=1e-4, atol=1e-4)

    def testCellListRandomFloat(self):
        aev_cl = self.aev_cl.to(self.device).to(torch.float)
        aev_fp = self.aev_fp.to(self.device).to(torch.float)
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

            _, aevs_cl = aev_cl(
                (idxs, coordinates),
                cell=self.cell.to(self.device),
                pbc=self.pbc.to(self.device),
            )
            _, aevs_fp = aev_fp(
                (idxs, coordinates),
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
class TestCellListLargeSystem(ANITest):
    def setUp(self):
        # JIT optimizations are avoided to prevent cuda bugs that make first
        # evaluations extremely slow (?)
        torch._C._jit_set_profiling_executor(False)
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
        self.aev_fp = self._setup(AEVComputer.like_1x(neighborlist="full_pairwise"))
        if self.jit and self.device == "cuda":
            # JIT + CUDA can have slightly different answers
            self.rtol = 1.0e-6
            self.atol = 1.0e-6
        else:
            self.rtol = 1.0e-7
            self.atol = 1.0e-7

    def tearDown(self) -> None:
        # JIT optimizations are reset since this generates bugs in the
        # repulsion tests (!) for some reason TODO: Figure out why and see if
        # this remains in pytorch 2
        torch._C._jit_set_profiling_executor(True)

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

            _, aevs_cl = self.aev_cl((idxs, coordinates))
            _, aevs_fp = self.aev_fp((idxs, coordinates))
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

            _, aevs_cl = self.aev_cl(
                (idxs, coordinates),
                cell=self.cell.to(self.device).double(),
                pbc=self.pbc.to(self.device),
            )
            _, aevs_fp = self.aev_fp(
                (idxs, coordinates),
                cell=self.cell.to(self.device).double(),
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
