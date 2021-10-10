import unittest

import torch

import torchani
from torchani.testing import TestCase
from torchani.aev import AEVComputer, CellList
from torchani.geometry import tile_into_tight_cell


class TestCellList(TestCase):

    def setUp(self):
        self.device = torch.device('cpu')
        cut = 5.2
        cell_size = cut * 3 + 0.1

        self.cut = cut
        self.cell_size = cell_size
        # The length of the box is ~ 3 * cutoff this so that
        # 3 buckets in each direction are needed to cover it, if one uses
        # a cutoff of 5.2 and a bucket length of 5.200001
        coordinates = torch.tensor(
            [[cut / 2, cut / 2, cut / 2],
             [cut / 2 + 0.1, cut / 2 + 0.1, cut / 2 + 0.1]]).unsqueeze(0)
        species = torch.tensor([0, 0]).unsqueeze(0)
        species, coordinates, _ = tile_into_tight_cell((species, coordinates),
                                                     fixed_displacement_size=cut, make_coordinates_positive=False)
        assert species.shape[1] == 54
        assert coordinates.shape[1] == 54

        self.coordinates = coordinates
        self.species = species
        # first bucket is 0 - 5.2, in 3 directions, and subsequent buckets
        # are on top of that
        self.pbc = torch.tensor([True, True, True], dtype=torch.bool)
        self.cell = torch.diag(torch.tensor([cell_size, cell_size, cell_size])).float()
        self.clist = CellList(cut)

    def testInitDefault(self):
        clist = CellList(self.cut, buckets_per_cutoff=1)
        self.assertTrue(clist.buckets_per_cutoff == 1)
        self.assertTrue(clist.cutoff == self.cut)
        self.assertTrue(clist.num_neighbors == 13)
        self.assertEqual(clist.bucket_length_lower_bound, torch.full(size=(3,), fill_value=self.cut + 0.00001, device=self.device))

    def testSetupCell(self):
        clist = self.clist
        clist._setup_variables(self.cell)
        # this creates a unit cell with 27 buckets
        # and a grid of 3 x 3 x 3 buckets (3 in each direction, Gx = Gy = Gz = 3)
        self.assertTrue(clist.total_buckets == 27)
        self.assertTrue(
            (clist.shape_buckets_grid == torch.tensor([3, 3, 3],
                                                      dtype=torch.long)).all())
        # since it is padded shape should be 5 5 5
        self.assertTrue(
            clist.vector_idx_to_flat.shape == torch.Size([5, 5, 5]))

    def testVectorIndexToFlat(self):
        clist = self.clist
        clist._setup_variables(self.cell)
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
        clist = self.clist
        clist._setup_variables(self.cell)
        # test coordinate fractionalization
        frac = clist._fractionalize_coordinates(self.coordinates)
        self.assertTrue((frac < 1.0).all())

    def testVectorBucketIndex(self):
        clist = self.clist
        clist._setup_variables(self.cell)

        frac = clist._fractionalize_coordinates(self.coordinates)
        main_vector_bucket_index = clist._fractional_to_vector_bucket_indices(
            frac)
        self.assertTrue(
            main_vector_bucket_index.shape == torch.Size([1, 54, 3]))
        self.assertTrue(
            (main_vector_bucket_index == vector_bucket_index_compare).all())

    def testFlatBucketIndex(self):
        clist = self.clist
        clist._setup_variables(self.cell)
        flat = clist._to_flat_index(vector_bucket_index_compare)
        # all flat bucket indices are present
        flat_compare = torch.repeat_interleave(
            torch.arange(0, 27).to(torch.long), 2)
        self.assertTrue((flat == flat_compare).all())

    def testFlatBucketIndexAlternative(self):
        clist = self.clist
        clist._setup_variables(self.cell)
        atoms = vector_bucket_index_compare.shape[1]
        flat = clist.vector_idx_to_flat[(vector_bucket_index_compare
            + torch.ones(1, dtype=torch.long)).reshape(-1,
                3).unbind(1)].reshape(1, atoms)
        self.assertTrue(clist.total_buckets == 27)
        flat_compare = torch.repeat_interleave(
            torch.arange(0, 27).to(torch.long), 2)
        self.assertTrue((flat == flat_compare).all())

    def testCounts(self):
        num_flat = 27
        clist = self.clist
        clist._setup_variables(self.cell)
        flat = clist._to_flat_index(vector_bucket_index_compare)
        self.assertTrue(clist.total_buckets == num_flat)
        count_in_flat, cumcount_in_flat, max_ = clist._get_atoms_in_flat_bucket_counts(
            flat)
        # these are all 2
        self.assertTrue(count_in_flat.shape == torch.Size([num_flat]))
        self.assertTrue((count_in_flat == 2).all())
        # these are all 0 2 4 6 ...
        self.assertTrue(cumcount_in_flat.shape == torch.Size([num_flat]))
        self.assertTrue((cumcount_in_flat == torch.arange(0, 54, 2)).all())
        # max counts in a bucket is 2
        self.assertTrue(max_ == 2)

    def testWithinBetween(self):
        clist = self.clist
        clist._setup_variables(self.cell)
        frac = clist._fractionalize_coordinates(self.coordinates)
        _, f_a = clist._get_bucket_indices(frac)
        A_f, Ac_f, A_star = clist._get_atoms_in_flat_bucket_counts(f_a)
        within = clist._get_within_image_pairs(A_f, Ac_f, A_star)
        # some hand comparisons with the within indices
        self.assertTrue(within.shape == torch.Size([2, 27]))
        self.assertTrue((within[0] == torch.arange(0, 54, 2)).all())
        self.assertTrue((within[1] == torch.arange(1, 55, 2)).all())

    def testCellListInit(self):
        AEVComputer.like_1x(neighborlist='cell_list')

    def _check_neighborlists_consistency(self, coordinates, species=None):
        if species is None:
            species = torch.tensor([0, 0, 0]).unsqueeze(0)
        aev_cl = AEVComputer.like_1x(neighborlist='cell_list')
        aev_fp = AEVComputer.like_1x(neighborlist='full_pairwise')
        _, aevs_cl = aev_cl((species, coordinates),
                            cell=self.cell,
                            pbc=self.pbc)
        _, aevs_fp = aev_fp((species, coordinates),
                            cell=self.cell,
                            pbc=self.pbc)
        self.assertEqual(aevs_cl, aevs_fp)

    def testCellListIsConsistentV1(self):
        cut = self.cut
        d = 0.5
        coordinates = torch.tensor([
            [cut / 2 - d, cut / 2 - d, cut / 2 - d],
            [cut / 2 + 0.1, cut / 2 + 0.1, cut / 2 + 0.1],
            [cut / 2, cut / 2 + 2.4 * cut, cut / 2 + 2.4 * cut],
        ]).unsqueeze(0)
        self._check_neighborlists_consistency(coordinates)

    def testCellListIsConsistentV2(self):
        cut = self.cut
        d = 0.5
        coordinates = torch.tensor([
            [cut / 2 - d, cut / 2 - d, cut / 2 - d],
            [cut / 2 + 0.1, cut / 2 + 0.1, cut / 2 + 0.1],
            [cut / 2 + 2.4 * cut, cut / 2 + 2.4 * cut, cut / 2 + 2.4 * cut],
        ]).unsqueeze(0)
        self._check_neighborlists_consistency(coordinates)

    def testCellListIsConsistentV3(self):
        coordinates = torch.tensor([[[1.0000e-03, 6.5207e+00, 1.0000e-03],
                                     [1.0000e-03, 1.5299e+01, 1.3643e+01],
                                     [1.0000e-03, 2.1652e+00, 1.5299e+01]]])
        self._check_neighborlists_consistency(coordinates)

    def testCellListIsConsistentV4(self):
        coordinates = torch.tensor([[[1.5299e+01, 1.0000e-03, 5.5613e+00],
                                     [1.0000e-03, 1.0000e-03, 3.8310e+00],
                                     [1.5299e+01, 1.1295e+01, 1.5299e+01]]])
        self._check_neighborlists_consistency(coordinates)

    def testCellListIsConsistentV5(self):
        coordinates = torch.tensor([[
            [1.0000e-03, 1.0000e-03, 1.0000e-03],
            [1.0389e+01, 1.0000e-03, 1.5299e+01],
        ]])
        species = torch.tensor([0, 0]).unsqueeze(0)
        self._check_neighborlists_consistency(coordinates, species)

    def testCellListIsConsistentV6(self):
        self._check_neighborlists_consistency(self.coordinates, self.species)

    def testCellListIsConsistentRandom(self):
        cut = self.cut
        for j in range(100):
            coordinates = torch.tensor(
                [[cut / 2, cut / 2, cut / 2],
                 [cut / 2 + 0.1, cut / 2 + 0.1, cut / 2 + 0.1]]).unsqueeze(0)
            species = torch.tensor([0, 0]).unsqueeze(0)
            species, coordinates, _ = tile_into_tight_cell((species, coordinates),
                                                        fixed_displacement_size=cut,
                                                        noise=0.1, make_coordinates_positive=False)
            self._check_neighborlists_consistency(coordinates, species)

    def testCellListIsConsistentRandomV2(self):
        for j in range(100):
            coordinates = torch.randn(10, 3).unsqueeze(0) * 3 * self.cell_size
            coordinates = torch.clamp(coordinates,
                                      min=0.0001,
                                      max=self.cell_size - 0.0001)
            species = torch.zeros(10).unsqueeze(0).to(torch.long)
            self._check_neighborlists_consistency(coordinates, species)


class TestCellListEnergies(TestCase):
    def setUp(self):
        self.device = torch.device('cpu')
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
            torch.tensor([self.cell_size, self.cell_size,
                          self.cell_size])).float()
        self.aev_cl = AEVComputer.like_1x(neighborlist='cell_list')
        self.aev_fp = AEVComputer.like_1x(neighborlist='full_pairwise')
        self.model_cl = torchani.models.ANI1x(model_index=0).to(self.device)
        self.model_fp = torchani.models.ANI1x(model_index=0).to(self.device)
        self.model_cl.aev_computer = self.aev_cl
        self.model_fp.aev_computer = self.aev_fp
        self.num_to_test = 10

    def testCellListEnergiesRandom(self):
        self.model_cl = self.model_cl.to(self.device).double()
        self.model_fp = self.model_fp.to(self.device).double()
        species = torch.zeros(100).unsqueeze(0).to(torch.long).to(self.device)
        for j in range(self.num_to_test):
            coordinates = torch.randn(100, 3).unsqueeze(0).to(self.device).to(
                torch.double) * 3 * self.cell_size
            coordinates = torch.clamp(coordinates,
                                      min=0.0001,
                                      max=self.cell_size - 0.0001)
            _, e_cl = self.model_cl((species, coordinates),
                                    cell=self.cell.to(self.device).double(),
                                    pbc=self.pbc.to(self.device))
            _, e_fp = self.model_fp((species, coordinates),
                                    cell=self.cell.to(self.device).double(),
                                    pbc=self.pbc.to(self.device))
            self.assertEqual(e_cl, e_fp)

    def testCellListEnergiesRandomFloat(self):
        # The tolerance of this test is slightly modified because otherwise the
        # test also fails with AEVComputer FullPairwise ** against itself **.
        # This is becausse non determinancy in order of operations in cuda
        # creates small floating point errors that may be larger than the
        # default threshold
        self.model_cl = self.model_cl.to(self.device).float()
        self.model_fp = self.model_fp.to(self.device).float()
        species = torch.zeros(100).unsqueeze(0).to(torch.long).to(self.device)
        for j in range(self.num_to_test):
            coordinates = torch.randn(100, 3).unsqueeze(0).to(self.device).to(
                torch.float) * 3 * self.cell_size
            coordinates = torch.clamp(coordinates,
                                      min=0.0001,
                                      max=self.cell_size - 0.0001).float()

            _, e_c = self.model_cl((species, coordinates),
                                   cell=self.cell.to(self.device).float(),
                                   pbc=self.pbc.to(self.device))
            _, e_j = self.model_fp((species, coordinates),
                                   cell=self.cell.to(self.device).float(),
                                   pbc=self.pbc.to(self.device))
            self.assertEqual(e_c, e_j, rtol=1e-4, atol=1e-4)

    def testCellListLargeRandom(self):
        aev_cl = self.aev_cl.to(self.device).double()
        aev_fp = self.aev_fp.to(self.device).double()
        species = torch.LongTensor(100).random_(0, 4).to(self.device).unsqueeze(0)
        for j in range(self.num_to_test):
            coordinates = torch.randn(100, 3).unsqueeze(0).to(self.device).to(
                torch.double) * 3 * self.cell_size
            coordinates = torch.clamp(coordinates,
                                      min=0.0001,
                                      max=self.cell_size - 0.0001)

            _, aevs_cl = aev_cl((species, coordinates),
                                cell=self.cell.to(self.device).double(),
                                pbc=self.pbc.to(self.device))
            _, aevs_fp = aev_fp((species, coordinates),
                                cell=self.cell.to(self.device).double(),
                                pbc=self.pbc.to(self.device))
            self.assertEqual(aevs_cl, aevs_fp)

    def testCellListLargeRandomNoPBC(self):
        aev_cl = self.aev_cl.to(self.device).double()
        aev_fp = self.aev_fp.to(self.device).double()
        species = torch.LongTensor(100).random_(0, 4).to(self.device).unsqueeze(0)
        for j in range(self.num_to_test):
            coordinates = torch.randn(100, 3).unsqueeze(0).to(self.device).to(
                torch.double) * 3 * self.cell_size
            coordinates = torch.clamp(coordinates,
                                      min=0.0001,
                                      max=self.cell_size - 0.0001)

            _, aevs_cl = aev_cl((species, coordinates))
            _, aevs_fp = aev_fp((species, coordinates))
            self.assertEqual(aevs_cl, aevs_fp)

    def testCellListLargeRandomJITNoPBC(self):
        # JIT optimizations are avoided to prevent cuda bugs that make first evaluations extremely slow
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)  # this also has an effect
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_set_texpr_fuser_enabled(False)  # this has an effect
        if torch.cuda.is_available():
            torch._C._jit_set_nvfuser_enabled(False)
        aev_cl = torch.jit.script(self.aev_cl).to(self.device).double()
        aev_fp = torch.jit.script(self.aev_fp).to(self.device).double()
        species = torch.LongTensor(100).random_(0, 4).to(self.device).unsqueeze(0)
        for j in range(self.num_to_test):
            coordinates = torch.randn(100, 3).unsqueeze(0).to(self.device).to(
                torch.double) * 3 * self.cell_size
            coordinates = torch.clamp(coordinates,
                                      min=0.0001,
                                      max=self.cell_size - 0.0001)

            _, aevs_cl = aev_cl((species, coordinates))
            _, aevs_fp = aev_fp((species, coordinates))
            self.assertEqual(aevs_cl, aevs_fp)

    def testCellListLargeRandomJIT(self):
        # JIT optimizations are avoided to prevent cuda bugs that make first evaluations extremely slow
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)  # this also has an effect
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_set_texpr_fuser_enabled(False)  # this has an effect
        if torch.cuda.is_available():
            torch._C._jit_set_nvfuser_enabled(False)
        aev_cl = torch.jit.script(self.aev_cl).to(self.device).double()
        aev_fp = torch.jit.script(self.aev_fp).to(self.device).double()
        species = torch.LongTensor(100).random_(0, 4).to(self.device).unsqueeze(0)
        for j in range(self.num_to_test):
            coordinates = torch.randn(100, 3).unsqueeze(0).to(self.device).to(
                torch.double) * 3 * self.cell_size
            coordinates = torch.clamp(coordinates,
                                      min=0.0001,
                                      max=self.cell_size - 0.0001)

            _, aevs_cl = aev_cl((species, coordinates),
                                cell=self.cell.to(self.device).double(),
                                pbc=self.pbc.to(self.device))
            _, aevs_fp = aev_fp((species, coordinates),
                                cell=self.cell.to(self.device).double(),
                                pbc=self.pbc.to(self.device))
            self.assertEqual(aevs_cl, aevs_fp)

    def testCellListRandomFloat(self):
        aev_cl = self.aev_cl.to(self.device).to(torch.float)
        aev_fp = self.aev_fp.to(self.device).to(torch.float)
        species = torch.LongTensor(100).random_(0, 4).to(self.device).unsqueeze(0)
        for j in range(self.num_to_test):
            coordinates = torch.randn(100, 3).unsqueeze(0).to(self.device).to(
                torch.float) * 3 * self.cell_size
            coordinates = torch.clamp(coordinates,
                                      min=0.0001,
                                      max=self.cell_size - 0.0001)

            _, aevs_cl = aev_cl((species, coordinates),
                                cell=self.cell.to(self.device),
                                pbc=self.pbc.to(self.device))
            _, aevs_fp = aev_fp((species, coordinates),
                                cell=self.cell.to(self.device),
                                pbc=self.pbc.to(self.device))
            self.assertEqual(aevs_cl, aevs_fp, rtol=1e-4, atol=1e-4)


@unittest.skipIf(not torch.cuda.is_available(), 'No cuda device found')
class TestCellListEnergiesCuda(TestCellListEnergies):
    def setUp(self):
        super().setUp()
        self.device = torch.device('cuda')
        self.num_to_test = 100


vector_bucket_index_compare = torch.tensor([[[0, 0, 0],
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
                                       [2, 2, 2]]], dtype=torch.long)


if __name__ == '__main__':
    unittest.main()
