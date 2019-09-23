import os
import torch
import torchani
import unittest
from torchani.data.cache_aev import cache_aev, cache_sparse_aev

path = os.path.dirname(os.path.realpath(__file__))
dataset_path = os.path.join(path, '../dataset/ani1-up_to_gdb4')
dataset_path2 = os.path.join(path, '../dataset/ani1-up_to_gdb4/ani_gdb_s01.h5')
batch_size = 256
ani1x = torchani.models.ANI1x()
consts = ani1x.consts
aev_computer = ani1x.aev_computer


class TestData(unittest.TestCase):

    def setUp(self):
        self.ds = torchani.data.load_ani_dataset(dataset_path,
                                                 consts.species_to_tensor,
                                                 batch_size)

    def _assertTensorEqual(self, t1, t2):
        self.assertLess((t1 - t2).abs().max().item(), 1e-6)

    def testSplitBatch(self):
        species1 = torch.randint(4, (5, 4), dtype=torch.long)
        coordinates1 = torch.randn(5, 4, 3)
        species2 = torch.randint(4, (2, 8), dtype=torch.long)
        coordinates2 = torch.randn(2, 8, 3)
        species3 = torch.randint(4, (10, 20), dtype=torch.long)
        coordinates3 = torch.randn(10, 20, 3)
        species_coordinates = torchani.utils.pad_atomic_properties([
            {'species': species1, 'coordinates': coordinates1},
            {'species': species2, 'coordinates': coordinates2},
            {'species': species3, 'coordinates': coordinates3},
        ])
        species = species_coordinates['species']
        coordinates = species_coordinates['coordinates']
        natoms = (species >= 0).to(torch.long).sum(1)
        chunks = torchani.data.split_batch(natoms, species_coordinates)
        start = 0
        last = None
        for chunk in chunks:
            s = chunk['species']
            c = chunk['coordinates']
            n = (s >= 0).to(torch.long).sum(1)
            if last is not None:
                self.assertNotEqual(last[-1], n[0])
            conformations = s.shape[0]
            self.assertGreater(conformations, 0)
            s_ = species[start:(start + conformations), ...]
            c_ = coordinates[start:(start + conformations), ...]
            sc = torchani.utils.strip_redundant_padding({'species': s_, 'coordinates': c_})
            s_ = sc['species']
            c_ = sc['coordinates']
            self._assertTensorEqual(s, s_)
            self._assertTensorEqual(c, c_)
            start += conformations

        sc = torchani.utils.pad_atomic_properties(chunks)
        s = sc['species']
        c = sc['coordinates']
        self._assertTensorEqual(s, species)
        self._assertTensorEqual(c, coordinates)

    def testTensorShape(self):
        for i in self.ds:
            input_, output = i
            input_ = [{'species': x[0], 'coordinates': x[1]} for x in input_]
            species_coordinates = torchani.utils.pad_atomic_properties(input_)
            species = species_coordinates['species']
            coordinates = species_coordinates['coordinates']
            energies = output['energies']
            self.assertEqual(len(species.shape), 2)
            self.assertLessEqual(species.shape[0], batch_size)
            self.assertEqual(len(coordinates.shape), 3)
            self.assertEqual(coordinates.shape[2], 3)
            self.assertEqual(coordinates.shape[:2], species.shape[:2])
            self.assertEqual(len(energies.shape), 1)
            self.assertEqual(coordinates.shape[0], energies.shape[0])

    def testNoUnnecessaryPadding(self):
        for i in self.ds:
            for input_ in i[0]:
                species, _ = input_
                non_padding = (species >= 0)[:, -1].nonzero()
                self.assertGreater(non_padding.numel(), 0)

    def testAEVCacheLoader(self):
        tmpdir = os.path.join(os.getcwd(), 'tmp')
        if not os.path.exists(tmpdir):
            os.makedirs(tmpdir)
        cache_aev(tmpdir, dataset_path2, 64, enable_tqdm=False)
        loader = torchani.data.AEVCacheLoader(tmpdir)
        ds = loader.dataset
        aev_computer_dev = aev_computer.to(loader.dataset.device)
        for _ in range(3):
            for (species_aevs, _), (species_coordinates, _) in zip(loader, ds):
                for (s1, a), (s2, c) in zip(species_aevs, species_coordinates):
                    self._assertTensorEqual(s1, s2)
                    s2, a2 = aev_computer_dev((s2, c))
                    self._assertTensorEqual(s1, s2)
                    self._assertTensorEqual(a, a2)

    def testSparseAEVCacheLoader(self):
        tmpdir = os.path.join(os.getcwd(), 'tmp')
        if not os.path.exists(tmpdir):
            os.makedirs(tmpdir)
        cache_sparse_aev(tmpdir, dataset_path2, 64, enable_tqdm=False)
        loader = torchani.data.SparseAEVCacheLoader(tmpdir)
        ds = loader.dataset
        aev_computer_dev = aev_computer.to(loader.dataset.device)
        for _ in range(3):
            for (species_aevs, _), (species_coordinates, _) in zip(loader, ds):
                for (s1, a), (s2, c) in zip(species_aevs, species_coordinates):
                    self._assertTensorEqual(s1, s2)
                    s2, a2 = aev_computer_dev((s2, c))
                    self._assertTensorEqual(s1, s2)
                    self._assertTensorEqual(a, a2)


if __name__ == '__main__':
    unittest.main()
