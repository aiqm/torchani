import os
import torch
import torchani
import unittest
from torchani.data.cache_aev import cache_aev

path = os.path.dirname(os.path.realpath(__file__))
dataset_path = os.path.join(path, '../dataset/ani1-up_to_gdb4')
dataset_path2 = os.path.join(path, '../dataset/ani1-up_to_gdb4/ani_gdb_s01.h5')
batch_size = 256
builtins = torchani.neurochem.Builtins()
consts = builtins.consts
aev_computer = builtins.aev_computer


class TestData(unittest.TestCase):

    def setUp(self):
        self.ds = torchani.data.BatchedANIDataset(dataset_path,
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
        species, coordinates = torchani.utils.pad_coordinates([
            (species1, coordinates1),
            (species2, coordinates2),
            (species3, coordinates3),
        ])
        natoms = (species >= 0).to(torch.long).sum(1)
        chunks = torchani.data.split_batch(natoms, species, coordinates)
        start = 0
        last = None
        for s, c in chunks:
            n = (s >= 0).to(torch.long).sum(1)
            if last is not None:
                self.assertNotEqual(last[-1], n[0])
            conformations = s.shape[0]
            self.assertGreater(conformations, 0)
            s_ = species[start:(start + conformations), ...]
            c_ = coordinates[start:(start + conformations), ...]
            s_, c_ = torchani.utils.strip_redundant_padding(s_, c_)
            self._assertTensorEqual(s, s_)
            self._assertTensorEqual(c, c_)
            start += conformations

        s, c = torchani.utils.pad_coordinates(chunks)
        self._assertTensorEqual(s, species)
        self._assertTensorEqual(c, coordinates)

    def testTensorShape(self):
        for i in self.ds:
            input_, output = i
            species, coordinates = torchani.utils.pad_coordinates(input_)
            energies = output['energies']
            self.assertEqual(len(species.shape), 2)
            self.assertLessEqual(species.shape[0], batch_size)
            self.assertEqual(len(coordinates.shape), 3)
            self.assertEqual(coordinates.shape[2], 3)
            self.assertEqual(coordinates.shape[1], coordinates.shape[1])
            self.assertEqual(coordinates.shape[0], coordinates.shape[0])
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


if __name__ == '__main__':
    unittest.main()
