import os
import torch
import torchani
import unittest

path = os.path.dirname(os.path.realpath(__file__))
dataset_path = os.path.join(path, '../dataset')
batch_size = 256
aev = torchani.AEVComputer()


class TestData(unittest.TestCase):

    def setUp(self):
        self.ds = torchani.training.BatchedANIDataset(dataset_path,
                                                      aev.species,
                                                      batch_size)

    def _assertTensorEqual(self, t1, t2):
        self.assertEqual((t1-t2).abs().max(), 0)

    def testSplitBatch(self):
        species1 = torch.randint(4, (5, 4), dtype=torch.long)
        coordinates1 = torch.randn(5, 4, 3)
        species2 = torch.randint(4, (2, 8), dtype=torch.long)
        coordinates2 = torch.randn(2, 8, 3)
        species3 = torch.randint(4, (10, 20), dtype=torch.long)
        coordinates3 = torch.randn(10, 20, 3)
        species, coordinates = torchani.padding.pad_and_batch([
            (species1, coordinates1),
            (species2, coordinates2),
            (species3, coordinates3),
        ])
        natoms = (species >= 0).to(torch.long).sum(1)
        chunks = torchani.training.data.split_batch(natoms, species, coordinates)
        start = 0
        last = None
        for s, c in chunks:
            n = (s >= 0).to(torch.long).sum(1)
            if last is not None:
                self.assertNotEqual(last[-1], n[0])
            conformations = s.shape[0]
            self.assertGreater(conformations, 0)
            s_ = species[start:start+conformations, ...]
            c_ = coordinates[start:start+conformations, ...]
            s_, c_ = torchani.padding.strip_redundant_padding(s_, c_)
            self._assertTensorEqual(s, s_)
            self._assertTensorEqual(c, c_)
            start += conformations

        s, c = torchani.padding.pad_and_batch(chunks)
        self._assertTensorEqual(s, species)
        self._assertTensorEqual(c, coordinates)

    def testTensorShape(self):
        for i in self.ds:
            input, output = i
            species, coordinates = torchani.padding.pad_and_batch(input)
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
            for input in i[0]:
                species, _ = input
                non_padding = (species >= 0)[:, -1].nonzero()
                self.assertGreater(non_padding.numel(), 0)


if __name__ == '__main__':
    unittest.main()
