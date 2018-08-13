import unittest
import torch
import torchani


class TestPadAndBatch(unittest.TestCase):

    def testVectorSpecies(self):
        species1 = torch.LongTensor([0, 2, 3, 1])
        coordinates1 = torch.zeros(5, 4, 3)
        species2 = torch.LongTensor([3, 2, 0, 1, 0])
        coordinates2 = torch.zeros(2, 5, 3)
        species, coordinates = torchani.padding.pad_and_batch([
            (species1, coordinates1),
            (species2, coordinates2),
        ])
        self.assertEqual(species.shape[0], 7)
        self.assertEqual(species.shape[1], 5)
        expected_species = torch.LongTensor([
            [0, 2, 3, 1, -1],
            [0, 2, 3, 1, -1],
            [0, 2, 3, 1, -1],
            [0, 2, 3, 1, -1],
            [0, 2, 3, 1, -1],
            [3, 2, 0, 1, 0],
            [3, 2, 0, 1, 0],
        ])
        self.assertEqual((species - expected_species).abs().max().item(), 0)
        self.assertEqual(coordinates.abs().max().item(), 0)

    def testTensorShape1NSpecies(self):
        species1 = torch.LongTensor([[0, 2, 3, 1]])
        coordinates1 = torch.zeros(5, 4, 3)
        species2 = torch.LongTensor([3, 2, 0, 1, 0])
        coordinates2 = torch.zeros(2, 5, 3)
        species, coordinates = torchani.padding.pad_and_batch([
            (species1, coordinates1),
            (species2, coordinates2),
        ])
        self.assertEqual(species.shape[0], 7)
        self.assertEqual(species.shape[1], 5)
        expected_species = torch.LongTensor([
            [0, 2, 3, 1, -1],
            [0, 2, 3, 1, -1],
            [0, 2, 3, 1, -1],
            [0, 2, 3, 1, -1],
            [0, 2, 3, 1, -1],
            [3, 2, 0, 1, 0],
            [3, 2, 0, 1, 0],
        ])
        self.assertEqual((species - expected_species).abs().max().item(), 0)
        self.assertEqual(coordinates.abs().max().item(), 0)

    def testTensorSpecies(self):
        species1 = torch.LongTensor([
            [0, 2, 3, 1],
            [0, 2, 3, 1],
            [0, 2, 3, 1],
            [0, 2, 3, 1],
            [0, 2, 3, 1],
        ])
        coordinates1 = torch.zeros(5, 4, 3)
        species2 = torch.LongTensor([3, 2, 0, 1, 0])
        coordinates2 = torch.zeros(2, 5, 3)
        species, coordinates = torchani.padding.pad_and_batch([
            (species1, coordinates1),
            (species2, coordinates2),
        ])
        self.assertEqual(species.shape[0], 7)
        self.assertEqual(species.shape[1], 5)
        expected_species = torch.LongTensor([
            [0, 2, 3, 1, -1],
            [0, 2, 3, 1, -1],
            [0, 2, 3, 1, -1],
            [0, 2, 3, 1, -1],
            [0, 2, 3, 1, -1],
            [3, 2, 0, 1, 0],
            [3, 2, 0, 1, 0],
        ])
        self.assertEqual((species - expected_species).abs().max().item(), 0)
        self.assertEqual(coordinates.abs().max().item(), 0)

    def testPresentSpecies(self):
        species = torch.LongTensor([0, 1, 1, 0, 3, 7, -1, -1])
        present_species = torchani.padding.present_species(species)
        expected = torch.LongTensor([0, 1, 3, 7])
        self.assertEqual((expected - present_species).abs().max().item(), 0)


class TestStripRedundantPadding(unittest.TestCase):

    def _assertTensorEqual(self, t1, t2):
        self.assertEqual((t1 - t2).abs().max().item(), 0)

    def testStripRestore(self):
        species1 = torch.randint(4, (5, 4), dtype=torch.long)
        coordinates1 = torch.randn(5, 4, 3)
        species2 = torch.randint(4, (2, 5), dtype=torch.long)
        coordinates2 = torch.randn(2, 5, 3)
        species12, coordinates12 = torchani.padding.pad_and_batch([
            (species1, coordinates1),
            (species2, coordinates2),
        ])
        species3 = torch.randint(4, (2, 10), dtype=torch.long)
        coordinates3 = torch.randn(2, 10, 3)
        species123, coordinates123 = torchani.padding.pad_and_batch([
            (species1, coordinates1),
            (species2, coordinates2),
            (species3, coordinates3),
        ])
        species1_, coordinates1_ = torchani.padding.strip_redundant_padding(
            species123[:5, ...], coordinates123[:5, ...])
        self._assertTensorEqual(species1_, species1)
        self._assertTensorEqual(coordinates1_, coordinates1)
        species12_, coordinates12_ = torchani.padding.strip_redundant_padding(
            species123[:7, ...], coordinates123[:7, ...])
        self._assertTensorEqual(species12_, species12)
        self._assertTensorEqual(coordinates12_, coordinates12)


if __name__ == '__main__':
    unittest.main()
