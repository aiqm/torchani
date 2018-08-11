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


if __name__ == '__main__':
    unittest.main()
