import unittest
import torch
import torchani
from torchani.testing import TestCase


b = torchani.utils.broadcast_first_dim


class TestPaddings(TestCase):

    def testVectorSpecies(self):
        species1 = torch.tensor([[0, 2, 3, 1]])
        coordinates1 = torch.zeros(5, 4, 3)
        species2 = torch.tensor([[3, 2, 0, 1, 0]])
        coordinates2 = torch.zeros(2, 5, 3)
        atomic_properties = torchani.utils.pad_atomic_properties([
            b({'species': species1, 'coordinates': coordinates1}),
            b({'species': species2, 'coordinates': coordinates2}),
        ])
        self.assertEqual(atomic_properties['species'].shape[0], 7)
        self.assertEqual(atomic_properties['species'].shape[1], 5)
        expected_species = torch.tensor([
            [0, 2, 3, 1, -1],
            [0, 2, 3, 1, -1],
            [0, 2, 3, 1, -1],
            [0, 2, 3, 1, -1],
            [0, 2, 3, 1, -1],
            [3, 2, 0, 1, 0],
            [3, 2, 0, 1, 0],
        ])
        self.assertEqual(atomic_properties['species'], expected_species)
        self.assertEqual(atomic_properties['coordinates'].abs().max().item(), 0)

    def testTensorShape1NSpecies(self):
        species1 = torch.tensor([[0, 2, 3, 1]])
        coordinates1 = torch.zeros(5, 4, 3)
        species2 = torch.tensor([[3, 2, 0, 1, 0]])
        coordinates2 = torch.zeros(2, 5, 3)
        atomic_properties = torchani.utils.pad_atomic_properties([
            b({'species': species1, 'coordinates': coordinates1}),
            b({'species': species2, 'coordinates': coordinates2}),
        ])
        self.assertEqual(atomic_properties['species'].shape[0], 7)
        self.assertEqual(atomic_properties['species'].shape[1], 5)
        expected_species = torch.tensor([
            [0, 2, 3, 1, -1],
            [0, 2, 3, 1, -1],
            [0, 2, 3, 1, -1],
            [0, 2, 3, 1, -1],
            [0, 2, 3, 1, -1],
            [3, 2, 0, 1, 0],
            [3, 2, 0, 1, 0],
        ])
        self.assertEqual(atomic_properties['species'], expected_species)
        self.assertEqual(atomic_properties['coordinates'].abs().max().item(), 0)

    def testTensorSpecies(self):
        species1 = torch.tensor([
            [0, 2, 3, 1],
            [0, 2, 3, 1],
            [0, 2, 3, 1],
            [0, 2, 3, 1],
            [0, 2, 3, 1],
        ])
        coordinates1 = torch.zeros(5, 4, 3)
        species2 = torch.tensor([[3, 2, 0, 1, 0]])
        coordinates2 = torch.zeros(2, 5, 3)
        atomic_properties = torchani.utils.pad_atomic_properties([
            b({'species': species1, 'coordinates': coordinates1}),
            b({'species': species2, 'coordinates': coordinates2}),
        ])
        self.assertEqual(atomic_properties['species'].shape[0], 7)
        self.assertEqual(atomic_properties['species'].shape[1], 5)
        expected_species = torch.tensor([
            [0, 2, 3, 1, -1],
            [0, 2, 3, 1, -1],
            [0, 2, 3, 1, -1],
            [0, 2, 3, 1, -1],
            [0, 2, 3, 1, -1],
            [3, 2, 0, 1, 0],
            [3, 2, 0, 1, 0],
        ])
        self.assertEqual(atomic_properties['species'], expected_species)
        self.assertEqual(atomic_properties['coordinates'].abs().max().item(), 0)

    def testPresentSpecies(self):
        species = torch.tensor([0, 1, 1, 0, 3, 7, -1, -1])
        present_species = torchani.utils.present_species(species)
        expected = torch.tensor([0, 1, 3, 7])
        self.assertEqual(expected, present_species)


class TestStripRedundantPadding(TestCase):

    def testStripRestore(self):
        species1 = torch.randint(4, (5, 4), dtype=torch.long)
        coordinates1 = torch.randn(5, 4, 3)
        species2 = torch.randint(4, (2, 5), dtype=torch.long)
        coordinates2 = torch.randn(2, 5, 3)
        atomic_properties12 = torchani.utils.pad_atomic_properties([
            b({'species': species1, 'coordinates': coordinates1}),
            b({'species': species2, 'coordinates': coordinates2}),
        ])
        species12 = atomic_properties12['species']
        coordinates12 = atomic_properties12['coordinates']
        species3 = torch.randint(4, (2, 10), dtype=torch.long)
        coordinates3 = torch.randn(2, 10, 3)
        atomic_properties123 = torchani.utils.pad_atomic_properties([
            b({'species': species1, 'coordinates': coordinates1}),
            b({'species': species2, 'coordinates': coordinates2}),
            b({'species': species3, 'coordinates': coordinates3}),
        ])
        species123 = atomic_properties123['species']
        coordinates123 = atomic_properties123['coordinates']
        species_coordinates1_ = torchani.utils.strip_redundant_padding(
            b({'species': species123[:5, ...], 'coordinates': coordinates123[:5, ...]}))
        species1_ = species_coordinates1_['species']
        coordinates1_ = species_coordinates1_['coordinates']
        self.assertEqual(species1_, species1)
        self.assertEqual(coordinates1_, coordinates1)
        species_coordinates12_ = torchani.utils.strip_redundant_padding(
            b({'species': species123[:7, ...], 'coordinates': coordinates123[:7, ...]}))
        species12_ = species_coordinates12_['species']
        coordinates12_ = species_coordinates12_['coordinates']
        self.assertEqual(species12_, species12)
        self.assertEqual(coordinates12_, coordinates12)


if __name__ == '__main__':
    unittest.main()
