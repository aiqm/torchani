import torch
import torchani
import unittest
import random


class TestEnergyShifter(unittest.TestCase):

    def setUp(self):
        self.tol = 1e-5
        self.species = torchani.AEVComputer().species
        self.prepare = torchani.PrepareInput(self.species)
        self.shift_energy = torchani.EnergyShifter(self.species)

    def testSAEMatch(self):
        species_coordinates = []
        saes = []
        for _ in range(10):
            k = random.choice(range(5, 30))
            species = random.choices(self.species, k=k)
            coordinates = torch.empty(1, k, 3)
            species_coordinates.append(self.prepare((species, coordinates)))
            e = self.shift_energy.sae_from_list(species)
            saes.append(e)
        species, _ = torchani.padding.pad_and_batch(species_coordinates)
        saes_ = self.shift_energy.sae_from_tensor(species)
        saes = torch.tensor(saes, dtype=saes_.dtype, device=saes_.device)
        self.assertLess((saes - saes_).abs().max(), self.tol)


if __name__ == '__main__':
    unittest.main()
