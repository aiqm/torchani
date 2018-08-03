import torch
import torchani
import unittest
import random


class TestEnergyShifter(unittest.TestCase):

    def setUp(self):
        self.tol = 1e-5
        self.species = torchani.SortedAEV().species
        self.prepare = torchani.PrepareInput(self.species)
        self.shift_energy = torchani.EnergyShifter(self.species)

    def testSAEMatch(self):
        for _ in range(10):
            k = random.choice(range(5, 30))
            species = random.choices(self.species, k=k)
            species_tensor = self.prepare.species_to_tensor(
                species, torch.device('cpu'))
            e1 = self.shift_energy.sae_from_list(species)
            e2 = self.shift_energy.sae_from_tensor(species_tensor)
            self.assertLess(abs(e1 - e2), self.tol)


if __name__ == '__main__':
    unittest.main()
