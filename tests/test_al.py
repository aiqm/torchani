import torch
import torchani
import math
import unittest
from torchani.testing import TestCase


class TestALAtomic(TestCase):
    def setUp(self):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torchani.models.ANI1x(periodic_table_index=True).to(
            self.device).double()
        self.converter = torchani.nn.SpeciesConverter(['H', 'C', 'N', 'O'])
        self.aev_computer = self.model.aev_computer
        self.ani_model = self.model.neural_networks
        self.first_model = self.model[0]
        # fully symmetric methane
        self.coordinates = torch.tensor(
            [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [0.5, 0.5, 0.5]],
            dtype=torch.double,
            device=self.device).unsqueeze(0)
        self.species = torch.tensor([[1, 1, 1, 1, 6]],
                                    dtype=torch.long,
                                    device=self.device)

    def testAverageAtomicEnergies(self):
        _, energies = self.model.atomic_energies(
            (self.species, self.coordinates))
        self.assertEqual(energies.shape, self.coordinates.shape[:-1])
        # energies of all hydrogens should be equal
        expect = torch.full(energies[:, :-1].shape, -0.54853380570289400620, dtype=torch.double, device=self.device)
        self.assertEqual(energies[:, :-1], expect)

    def testAtomicEnergies(self):
        _, energies = self.model.atomic_energies(
            (self.species, self.coordinates), average=False)
        self.assertTrue(energies.shape[1:] == self.coordinates.shape[:-1])
        self.assertTrue(energies.shape[0] == len(self.model.neural_networks))
        # energies of all hydrogens should be equal
        self.assertEqual(energies[0, 0, 0], torch.tensor(-0.54562734428531045605, device=self.device,
                    dtype=torch.double))
        for e in energies:
            self.assertTrue((e[:, :-1] == e[:, 0]).all())


class TestALQBC(TestALAtomic):
    def testMemberEnergies(self):
        # fully symmetric methane
        _, energies = self.model.members_energies(
            (self.species, self.coordinates))

        # correctness of shape
        torch.set_printoptions(precision=15)
        self.assertEqual(energies.shape[-1], self.coordinates.shape[0])
        self.assertEqual(energies.shape[0], len(self.model.neural_networks))
        self.assertEqual(
            energies[0], self.first_model((self.species,
                                           self.coordinates)).energies)
        expect = torch.tensor([-40.277153758433975],
                             dtype=torch.double,
                             device=self.device)
        self.assertEqual(energies[0], expect)

    def testQBC(self):
        # fully symmetric methane
        _, _, qbc = self.model.energies_qbcs((self.species, self.coordinates))

        torch.set_printoptions(precision=15)
        std = self.model.members_energies(
            (self.species, self.coordinates)).energies.std(dim=0,
                                                           unbiased=True)
        self.assertTrue(
            torch.isclose(std / math.sqrt(self.coordinates.shape[1]), qbc))

        # also test with multiple coordinates
        coord1 = torch.tensor(
            [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [0.5, 0.5, 0.5]],
            dtype=torch.double,
            device=self.device).unsqueeze(0)
        coord2 = torch.randn(1, 5, 3, dtype=torch.double, device=self.device)

        coordinates = torch.cat((coord1, coord2), dim=0)
        species = torch.tensor([[1, 1, 1, 1, 6], [-1, 1, 1, 1, 1]],
                               dtype=torch.long,
                               device=self.device)
        std = self.model.members_energies(
            (species, coordinates)).energies.std(dim=0, unbiased=True)
        _, _, qbc = self.model.energies_qbcs((species, coordinates))
        std[0] = std[0] / math.sqrt(5)
        std[1] = std[1] / math.sqrt(4)
        self.assertEqual(std, qbc)


if __name__ == '__main__':
    unittest.main()
