import math
import unittest

import torch
import torchani

from torchani._testing import ANITestCase, expand
from torchani.grad import forces, energies_and_forces


@expand()
class TestActiveLearning(ANITestCase):
    def setUp(self):
        model = torchani.models.ANI1x().double()
        self.model = self._setup(model)
        self.num_networks = model.neural_networks.get_active_members_num()
        self.first_model = self._setup(model[0])

        # fully symmetric methane
        self.coordinates = torch.tensor(
            [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [0.5, 0.5, 0.5]],
            dtype=torch.double,
            device=self.device,
        ).unsqueeze(0)
        self.species = torch.tensor(
            [[1, 1, 1, 1, 6]], dtype=torch.long, device=self.device
        )

    def testMemberEnergies(self):
        # fully symmetric methane
        _, energies = self.model((self.species, self.coordinates), ensemble_values=True)

        # correctness of shape
        self.assertEqual(energies.shape[-1], self.coordinates.shape[0])
        self.assertEqual(energies.shape[0], self.num_networks)
        self.assertEqual(
            energies[0], self.first_model((self.species, self.coordinates)).energies
        )
        expect = torch.tensor(
            [-40.277153758433975], dtype=torch.double, device=self.device
        )
        self.assertEqual(energies[0], expect)

    def testQBC(self):
        # fully symmetric methane
        _, _, qbc = self.model.energies_qbcs((self.species, self.coordinates))

        std = self.model(
            (self.species, self.coordinates),
            ensemble_values=True,
        ).energies.std(dim=0, unbiased=True)
        self.assertTrue(torch.isclose(std / math.sqrt(self.coordinates.shape[1]), qbc))

        # also test with multiple coordinates
        coord1 = torch.tensor(
            [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [0.5, 0.5, 0.5]],
            dtype=torch.double,
            device=self.device,
        ).unsqueeze(0)
        coord2 = torch.randn(1, 5, 3, dtype=torch.double, device=self.device)

        coordinates = torch.cat((coord1, coord2), dim=0)
        species = torch.tensor(
            [[1, 1, 1, 1, 6], [-1, 1, 1, 1, 1]], dtype=torch.long, device=self.device
        )
        std = self.model((species, coordinates), ensemble_values=True).energies.std(
            dim=0, unbiased=True
        )
        _, _, qbc = self.model.energies_qbcs((species, coordinates))
        std[0] = std[0] / math.sqrt(5)
        std[1] = std[1] / math.sqrt(4)
        self.assertEqual(std, qbc)


# Note that forces functions are non-jittable
@expand(jit=False)
class TestActiveLearningForces(ANITestCase):
    def setUp(self):
        self.model = self._setup(torchani.models.ANI1x().double())
        # fully symmetric methane
        self.coordinates = torch.tensor(
            [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [0.5, 0.5, 0.5]],
            dtype=torch.double,
            device=self.device,
        ).unsqueeze(0)
        self.species = torch.tensor(
            [[1, 1, 1, 1, 6]], dtype=torch.long, device=self.device
        )

    def testAverageMembersForces(self):
        # Symmetric methane
        _, expect_forces = energies_and_forces(
            self.model, self.species, self.coordinates
        )
        self.coordinates.requires_grad_(True)
        members_energies = self.model(
            (self.species, self.coordinates),
            ensemble_values=True,
        ).energies
        forces_list = [
            forces(
                energies,
                self.coordinates,
                retain_graph=True,
            )
            for energies in members_energies
        ]
        test_forces = torch.stack(forces_list, dim=0)
        self.assertEqual(expect_forces, test_forces.mean(0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
