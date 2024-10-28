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

    def testAverageAtomicEnergies(self):
        _, energies = self.model.atomic_energies((self.species, self.coordinates))
        self.assertEqual(energies.shape, self.coordinates.shape[:-1])
        # energies of all hydrogens should be equal
        expect = torch.full(
            energies[:, :-1].shape,
            -0.54853380570289400620,
            dtype=torch.double,
            device=self.device,
        )
        self.assertEqual(energies[:, :-1], expect)

    def testAtomicEnergies(self):
        _, energies = self.model.atomic_energies(
            (self.species, self.coordinates),
            ensemble_values=True,
        )
        self.assertTrue(energies.shape[1:] == self.coordinates.shape[:-1])
        self.assertTrue(energies.shape[0] == self.num_networks)
        # energies of all hydrogens should be equal
        self.assertEqual(
            energies[0, 0, 0],
            torch.tensor(
                -0.54562734428531045605, device=self.device, dtype=torch.double
            ),
        )
        for e in energies:
            self.assertTrue((e[:, :-1] == e[:, 0]).all())

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

    def testAtomicStdev(self):
        # Symmetric methane
        atomic_stdev = self.model.atomic_stdev(
            (self.species, self.coordinates)
        ).stdev_atomic_energies
        _, atomic_energies = self.model.atomic_energies(
            (self.species, self.coordinates), ensemble_values=True
        )
        stdev_atomic_energies = atomic_energies.std(0)
        self.assertEqual(stdev_atomic_energies, atomic_stdev)

        # Asymmetric methane
        ch4_coord = torch.tensor(
            [
                [
                    [4.9725e-04, -2.3656e-02, -4.6554e-02],
                    [-9.4934e-01, -4.6713e-01, -2.1225e-01],
                    [-2.1828e-01, 6.4611e-01, 8.7319e-01],
                    [3.7291e-01, 6.5190e-01, -6.9571e-01],
                    [7.9173e-01, -6.8895e-01, 3.1410e-01],
                ]
            ],
            dtype=torch.double,
            device=self.device,
        )
        _, _, atomic_qbc = self.model.atomic_stdev((self.species, ch4_coord))
        _, atomic_energies = self.model.atomic_energies(
            (self.species, ch4_coord), ensemble_values=True
        )

        stdev_atomic_energies = atomic_energies.std(0, unbiased=True)
        self.assertEqual(stdev_atomic_energies, atomic_qbc)


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

    def testMembersForces(self):
        # Symmetric methane
        expect_forces = self.model.members_forces(
            (self.species, self.coordinates),
        ).forces
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
        _forces = torch.stack(forces_list, dim=0)
        self.assertEqual(expect_forces, _forces)

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

    def testForceMagnitudes(self):
        # This test imperfectly checks that force_magnitudes works as it
        # is intended to; however, dividing by the mean magnitude in
        # near-equilibrium geometries can lead to issues
        ch4_coord = torch.tensor(
            [
                [
                    [4.9725e-04, -2.3656e-02, -4.6554e-02],
                    [-9.4934e-01, -4.6713e-01, -2.1225e-01],
                    [-2.1828e-01, 6.4611e-01, 8.7319e-01],
                    [3.7291e-01, 6.5190e-01, -6.9571e-01],
                    [7.9173e-01, -6.8895e-01, 3.1410e-01],
                ]
            ],
            dtype=torch.double,
            device=self.device,
        )
        _, magnitudes = self.model.force_magnitudes(
            (self.species, ch4_coord),
            ensemble_values=True,
        )
        _, _, _members_forces = self.model.members_forces((self.species, ch4_coord))
        _magnitudes = _members_forces.norm(dim=-1)
        self.assertEqual(magnitudes, _magnitudes)

    def testForceQBC(self):
        # Same as above test case, checks that this works for asymmetrical
        # geometry Also note that ensemble_values=True for force_qbc and
        # force_magnitudes
        ch4_coord = torch.tensor(
            [
                [
                    [4.9725e-04, -2.3656e-02, -4.6554e-02],
                    [-9.4934e-01, -4.6713e-01, -2.1225e-01],
                    [-2.1828e-01, 6.4611e-01, 8.7319e-01],
                    [3.7291e-01, 6.5190e-01, -6.9571e-01],
                    [7.9173e-01, -6.8895e-01, 3.1410e-01],
                ]
            ],
            dtype=torch.double,
            device=self.device,
        )
        _, magnitudes, relative_stdev, relative_range = self.model.force_qbc(
            (self.species, ch4_coord),
            ensemble_values=True,
        )
        _, _magnitudes = self.model.force_magnitudes(
            (self.species, ch4_coord), ensemble_values=True
        )
        _max_mag = _magnitudes.max(dim=0).values
        _min_mag = _magnitudes.min(dim=0).values
        _mean_magnitudes = _magnitudes.mean(0)
        _relative_stdev = (_magnitudes.std(0, unbiased=True) + 1e-8) / (
            _mean_magnitudes + 1e-8
        )
        _relative_range = ((_max_mag - _min_mag) + 1e-8) / (_mean_magnitudes + 1e-8)
        self.assertEqual(magnitudes, _magnitudes)
        self.assertEqual(relative_range, _relative_range)
        self.assertEqual(relative_stdev, _relative_stdev)


if __name__ == "__main__":
    unittest.main(verbosity=2)
