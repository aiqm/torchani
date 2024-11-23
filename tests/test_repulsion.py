import typing as tp
import unittest

import torch
from torch import Tensor

import torchani
from torchani.utils import SYMBOLS_1X
from torchani._testing import ANITestCase, expand
from torchani.potentials import RepulsionXTB
from torchani.neighbors import Neighbors


@expand()
class TestRepulsion(ANITestCase):
    def setUp(self):
        self.rep = self._setup(RepulsionXTB(symbols=SYMBOLS_1X, cutoff=5.2))

    def testPotential(self):
        element_idxs = torch.tensor([[0, 0]], device=self.device)
        energies = torch.tensor([0.0], device=self.device)
        coords = torch.tensor([[[0, 0, 0], [3.5, 0, 0]]], device=self.device)
        neighbors = Neighbors(
            indices=torch.tensor([[0], [1]], device=self.device),
            distances=torch.tensor([3.5], device=self.device),
            diff_vectors=torch.tensor([[3.5, 0, 0]], device=self.device),
        )
        energies = self.rep.compute_from_neighbors(
            element_idxs, coords, neighbors
        ).energies
        self.assertEqual(torch.tensor([3.5325e-08], device=self.device), energies)

    def testStandalone(self):
        if self.jit:
            self.skipTest("calc is non-jittable")
        coordinates = torch.tensor(
            [[0.0, 0.0, 0.0], [3.5, 0.0, 0.0]], device=self.device
        ).unsqueeze(0)
        species = torch.tensor([[1, 1]], device=self.device)
        energies = self.rep(species, coordinates)
        self.assertEqual(torch.tensor([3.5325e-08], device=self.device), energies)

    def testBatches(self):
        if self.jit:
            self.skipTest("calc is non-jittable")
        coordinates1 = torch.tensor(
            [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]],
            device=self.device,
        ).unsqueeze(0)
        coordinates2 = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.5, 0.0, 0.0]],
            device=self.device,
        ).unsqueeze(0)
        coordinates3 = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [3.5, 0.0, 0.0]],
            device=self.device,
        ).unsqueeze(0)
        species1 = torch.tensor([[1, 6, 7]], device=self.device)
        species2 = torch.tensor([[-1, 1, 6]], device=self.device)
        species3 = torch.tensor([[-1, 1, 1]], device=self.device)
        coordinates_cat = torch.cat((coordinates1, coordinates2, coordinates3), dim=0)
        species_cat = torch.cat((species1, species2, species3), dim=0)

        energy1 = self.rep(species1, coordinates1)
        # avoid first atom since it isdummy
        energy2 = self.rep(species2[:, 1:], coordinates2[:, 1:, :])
        energy3 = self.rep(species3[:, 1:], coordinates3[:, 1:, :])
        energies_cat = torch.cat((energy1, energy2, energy3))
        energies = self.rep(species_cat, coordinates_cat)
        self.assertEqual(energies, energies_cat)

    def testLongDistances(self):
        element_idxs = torch.tensor([[0, 0]], device=self.device)
        energies = torch.tensor([0.0], device=self.device)
        coords = torch.tensor([[[0, 0, 0], [6.0, 0, 0]]], device=self.device)
        neighbors = Neighbors(
            indices=torch.tensor([[0], [1]], device=self.device),
            distances=torch.tensor([6.0], device=self.device),
            diff_vectors=torch.tensor([[6.0, 0, 0]], device=self.device),
        )
        energies = self.rep.compute_from_neighbors(
            element_idxs, coords, neighbors
        ).energies
        self.assertEqual(torch.tensor([0.0], device=self.device), energies)

    def testAtomicEnergy(self):
        model = self._setup(torchani.models.ANIdr(model_index=0))
        species = torch.tensor([[8, 1, 1]], device=self.device)
        _energies: tp.List[Tensor] = []
        _atomic_energies: tp.List[Tensor] = []
        distances = torch.linspace(0.1, 6.0, 100, device=self.device)
        for d in distances:
            coordinates = torch.tensor(
                [
                    [
                        [0.0, 0.0, 0.0],
                        [0.97, 0.0, 0.0],
                        [-0.250380004 * d, 0.96814764 * d, 0.0],
                    ]
                ],
                requires_grad=True,
                dtype=torch.float,
                device=self.device,
            )
            _atomic_energies.append(
                model.atomic_energies((species, coordinates)).energies.sum(-1).item()
            )
            _energies.append(model((species, coordinates)).energies.item())
        self.assertEqual(_atomic_energies, _energies)


if __name__ == "__main__":
    unittest.main(verbosity=2)
