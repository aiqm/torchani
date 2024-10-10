import typing as tp
import unittest

import torch
from torch import Tensor

import torchani
from torchani.utils import SYMBOLS_1X
from torchani.testing import ANITest, expand
from torchani.potentials import RepulsionXTB, StandaloneRepulsionXTB
from torchani.neighbors import NeighborData


@expand()
class TestRepulsion(ANITest):
    def setUp(self):
        self.rep = self._setup(RepulsionXTB(symbols=SYMBOLS_1X, cutoff=5.2))
        self.sa_rep = self._setup(
            StandaloneRepulsionXTB(symbols=SYMBOLS_1X, cutoff=5.2)
        )

    def testPotential(self):
        element_idxs = torch.tensor([[0, 0]], device=self.device)
        energies = torch.tensor([0.0], device=self.device)
        neighbors = NeighborData(
            indices=torch.tensor([[0], [1]], device=self.device),
            distances=torch.tensor([3.5], device=self.device),
            diff_vectors=torch.tensor([[3.5, 0, 0]], device=self.device),
        )
        energies = self.rep(element_idxs, neighbors)
        self.assertEqual(torch.tensor([3.5325e-08], device=self.device), energies)

    def testStandalone(self):
        coordinates = torch.tensor(
            [[0.0, 0.0, 0.0], [3.5, 0.0, 0.0]], device=self.device
        ).unsqueeze(0)
        species = torch.tensor([[1, 1]], device=self.device)
        energies = self.sa_rep((species, coordinates)).energies
        self.assertEqual(torch.tensor([3.5325e-08], device=self.device), energies)

    def testBatches(self):
        rep = self.sa_rep
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

        energy1 = rep((species1, coordinates1)).energies
        # avoid first atom since it isdummy
        energy2 = rep((species2[:, 1:], coordinates2[:, 1:, :])).energies
        energy3 = rep((species3[:, 1:], coordinates3[:, 1:, :])).energies
        energies_cat = torch.cat((energy1, energy2, energy3))
        energies = rep((species_cat, coordinates_cat)).energies
        self.assertEqual(energies, energies_cat)

    def testLongDistances(self):
        element_idxs = torch.tensor([[0, 0]], device=self.device)
        energies = torch.tensor([0.0], device=self.device)
        neighbors = NeighborData(
            indices=torch.tensor([[0], [1]], device=self.device),
            distances=torch.tensor([6.0], device=self.device),
            diff_vectors=torch.tensor([[6.0, 0, 0]], device=self.device),
        )
        energies = self.rep(element_idxs, neighbors)
        self.assertEqual(torch.tensor([0.0], device=self.device), energies)

    def testAtomicEnergy(self):
        model = self._setup(torchani.models.ANIdr(model_index=0, pretrained=True))
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
                model.atomic_energies((species, coordinates), ensemble_average=True)
                .energies.sum(-1)
                .item()
            )
            _energies.append(model((species, coordinates)).energies.item())
        self.assertEqual(_atomic_energies, _energies)


if __name__ == "__main__":
    unittest.main(verbosity=2)
