import unittest
import torch
import torchani
from pathlib import Path
from torchani.models import _fetch_state_dict
from torchani.testing import TestCase
from torchani.repulsion import RepulsionXTB, StandaloneRepulsionXTB


class TestRepulsion(TestCase):
    def setUp(self):
        self.rep = RepulsionXTB(5.2)
        self.sa_rep = StandaloneRepulsionXTB(cutoff=5.2, neighborlist_cutoff=5.2)

    def testRepulsionXTB(self):
        neighbor_idxs = torch.tensor([[0], [1]])
        distances = torch.tensor([3.5])
        element_idxs = torch.tensor([[0, 0]])
        energies = torch.tensor([0.0])
        energies = self.rep(element_idxs, neighbor_idxs, distances)
        self.assertEqual(torch.tensor([3.5325e-08]), energies)

    def testStandalone(self):
        coordinates = torch.tensor([[0.0, 0.0, 0.0],
                                    [3.5, 0.0, 0.0]]).unsqueeze(0)
        species = torch.tensor([[1, 1]])
        energies = self.sa_rep((species, coordinates)).energies
        self.assertEqual(torch.tensor([3.5325e-08]), energies)

    def testRepulsionBatches(self):
        rep = self.sa_rep
        coordinates1 = torch.tensor([[0.0, 0.0, 0.0],
                                    [1.5, 0.0, 0.0],
                                    [3.0, 0.0, 0.0]]).unsqueeze(0)
        coordinates2 = torch.tensor([[0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0],
                                    [2.5, 0.0, 0.0]]).unsqueeze(0)
        coordinates3 = torch.tensor([[0.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0],
                                     [3.5, 0.0, 0.0]]).unsqueeze(0)
        species1 = torch.tensor([[1, 6, 7]])
        species2 = torch.tensor([[-1, 1, 6]])
        species3 = torch.tensor([[-1, 1, 1]])
        coordinates_cat = torch.cat((coordinates1, coordinates2, coordinates3), dim=0)
        species_cat = torch.cat((species1, species2, species3), dim=0)

        energy1 = rep((species1, coordinates1)).energies
        # avoid first atom since it isdummy
        energy2 = rep((species2[:, 1:], coordinates2[:, 1:, :])).energies
        energy3 = rep((species3[:, 1:], coordinates3[:, 1:, :])).energies
        energies_cat = torch.cat((energy1, energy2, energy3))
        energies = rep((species_cat, coordinates_cat)).energies
        self.assertEqual(energies, energies_cat)

    def testRepulsionLongDistances(self):
        neighbor_idxs = torch.tensor([[0], [1]])
        distances = torch.tensor([6.0])
        element_idxs = torch.tensor([[0, 0]])
        energies = torch.tensor([0.0])
        energies = self.rep(element_idxs, neighbor_idxs, distances)
        self.assertEqual(torch.tensor([0.0]), energies)

    def testRepulsionEnergy(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = torchani.models.ANI1x(
            repulsion=True,
            pretrained=False,
            model_index=0,
            cutoff_fn='smooth'
        )
        model.load_state_dict(_fetch_state_dict('ani1x_state_dict.pt', 0), strict=False)
        model = model.to(device=device, dtype=torch.double)
        self._testRepulsionEnergy(model, device)

    def _testRepulsionEnergy(self, model, device):
        species = torch.tensor([[8, 1, 1]], device=device)
        energies = []
        distances = torch.linspace(0.1, 6.0, 100)
        for d in distances:
            coordinates = torch.tensor([[[0.0, 0.0, 0.0],
                                        [0.97, 0.0, 0.0],
                                        [-0.250380004 * d, 0.96814764 * d, 0.0]]],
                                       requires_grad=True, device=device, dtype=torch.double)
            energies.append(model((species, coordinates)).energies.item())
        energies = torch.tensor(energies)
        path = Path(__file__).resolve().parent.joinpath('test_data/energies_repulsion_1x.pkl')
        with open(path, 'rb') as f:
            energies_expect = torch.load(f)
        self.assertEqual(energies_expect, energies)


class TestRepulsionJIT(TestRepulsion):
    # JIT compile and repeat all tests except Repulsion Energy
    def setUp(self):
        super().setUp()
        self.rep = torch.jit.script(self.rep)
        self.sa_rep = torch.jit.script(self.sa_rep)

    def testRepulsionEnergy(self):
        device = torch.device('cpu')
        model = torchani.models.ANI1x(
            repulsion=True,
            pretrained=False,
            model_index=0,
            cutoff_fn='smooth'
        )
        model.load_state_dict(_fetch_state_dict('ani1x_state_dict.pt', 0), strict=False)
        model = torch.jit.script(model)
        model = model.to(device=device, dtype=torch.double)
        self._testRepulsionEnergy(model, device)


if __name__ == '__main__':
    unittest.main()
