from pathlib import Path
import unittest
import pickle

import torch

from torchani._testing import ANITestCase, expand
from torchani.models import ANI1x
from torchani.grad import energies_and_forces, forces
from torchani.nn import ANINetworks, AtomicNetwork, BmmEnsemble, Ensemble


@expand()
class TestEnsemble(ANITestCase):
    def setUp(self):
        model = ANI1x(periodic_table_index=False)
        self.ensemble = self._setup(model)
        self.model_list = [self._setup(model[j]) for j in range(len(model))]
        self.num_conformers = 10
        self.file_path = (Path(__file__).resolve().parent / "resources") / "ANI1_subset"

    def testGDB(self):
        for i in range(self.num_conformers):
            with open(self.file_path / str(i), "rb") as f:
                coordinates, species, _, _, _, _ = pickle.load(f)
                coordinates = torch.tensor(coordinates, device=self.device)
                species = torch.tensor(species, device=self.device, dtype=torch.long)
                energies_mean, forces_mean = energies_and_forces(
                    self.ensemble, species, coordinates
                )
                coordinates.requires_grad_(True)
                energies_expect = sum(
                    [m((species, coordinates)).energies for m in self.model_list]
                ) / len(self.model_list)
                forces_expect = forces(energies_expect, coordinates)

                self.assertEqual(energies_mean, energies_expect)
                self.assertEqual(forces_mean, forces_expect)


@expand()
class TestBmmEnsembleActiveMembers(ANITestCase):
    def setUp(self):
        torch.manual_seed(0)
        symbols = ("H", "C")
        self.members = [
            ANINetworks({s: AtomicNetwork((8, 4, 3, 1), bias=True) for s in symbols})
            for _ in range(4)
        ]
        self.bmm = self._setup(BmmEnsemble(Ensemble(self.members)))
        self.elem_idxs = torch.tensor([[0, 1, 1, 0, 1]], device=self.device)
        self.aevs = torch.randn(1, 5, 8, device=self.device)

    def testPrefixSelectionIsAccepted(self):
        self.bmm.set_active_members([0, 1])
        self.assertEqual(list(self.bmm.active_members_idxs), [0, 1])

    def testPrefixSelectionAveragesFirstMembers(self):
        # eager BmmEnsemble.forward calls torch.cuda.nvtx, absent in CPU-only builds
        if not (self.jit or torch.cuda.is_available()):
            self.skipTest("eager BmmEnsemble.forward needs a CUDA build of torch")
        for n in (1, 2, 3, 4):
            self.bmm.set_active_members(list(range(n)))
            expect = Ensemble(self.members[:n]).to(self.device)(
                self.elem_idxs, self.aevs
            )
            self.assertEqual(self.bmm(self.elem_idxs, self.aevs), expect)

    def testNonPrefixSelectionRaises(self):
        with self.assertRaisesRegex(Exception, "idxs must be"):
            self.bmm.set_active_members([1, 0])
        with self.assertRaisesRegex(Exception, "idxs must be"):
            self.bmm.set_active_members([1, 2])

    def testOutOfRangeRaises(self):
        with self.assertRaisesRegex(Exception, "should be 0 <= idx"):
            self.bmm.set_active_members([0, 4])



if __name__ == "__main__":
    unittest.main(verbosity=2)
