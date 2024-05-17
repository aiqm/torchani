from pathlib import Path
import unittest
import pickle

import torch

from torchani.testing import ANITest, expand
from torchani.models import ANI1x


@expand()
class TestEnsemble(ANITest):
    def setUp(self):
        model = ANI1x(periodic_table_index=False)
        self.ensemble = self._setup(model)
        self.model_list = [self._setup(model[j]) for j in range(len(model))]
        self.num_conformers = 10
        self.file_path = (Path(__file__).resolve().parent / "test_data") / "ANI1_subset"

    def testGDB(self):
        for i in range(self.num_conformers):
            with open(self.file_path / str(i), "rb") as f:
                coordinates, species, _, _, _, _ = pickle.load(f)
                coordinates = torch.tensor(
                    coordinates, requires_grad=True, device=self.device
                )
                species = torch.tensor(species, device=self.device, dtype=torch.long)
                coordinates.requires_grad_(True)

                energy_mean = self.ensemble((species, coordinates)).energies
                energy_expect = sum(
                    [m((species, coordinates)).energies for m in self.model_list]
                ) / len(self.model_list)
                self.assertEqual(energy_mean, energy_expect)

                force_mean = torch.autograd.grad(energy_mean.sum(), coordinates)[0]
                force_expect = torch.autograd.grad(energy_expect.sum(), coordinates)[0]
                self.assertEqual(force_mean, force_expect)


if __name__ == "__main__":
    unittest.main(verbosity=2)
