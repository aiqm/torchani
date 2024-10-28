from pathlib import Path
import unittest
import pickle

import torch

from torchani._testing import ANITestCase, expand
from torchani.models import ANI1x
from torchani.grad import energies_and_forces, forces


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
