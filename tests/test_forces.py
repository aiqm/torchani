from pathlib import Path
import unittest
import pickle

import torch

from torchani.models import ANI1x
from torchani._testing import ANITestCase, expand
from torchani.utils import pad_atomic_properties
from torchani.grad import energies_and_forces


@expand()
class TestForce(ANITestCase):
    def setUp(self):
        self.model = self._setup(ANI1x(model_index=0, periodic_table_index=False))
        self.num_conformers = 50
        self.file_path = (Path(__file__).resolve().parent / "resources") / "ANI1_subset"

    def testIsomers(self):
        for i in range(self.num_conformers):
            with open(self.file_path / str(i), "rb") as f:
                coordinates, species, _, _, _, forces = pickle.load(f)
                coordinates = torch.tensor(coordinates, device=self.device)
                species = torch.tensor(species, device=self.device, dtype=torch.long)
                forces = torch.tensor(forces, device=self.device)
                energies, forces_pred = energies_and_forces(
                    self.model, species, coordinates
                )
                self.assertEqual(forces, forces_pred)

    def testPadding(self):
        batch = []
        for i in range(self.num_conformers):
            with open(self.file_path / str(i), "rb") as f:
                coordinates, species, _, _, _, forces = pickle.load(f)
                coordinates = torch.tensor(
                    coordinates,
                    device=self.device,
                )
                species = torch.tensor(species, device=self.device, dtype=torch.long)
                forces = torch.tensor(forces, device=self.device)
                batch.append(
                    {
                        "species": species,
                        "coordinates": coordinates,
                        "forces": forces,
                    }
                )
        padded_batch = pad_atomic_properties(batch)
        coordinates = padded_batch["coordinates"].to(self.device)
        species = padded_batch["species"].to(self.device)
        forces = padded_batch["forces"].to(self.device)
        energies, pred_forces = energies_and_forces(self.model, species, coordinates)
        self.assertEqual(pred_forces, forces)


if __name__ == "__main__":
    unittest.main(verbosity=2)
