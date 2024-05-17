from pathlib import Path
import unittest
import pickle

import torch

from torchani.models import ANI1x
from torchani.testing import ANITest, expand
from torchani.utils import pad_atomic_properties, broadcast_first_dim


@expand()
class TestForce(ANITest):
    def setUp(self):
        self.model = self._setup(ANI1x(model_index=0, periodic_table_index=False))
        self.num_conformers = 50
        self.file_path = (Path(__file__).resolve().parent / "test_data") / "ANI1_subset"

    def testIsomers(self):
        for i in range(self.num_conformers):
            with open(self.file_path / str(i), "rb") as f:
                coordinates, species, _, _, _, forces = pickle.load(f)
                coordinates = torch.tensor(
                    coordinates, device=self.device, requires_grad=True
                )
                species = torch.tensor(species, device=self.device, dtype=torch.long)
                forces = torch.tensor(forces, device=self.device)
                _, energies = self.model((species, coordinates))
                derivative = torch.autograd.grad(energies.sum(), coordinates)[0]
                self.assertEqual(forces, -derivative)

    def testPadding(self):
        batch = []
        for i in range(self.num_conformers):
            with open(self.file_path / str(i), "rb") as f:
                coordinates, species, _, _, _, forces = pickle.load(f)
                coordinates = torch.tensor(
                    coordinates, device=self.device, requires_grad=True
                )
                species = torch.tensor(species, device=self.device, dtype=torch.long)
                forces = torch.tensor(forces, device=self.device)
                batch.append(
                    broadcast_first_dim(
                        {
                            "species": species,
                            "coordinates": coordinates,
                            "forces": forces,
                        }
                    )
                )
        padded_batch = pad_atomic_properties(batch)
        coordinates = padded_batch["coordinates"].to(self.device)
        species = padded_batch["species"].to(self.device)
        forces = padded_batch["forces"].to(self.device)
        energies = self.model((species, coordinates)).energies
        pred_forces = -torch.autograd.grad(
            energies.sum(),
            coordinates,
        )[0]
        self.assertEqual(pred_forces, forces)


if __name__ == "__main__":
    unittest.main(verbosity=2)
