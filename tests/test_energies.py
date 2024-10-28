from pathlib import Path
import unittest
import pickle

import torch

from torchani._testing import ANITestCase, expand
from torchani.utils import pad_atomic_properties
from torchani.models import ANI1x, ANI2x, ANIdr
from torchani.units import hartree2kcalpermol


@expand()
class TestANI2x(ANITestCase):
    def setUp(self):
        self.model_pti = self._setup(ANI2x(model_index=0))
        self.model = self._setup(ANI2x(model_index=0, periodic_table_index=False))

    def testDiatomics(self):
        coordinates = torch.tensor(
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]], device=self.device
        )
        coordinates = coordinates.repeat(4, 1, 1)
        # F2, S2, O2, Cl2
        species_pti = torch.tensor(
            [[9, 9], [16, 16], [8, 8], [17, 17]], device=self.device, dtype=torch.long
        )
        # in 2x the species are not in periodic table order unfortunately
        species = torch.tensor(
            [[5, 5], [4, 4], [3, 3], [6, 6]], device=self.device, dtype=torch.long
        )
        e_pti = self.model_pti((species_pti, coordinates)).energies
        e = self.model((species, coordinates)).energies
        self.assertEqual(e_pti, e)

        # compare against 2x energies calculated directly from original model
        e = hartree2kcalpermol(e)
        e_expect = torch.tensor(
            [-125100.7729, -499666.2354, -94191.3460, -577504.1792], device=self.device
        )
        self.assertEqual(e_expect, e)


@expand()
class TestANIdr(ANITestCase):
    def setUp(self):
        self.model = self._setup(ANIdr(model_index=0))

    def testDiatomics(self):
        coordinates = torch.tensor(
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]], dtype=torch.float, device=self.device
        )
        coordinates = coordinates.repeat(4, 1, 1)
        # F2, S2, O2, Cl2
        species = torch.tensor(
            [[9, 9], [16, 16], [8, 8], [17, 17]], dtype=torch.long, device=self.device
        )
        e = self.model((species, coordinates)).energies

        e = hartree2kcalpermol(e)
        e_expect = torch.tensor(
            [-125122.7685, -499630.9805, -94078.2276, -577468.0107],
            device=self.device,
            dtype=torch.float,
        )
        self.assertEqual(e_expect, e)


@expand(device="cpu", jit=False)
class TestCorrectInput(ANITestCase):
    def setUp(self):
        model = ANI1x(model_index=0, periodic_table_index=False)
        self.model = self._setup(model)
        self.converter = self._setup(model.species_converter)
        self.aev_computer = self._setup(model.aev_computer)
        self.ani_networks = self._setup(model.neural_networks)

    def testUnknownSpecies(self):
        # unsupported atomic number raises a value error
        with self.assertRaises(ValueError):
            _ = self.converter(torch.tensor([[1, 1, 7, 10]]))

        # larger index than supported by the model raises a value error
        with self.assertRaises(ValueError):
            _ = self.model((torch.tensor([[0, 1, 2, 4]]), torch.zeros((1, 4, 3))))

    def testIncorrectShape(self):
        # non matching shapes between species and coordinates
        self.assertRaises(
            AssertionError,
            self.model,
            (torch.tensor([[0, 1, 2, 3]]), torch.zeros((1, 3, 3))),
        )
        self.assertRaises(
            AssertionError,
            self.aev_computer,
            torch.tensor([[0, 1, 2, 3]]),
            torch.zeros((1, 3, 3)),
        )
        self.assertRaises(
            AssertionError,
            self.ani_networks,
            torch.tensor([[0, 1, 2, 3]]),
            torch.zeros((1, 3, 384)),
        )
        self.assertRaises(
            AssertionError,
            self.model,
            (torch.tensor([[0, 1, 2, 3]]), torch.zeros((1, 4, 4))),
        )
        self.assertRaises(
            AssertionError,
            self.model,
            (torch.tensor([0, 1, 2, 3]), torch.zeros((4, 3))),
        )


@expand()
class TestEnergies(ANITestCase):
    def setUp(self):
        self.model = self._setup(ANI1x(model_index=0, periodic_table_index=False))
        self.num_conformers = 50
        self.file_path = (Path(__file__).resolve().parent / "resources") / "ANI1_subset"

    def testIsomers(self):
        for i in range(self.num_conformers):
            with open(self.file_path / str(i), "rb") as f:
                coordinates, species, _, _, energies, _ = pickle.load(f)
                coordinates = torch.tensor(
                    coordinates, dtype=torch.float, device=self.device
                )
                species = torch.tensor(species, device=self.device, dtype=torch.long)
                energies = torch.tensor(energies, dtype=torch.float, device=self.device)
                energies_ = self.model((species, coordinates)).energies
                self.assertEqual(energies, energies_)

    def testPadding(self):
        batch = []
        for i in range(self.num_conformers):
            with open(self.file_path / str(i), "rb") as f:
                coordinates, species, _, _, energies, _ = pickle.load(f)
                coordinates = torch.tensor(
                    coordinates, dtype=torch.float, device=self.device
                )
                species = torch.tensor(species, device=self.device, dtype=torch.long)
                energies = torch.tensor(energies, dtype=torch.float, device=self.device)
                batch.append(
                    {
                        "species": species,
                        "coordinates": coordinates,
                        "energies": energies,
                    }
                )
        padded_batch = pad_atomic_properties(batch)
        species = padded_batch["species"]
        coordinates = padded_batch["coordinates"]
        energies_expect = padded_batch["energies"]
        energies = self.model((species, coordinates)).energies
        self.assertEqual(energies, energies_expect)


if __name__ == "__main__":
    unittest.main(verbosity=2)
