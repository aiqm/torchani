# type: ignore
# This file tests legacy API, it should not by type checked
import unittest
import os

import torch

from torchani._testing import TestCase
from torchani.utils import EnergyShifter
from torchani import legacy_data

batch_size = 256
path = os.path.dirname(os.path.realpath(__file__))
dataset_path = os.path.join(path, "../dataset/ani-1x/sample.h5")
ani1x_sae_dict = {
    "H": -0.60095298,
    "C": -38.08316124,
    "N": -54.7077577,
    "O": -75.19446356,
}


class TestData(TestCase):
    def testTensorShape(self):
        ds = (
            legacy_data.load(dataset_path)
            .subtract_self_energies(ani1x_sae_dict)
            .species_to_indices()
            .shuffle()
            .collate(batch_size)
            .cache()
        )
        for d in ds:
            species = d["species"]
            coordinates = d["coordinates"]
            energies = d["energies"]
            self.assertEqual(len(species.shape), 2)
            self.assertLessEqual(species.shape[0], batch_size)
            self.assertEqual(len(coordinates.shape), 3)
            self.assertEqual(coordinates.shape[2], 3)
            self.assertEqual(coordinates.shape[:2], species.shape[:2])
            self.assertEqual(len(energies.shape), 1)
            self.assertEqual(coordinates.shape[0], energies.shape[0])

    def testNoUnnecessaryPadding(self):
        ds = (
            legacy_data.load(dataset_path)
            .subtract_self_energies(ani1x_sae_dict)
            .species_to_indices()
            .shuffle()
            .collate(batch_size)
            .cache()
        )
        for d in ds:
            species = d["species"]
            non_padding = (species >= 0)[:, -1].nonzero()
            self.assertGreater(non_padding.numel(), 0)

    def testReEnter(self):
        # make sure that a dataset can be iterated multiple times
        ds = legacy_data.load(dataset_path)
        for _ in ds:
            pass
        entered = False
        for d in ds:
            entered = True
        self.assertTrue(entered)

        ds = ds.subtract_self_energies(ani1x_sae_dict)
        entered = False
        for d in ds:
            entered = True
        self.assertTrue(entered)
        entered = False
        for d in ds:
            entered = True
        self.assertTrue(entered)

        ds = ds.species_to_indices()
        entered = False
        for d in ds:
            entered = True
        self.assertTrue(entered)
        entered = False
        for d in ds:
            entered = True
        self.assertTrue(entered)

        ds = ds.shuffle()
        entered = False
        for d in ds:
            entered = True
        self.assertTrue(entered)
        entered = False
        for d in ds:
            entered = True
        self.assertTrue(entered)

        ds = ds.collate(batch_size)
        entered = False
        for d in ds:
            entered = True
        self.assertTrue(entered)
        entered = False
        for d in ds:
            entered = True
        self.assertTrue(entered)

        ds = ds.cache()
        entered = False
        for d in ds:
            entered = True
        self.assertTrue(entered)
        entered = False
        for d in ds:
            entered = True
        self.assertTrue(entered)

    def testShapeInference(self):
        shifter = EnergyShifter(None)
        ds = legacy_data.load(dataset_path).subtract_self_energies(shifter)
        len(ds)
        ds = ds.species_to_indices()
        len(ds)
        ds = ds.shuffle()
        len(ds)
        ds = ds.collate(batch_size)
        len(ds)

    def testSAE(self):
        shifter = EnergyShifter(None)
        legacy_data.load(dataset_path).subtract_self_energies(shifter)
        true_self_energies = torch.tensor(
            [
                -19.354171758844188,
                -19.354171758844046,
                -54.712238523648587,
                -75.162829556770987,
            ],
            dtype=torch.float64,
        )
        self.assertEqual(true_self_energies, shifter.self_energies)

    def testDataloader(self):
        shifter = EnergyShifter(None)
        dataset = list(
            legacy_data.load(dataset_path)
            .subtract_self_energies(shifter)
            .species_to_indices()
            .shuffle()
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=legacy_data.collate_fn,
            num_workers=0,
        )
        for _ in loader:
            pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
