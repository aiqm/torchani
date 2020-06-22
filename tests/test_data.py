import os
import torch
import torchani
import unittest

path = os.path.dirname(os.path.realpath(__file__))
dataset_path = os.path.join(path, '../dataset/ani-1x/sample.h5')
batch_size = 256
ani1x = torchani.models.ANI1x()
sae_dict = ani1x.sae_dict
aev_computer = ani1x.aev_computer


class TestData(unittest.TestCase):

    def testTensorShape(self):
        ds = torchani.data.load(dataset_path).subtract_self_energies(sae_dict).species_to_indices().shuffle().collate(batch_size).cache()
        for d in ds:
            species = d['species']
            coordinates = d['coordinates']
            energies = d['energies']
            self.assertEqual(len(species.shape), 2)
            self.assertLessEqual(species.shape[0], batch_size)
            self.assertEqual(len(coordinates.shape), 3)
            self.assertEqual(coordinates.shape[2], 3)
            self.assertEqual(coordinates.shape[:2], species.shape[:2])
            self.assertEqual(len(energies.shape), 1)
            self.assertEqual(coordinates.shape[0], energies.shape[0])

    def testNoUnnecessaryPadding(self):
        ds = torchani.data.load(dataset_path).subtract_self_energies(sae_dict).species_to_indices().shuffle().collate(batch_size).cache()
        for d in ds:
            species = d['species']
            non_padding = (species >= 0)[:, -1].nonzero()
            self.assertGreater(non_padding.numel(), 0)

    def testReEnter(self):
        # make sure that a dataset can be iterated multiple times
        ds = torchani.data.load(dataset_path)
        for d in ds:
            pass
        entered = False
        for d in ds:
            entered = True
        self.assertTrue(entered)

        ds = ds.subtract_self_energies(sae_dict)
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
            pass
        self.assertTrue(entered)
        entered = False
        for d in ds:
            entered = True
        self.assertTrue(entered)

        ds = ds.collate(batch_size)
        entered = False
        for d in ds:
            entered = True
            pass
        self.assertTrue(entered)
        entered = False
        for d in ds:
            entered = True
        self.assertTrue(entered)

        ds = ds.cache()
        entered = False
        for d in ds:
            entered = True
            pass
        self.assertTrue(entered)
        entered = False
        for d in ds:
            entered = True
        self.assertTrue(entered)

    def testShapeInference(self):
        shifter = torchani.EnergyShifter(None)
        ds = torchani.data.load(dataset_path).subtract_self_energies(shifter)
        len(ds)
        ds = ds.species_to_indices()
        len(ds)
        ds = ds.shuffle()
        len(ds)
        ds = ds.collate(batch_size)
        len(ds)

    def testSAE(self):
        tolerance = 1e-5
        shifter = torchani.EnergyShifter(None)
        torchani.data.load(dataset_path).subtract_self_energies(shifter)
        true_self_energies = torch.tensor([-19.354171758844188,
                                           -19.354171758844046,
                                           -54.712238523648587,
                                           -75.162829556770987], dtype=torch.float64)
        diff = torch.abs(true_self_energies - shifter.self_energies)
        for e in diff:
            self.assertLess(e, tolerance)

    def testDataloader(self):
        shifter = torchani.EnergyShifter(None)
        dataset = list(torchani.data.load(dataset_path).subtract_self_energies(shifter).species_to_indices().shuffle())
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=torchani.data.collate_fn, num_workers=64)
        for i in loader:
            pass


if __name__ == '__main__':
    unittest.main()
