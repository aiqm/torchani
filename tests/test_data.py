import os
import torchani
import unittest

path = os.path.dirname(os.path.realpath(__file__))
dataset_path = os.path.join(path, 'dataset/ani-1x/sample.h5')
batch_size = 256
ani1x = torchani.models.ANI1x()
consts = ani1x.consts
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


if __name__ == '__main__':
    unittest.main()
