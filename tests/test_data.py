import os
import torchani
import unittest

path = os.path.dirname(os.path.realpath(__file__))
dataset_path = os.path.join(path, '../dataset')
print(dataset_path)
batch_size = 256
aev = torchani.AEVComputer()


class TestData(unittest.TestCase):

    def setUp(self):
        self.ds = torchani.training.BatchedANIDataset(dataset_path,
                                                      aev.species,
                                                      batch_size)

    def testTensorShape(self):
        for i in self.ds:
            input, output = i
            species, coordinates = input
            energies = output['energies']
            self.assertEqual(len(species.shape), 2)
            self.assertLessEqual(species.shape[0], batch_size)
            self.assertEqual(len(coordinates.shape), 3)
            self.assertEqual(coordinates.shape[2], 3)
            self.assertEqual(coordinates.shape[1], coordinates.shape[1])
            self.assertEqual(coordinates.shape[0], coordinates.shape[0])
            self.assertEqual(len(energies.shape), 1)
            self.assertEqual(coordinates.shape[0], energies.shape[0])

    def testNoUnnecessaryPadding(self):
        for i in self.ds:
            input, _ = i
            species, _ = input
            non_padding = (species >= 0)[:, -1].nonzero()
            self.assertGreater(non_padding.numel(), 0)


if __name__ == '__main__':
    unittest.main()
