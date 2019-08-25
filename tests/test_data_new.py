import torchani
import unittest
import pkbar
import torch
import os

path = os.path.dirname(os.path.realpath(__file__))
dspath = os.path.join(path, '../dataset/ani1-up_to_gdb4/ani_gdb_s03.h5')

batch_size = 2560
chunk_threshold = 5

other_properties = {'properties': ['dipoles', 'forces', 'energies'],
                    'padding_values': [None, 0, None],
                    'padded_shapes': [(batch_size, 3), (batch_size, -1, 3), (batch_size, )],
                    'dtypes': [torch.float32, torch.float32, torch.float64],
                    }

other_properties = {'properties': ['energies'],
                    'padding_values': [None],
                    'padded_shapes': [(batch_size, )],
                    'dtypes': [torch.float64],
                    }


class TestFindThreshold(unittest.TestCase):
    def setUp(self):
        print('.. check find threshold to split chunks')

    def testFindThreshould(self):
        torchani.data.find_threshold(dspath, batch_size=batch_size, threshold_max=10)


class TestShuffledData(unittest.TestCase):

    def setUp(self):
        print('.. setup shuffle dataset')
        self.ds = torchani.data.ShuffledDataset(dspath, batch_size=batch_size,
                                                chunk_threshold=chunk_threshold,
                                                num_workers=2,
                                                other_properties=other_properties,
                                                subtract_self_energies=True)
        self.chunks, self.properties = iter(self.ds).next()

    def testTensorShape(self):
        print('=> checking tensor shape')
        print('the first batch is ([chunk1, chunk2, ...], {"energies", "force", ...}) in which chunk1=(species, coordinates)')
        batch_len = 0
        print('1. chunks')
        for i, chunk in enumerate(self.chunks):
            print('chunk{}'.format(i + 1), 'species:', list(chunk[0].size()), chunk[0].dtype,
                  'coordinates:', list(chunk[1].size()), chunk[1].dtype)
            # check dtype
            self.assertEqual(chunk[0].dtype, torch.int64)
            self.assertEqual(chunk[1].dtype, torch.float32)
            # check shape
            self.assertEqual(chunk[1].shape[2], 3)
            self.assertEqual(chunk[1].shape[:2], chunk[0].shape[:2])
            batch_len += chunk[0].shape[0]
        print('2. properties')
        for i, key in enumerate(other_properties['properties']):
            print(key, list(self.properties[key].size()), self.properties[key].dtype)
            # check dtype
            self.assertEqual(self.properties[key].dtype, other_properties['dtypes'][i])
            # shape[0] == batch_size
            self.assertEqual(self.properties[key].shape[0], other_properties['padded_shapes'][i][0])
            # check len(shape)
            self.assertEqual(len(self.properties[key].shape), len(other_properties['padded_shapes'][i]))

    def testLoadDataset(self):
        print('=> test loading all dataset')
        pbar = pkbar.Pbar('loading and processing dataset into cpu memory, total '
                          + 'batches: {}, batch_size: {}'.format(len(self.ds), batch_size),
                          len(self.ds))
        for i, _ in enumerate(self.ds):
            pbar.update(i)

    def testSplitDataset(self):
        print('=> test splitting dataset')
        train_ds, val_ds = torchani.data.ShuffledDataset(dspath, batch_size=batch_size, chunk_threshold=chunk_threshold, num_workers=2, validation_split=0.1)
        frac = len(val_ds) / (len(val_ds) + len(train_ds))
        self.assertLess(abs(frac - 0.1), 0.05)

    def testNoUnnecessaryPadding(self):
        print('=> checking No Unnecessary Padding')
        for i, chunk in enumerate(self.chunks):
            species, _ = chunk
            non_padding = (species >= 0)[:, -1].nonzero()
            self.assertGreater(non_padding.numel(), 0)


class TestCachedData(unittest.TestCase):

    def setUp(self):
        print('.. setup cached dataset')
        self.ds = torchani.data.CachedDataset(dspath, batch_size=batch_size, device='cpu',
                                              chunk_threshold=chunk_threshold,
                                              other_properties=other_properties,
                                              subtract_self_energies=True)
        self.chunks, self.properties = self.ds[0]

    def testTensorShape(self):
        print('=> checking tensor shape')
        print('the first batch is ([chunk1, chunk2, ...], {"energies", "force", ...}) in which chunk1=(species, coordinates)')
        batch_len = 0
        print('1. chunks')
        for i, chunk in enumerate(self.chunks):
            print('chunk{}'.format(i + 1), 'species:', list(chunk[0].size()), chunk[0].dtype,
                  'coordinates:', list(chunk[1].size()), chunk[1].dtype)
            # check dtype
            self.assertEqual(chunk[0].dtype, torch.int64)
            self.assertEqual(chunk[1].dtype, torch.float32)
            # check shape
            self.assertEqual(chunk[1].shape[2], 3)
            self.assertEqual(chunk[1].shape[:2], chunk[0].shape[:2])
            batch_len += chunk[0].shape[0]
        print('2. properties')
        for i, key in enumerate(other_properties['properties']):
            print(key, list(self.properties[key].size()), self.properties[key].dtype)
            # check dtype
            self.assertEqual(self.properties[key].dtype, other_properties['dtypes'][i])
            # shape[0] == batch_size
            self.assertEqual(self.properties[key].shape[0], other_properties['padded_shapes'][i][0])
            # check len(shape)
            self.assertEqual(len(self.properties[key].shape), len(other_properties['padded_shapes'][i]))

    def testLoadDataset(self):
        print('=> test loading all dataset')
        self.ds.load()

    def testSplitDataset(self):
        print('=> test splitting dataset')
        train_dataset, val_dataset = self.ds.split(0.1)
        frac = len(val_dataset) / len(self.ds)
        self.assertLess(abs(frac - 0.1), 0.05)

    def testNoUnnecessaryPadding(self):
        print('=> checking No Unnecessary Padding')
        for i, chunk in enumerate(self.chunks):
            species, _ = chunk
            non_padding = (species >= 0)[:, -1].nonzero()
            self.assertGreater(non_padding.numel(), 0)


if __name__ == "__main__":
    unittest.main()
