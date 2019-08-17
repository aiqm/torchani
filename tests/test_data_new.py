import torchani
import unittest
import pkbar
import torch

# TODO no calculation is need for split chunks, use probability to split

dspath = '/home/richard/dev/torchani/download/dataset/ani-1x/ANI-1x_complete.h5'
# dspath = '/home/richard/dev/torchani/download/dataset/ani1-up_to_gdb4/ani_gdb_s03.h5'
batch_size = 2560
bar = 5


class TestShuffleData(unittest.TestCase):

    def setUp(self):
        print('.. setup shuffle dataset')
        self.ds = torchani.data.ShuffleDataset(dspath, batch_size=batch_size, bar=bar, num_workers=2, test_bar_max=None)
        self.chunks, self.properties = iter(self.ds).next()

    def testTensorShape(self):
        print('=> checking tensor shape')
        print('the first batch is ([chunk1, chunk2, ...], {"energies", "force", ...}) in which chunk1=(species, coordinates)')
        batch_len = 0
        for i, chunk in enumerate(self.chunks):
            print('chunk{}'.format(i + 1), list(chunk[0].size()), chunk[0].dtype, list(chunk[1].size()), chunk[1].dtype)
            # check dtype
            self.assertEqual(chunk[0].dtype, torch.int64)
            self.assertEqual(chunk[1].dtype, torch.float32)
            # check shape
            self.assertEqual(chunk[1].shape[2], 3)
            self.assertEqual(chunk[1].shape[:2], chunk[0].shape[:2])
            batch_len += chunk[0].shape[0]

        for key, value in self.properties.items():
            print(key, list(value.size()), value.dtype)
            self.assertEqual(value.dtype, torch.float32)
            self.assertEqual(len(value.shape), 1)
            self.assertEqual(value.shape[0], batch_len)

    def testLoadDataset(self):
        print('=> test loading all dataset')
        pbar = pkbar.Pbar('loading and processing dataset into cpu memory, total '
                          + 'batches: {}, batch_size: {}'.format(len(self.ds), batch_size),
                          len(self.ds))
        for i, t in enumerate(self.ds):
            pbar.update(i)

    def testNoUnnecessaryPadding(self):
        print('=> checking No Unnecessary Padding')
        for i, chunk in enumerate(self.chunks):
            species, _ = chunk
            non_padding = (species >= 0)[:, -1].nonzero()
            self.assertGreater(non_padding.numel(), 0)


class TestCacheData(unittest.TestCase):

    def setUp(self):
        print('.. setup cached dataset')
        self.ds = torchani.data.CacheDataset(dspath, batch_size=batch_size, device='cpu', bar=bar, test_bar_max=None)
        self.chunks, self.properties = self.ds[0]

    def testTensorShape(self):
        print('=> checking tensor shape')
        print('the first batch is ([chunk1, chunk2, ...], {"energies", "force", ...}) in which chunk1=(species, coordinates)')
        batch_len = 0
        for i, chunk in enumerate(self.chunks):
            print('chunk{}'.format(i + 1), list(chunk[0].size()), chunk[0].dtype, list(chunk[1].size()), chunk[1].dtype)
            # check dtype
            self.assertEqual(chunk[0].dtype, torch.int64)
            self.assertEqual(chunk[1].dtype, torch.float32)
            # check shape
            self.assertEqual(chunk[1].shape[2], 3)
            self.assertEqual(chunk[1].shape[:2], chunk[0].shape[:2])
            batch_len += chunk[0].shape[0]

        for key, value in self.properties.items():
            print(key, list(value.size()), value.dtype)
            self.assertEqual(value.dtype, torch.float32)
            self.assertEqual(len(value.shape), 1)
            self.assertEqual(value.shape[0], batch_len)

    def testLoadDataset(self):
        print('=> test loading all dataset')
        pbar = pkbar.Pbar('loading and processing dataset into cpu memory, total '
                          + 'batches: {}, batch_size: {}'.format(len(self.ds), batch_size),
                          len(self.ds))
        for i, t in enumerate(self.ds):
            pbar.update(i)

    def testNoUnnecessaryPadding(self):
        print('=> checking No Unnecessary Padding')
        for i, chunk in enumerate(self.chunks):
            species, _ = chunk
            non_padding = (species >= 0)[:, -1].nonzero()
            self.assertGreater(non_padding.numel(), 0)


if __name__ == "__main__":
    unittest.main()
