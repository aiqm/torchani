import sys

if sys.version_info.major >= 3:
    import torchani
    import unittest
    import tempfile
    import os
    import torch
    import torchani.pyanitools as pyanitools
    import torchani.data
    from math import ceil
    from bisect import bisect
    from pickle import dump, load

    path = os.path.dirname(os.path.realpath(__file__))
    dataset_dir = os.path.join(path, 'dataset')

    class TestDataset(unittest.TestCase):

        def setUp(self, data_path=dataset_dir):
            self.data_path = data_path
            self.ds = torchani.data.load_dataset(data_path)

        def testLen(self):
            # compute data length using Dataset
            l1 = len(self.ds)
            # compute data lenght using pyanitools
            l2 = 0
            for f in os.listdir(self.data_path):
                f = os.path.join(self.data_path, f)
                if os.path.isfile(f) and \
                   (f.endswith('.h5') or f.endswith('.hdf5')):
                    for j in pyanitools.anidataloader(f):
                        l2 += j['energies'].shape[0]
            # compute data length using iterator
            l3 = len(list(self.ds))
            # these lengths should match
            self.assertEqual(l1, l2)
            self.assertEqual(l1, l3)

        def testNumChunks(self):
            chunksize = 64
            # compute number of chunks using batch sampler
            bs = torchani.data.BatchSampler(self.ds, chunksize, 1)
            l1 = len(bs)
            # compute number of chunks using pyanitools
            l2 = 0
            for f in os.listdir(self.data_path):
                f = os.path.join(self.data_path, f)
                if os.path.isfile(f) and \
                   (f.endswith('.h5') or f.endswith('.hdf5')):
                    for j in pyanitools.anidataloader(f):
                        conformations = j['energies'].shape[0]
                        l2 += ceil(conformations / chunksize)
            # compute number of chunks using iterator
            l3 = len(list(bs))
            # these lengths should match
            self.assertEqual(l1, l2)
            self.assertEqual(l1, l3)

        def testNumBatches(self):
            chunksize = 64
            batch_chunks = 4
            # compute number of batches using batch sampler
            bs = torchani.data.BatchSampler(self.ds, chunksize, batch_chunks)
            l1 = len(bs)
            # compute number of batches by simple math
            bs2 = torchani.data.BatchSampler(self.ds, chunksize, 1)
            l2 = ceil(len(bs2) / batch_chunks)
            # compute number of batches using iterator
            l3 = len(list(bs))
            # these lengths should match
            self.assertEqual(l1, l2)
            self.assertEqual(l1, l3)

        def testBatchSize1(self):
            bs = torchani.data.BatchSampler(self.ds, 1, 1)
            self.assertEqual(len(bs), len(self.ds))

        def testSplitSize(self):
            chunksize = 64
            bs = torchani.data.BatchSampler(self.ds, chunksize, 1)
            chunks = len(bs)
            ds1, ds2 = torchani.data.random_split(
                self.ds, [200, chunks-200], chunksize)
            bs1 = torchani.data.BatchSampler(ds1, chunksize, 1)
            bs2 = torchani.data.BatchSampler(ds2, chunksize, 1)
            self.assertEqual(len(bs1), 200)
            self.assertEqual(len(bs2), chunks-200)

        def testSplitNoOverlap(self):
            chunksize = 64
            bs = torchani.data.BatchSampler(self.ds, chunksize, 1)
            chunks = len(bs)
            ds1, ds2 = torchani.data.random_split(
                self.ds, [200, chunks-200], chunksize)
            indices1 = ds1.dataset.indices
            indices2 = ds2.dataset.indices
            self.assertEqual(len(indices1), len(ds1))
            self.assertEqual(len(indices2), len(ds2))
            self.assertEqual(len(indices1), len(set(indices1)))
            self.assertEqual(len(indices2), len(set(indices2)))
            self.assertEqual(len(self.ds), len(set(indices1+indices2)))

        def _testMolSizes(self, ds):
            for i in range(len(ds)):
                left = bisect(ds.cumulative_sizes, i)
                moli = ds[i][0].item()
                for j in range(len(ds)):
                    left2 = bisect(ds.cumulative_sizes, j)
                    molj = ds[j][0].item()
                    if left == left2:
                        self.assertEqual(moli, molj)
                    else:
                        if moli == molj:
                            print(i, j)
                        self.assertNotEqual(moli, molj)

        def testMolSizes(self):
            chunksize = 8
            bs = torchani.data.BatchSampler(self.ds, chunksize, 1)
            chunks = len(bs)
            ds1, ds2 = torchani.data.random_split(
                self.ds, [50, chunks-50], chunksize)
            self._testMolSizes(ds1)

        def testSaveLoad(self):
            chunksize = 8
            bs = torchani.data.BatchSampler(self.ds, chunksize, 1)
            chunks = len(bs)
            ds1, ds2 = torchani.data.random_split(
                self.ds, [50, chunks-50], chunksize)

            tmpdir = tempfile.TemporaryDirectory()
            tmpdirname = tmpdir.name
            filename = os.path.join(tmpdirname, 'test.obj')

            with open(filename, 'wb') as f:
                dump(ds1, f)

            with open(filename, 'rb') as f:
                ds1_loaded = load(f)

            self.assertEqual(len(ds1), len(ds1_loaded))
            self.assertListEqual(ds1.sizes, ds1_loaded.sizes)
            self.assertIsInstance(ds1_loaded, torchani.data.ANIDataset)

            for i in range(len(ds1)):
                i1 = ds1[i]
                i2 = ds1_loaded[i]
                molid1 = i1[0].item()
                molid2 = i2[0].item()
                self.assertEqual(molid1, molid2)
                xyz1 = i1[1]
                xyz2 = i2[1]
                maxdiff = torch.max(torch.abs(xyz1-xyz2)).item()
                self.assertEqual(maxdiff, 0)
                e1 = i1[2].item()
                e2 = i2[2].item()
                self.assertEqual(e1, e2)

    if __name__ == '__main__':
        unittest.main()
