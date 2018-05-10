import torchani
import pkg_resources
import unittest
import copy
import tempfile
import os
import torch


class TestDataset(unittest.TestCase):

    def setUp(self, data_file=torchani.buildin_dataset_dir):
        self.ds = torchani.Dataset(data_file)

    def testLen(self):
        # compute data length using Dataset
        l1 = 0
        for i in self.ds.iter(1024):
            i, _, _ = i
            l1 += i.shape[0]
        # compute data lenght using pyanitools
        totoal_conformations = 0
        for i in self.ds._loaders:
            loader = self.ds._loaders[i]
            for j in loader:
                totoal_conformations += j['energies'].shape[0]
        self.assertEqual(l1, totoal_conformations)

    def testSaveLoad(self):
        # save to file
        path = os.path.join(tempfile.mkdtemp(), 'dataset.json')
        self.ds.shuffle()
        self.ds.split(('subset1', 0.8), ('subset2', 0.2))
        self.ds.save(path)
        # load from file
        ds2 = torchani.Dataset()
        ds2.load(path)
        # check
        self.assertListEqual(self.ds._keys, ds2._keys)
        self.assertListEqual(
            self.ds._subsets['subset1'], ds2._subsets['subset1'])
        self.assertListEqual(
            self.ds._subsets['subset2'], ds2._subsets['subset2'])
        # check iter
        for i, j in zip(self.ds.iter(1024), ds2.iter(1024)):
            i, _, _ = i
            j, _, _ = j
            self.assertEqual(torch.max(i-j), 0)


if __name__ == '__main__':
    unittest.main()
