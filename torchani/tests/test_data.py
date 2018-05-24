import torchani
import unittest
import copy
import tempfile
import os
import torch
import torchani.pyanitools as pyanitools
import torchani.data

class TestDataset(unittest.TestCase):

    def setUp(self, data_path=torchani.buildin_dataset_dir):
        self.data_path = data_path
        self.ds = torchani.data.load_dataset(data_path)

    def testLen(self):
        # compute data length using Dataset
        l1 = len(self.ds)
        # compute data lenght using pyanitools
        l2 = 0
        for f in os.listdir(self.data_path):
            f = os.path.join(self.data_path, f)
            if os.isfile(f) and (f.endswith('.h5') or f.endswith('.hdf5')):
                for j in pyanitools.anidataloader(f):
                    l2 += j['energies'].shape[0]
        self.assertEqual(l1, l2)

    def testSaveLoad(self):
        pass


if __name__ == '__main__':
    unittest.main()
