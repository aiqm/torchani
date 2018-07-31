import sys

if sys.version_info.major >= 3:
    import os
    import unittest
    import torchani.data

    path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(path, '../dataset')

    class TestDataset(unittest.TestCase):

        def _test_chunksize(self, chunksize):
            ds = torchani.data.ANIDataset(path, chunksize)
            for i in ds:
                self.assertLessEqual(i['coordinates'].shape[0], chunksize)

        def testChunk64(self):
            self._test_chunksize(64)

        def testChunk128(self):
            self._test_chunksize(128)

        def testChunk32(self):
            self._test_chunksize(32)

        def testChunk256(self):
            self._test_chunksize(256)

    if __name__ == '__main__':
        unittest.main()
