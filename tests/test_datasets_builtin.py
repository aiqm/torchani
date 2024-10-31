import typing as tp
import tempfile

from torchani._testing import TestCase
from torchani import datasets
from torchani.cli import data_pull
from torchani.datasets import _DatasetId, _LotId
from torchani.paths import set_data_dir


class TestBuiltinDatasets(TestCase):
    tmpdir: tp.Any

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        set_data_dir(cls.tmpdir.name)

    @classmethod
    def tearDownClass(cls):
        set_data_dir()
        cls.tmpdir.cleanup()

    def testSmallSample(self):
        ds = datasets.TestData()
        self.assertEqual(ds.grouping, "by_num_atoms")

    def testDownloadSmallSample(self):
        data_pull([_DatasetId.TESTDATA], [_LotId.WB97X_631GD])
        files = datasets.TestData().store_locations
        self.assertGreater(len(files), 0)

    def testBuiltins(self):
        # all these have default levels of theory
        for ds_id in _DatasetId:
            if ds_id is _DatasetId.TESTDATA:
                continue
            c = ds_id.value
            with self.assertRaisesRegex(RuntimeError, "Dataset not found"):
                getattr(datasets, c)(download=False)

        # these also have the B973c/def2mTZVP LoT
        for c in ["ANI1x", "ANI2x", "COMP6v1", "COMP6v2", "AminoacidDimers"]:
            with self.assertRaisesRegex(RuntimeError, "Dataset not found"):
                getattr(datasets, c)(download=False, lot="B973c-def2mTZVP")

        # these also have the wB97M-D3BJ/def2TZVPP LoT
        for c in ["ANI1x", "ANI2x", "COMP6v1", "COMP6v2"]:
            with self.assertRaisesRegex(RuntimeError, "Dataset not found"):
                getattr(datasets, c)(download=False, lot="wB97MD3BJ-def2TZVPP")
            # Case insensitivity
            with self.assertRaisesRegex(RuntimeError, "Dataset not found"):
                getattr(datasets, c)(download=False, lot="Wb97md3Bj-DEF2tZvPp")
