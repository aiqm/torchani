import tempfile
from pathlib import Path

from torchani.testing import TestCase
from torchani import datasets
from torchani.datasets import (
    download_builtin_dataset,
    _BUILTIN_DATASETS,
)


class TestBuiltinDatasets(TestCase):
    def testSmallSample(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = getattr(datasets, "TestData")(tmpdir)
            self.assertEqual(ds.grouping, "by_num_atoms")

    def testDownloadSmallSample(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            download_builtin_dataset("TestData", "wb97x-631gd", tmpdir)
            num_h5_files = len(list(Path(tmpdir).glob("*.h5")))
            self.assertGreater(num_h5_files, 0)

    def testBuiltins(self):
        # all these have default levels of theory
        classes = _BUILTIN_DATASETS
        for c in classes:
            with tempfile.TemporaryDirectory() as tmpdir:
                with self.assertRaisesRegex(RuntimeError, "Dataset not found"):
                    getattr(datasets, c)(tmpdir, download=False)

        # these also have the B973c/def2mTZVP LoT
        for c in ["ANI1x", "ANI2x", "COMP6v1", "COMP6v2", "AminoacidDimers"]:
            with tempfile.TemporaryDirectory() as tmpdir:
                with self.assertRaisesRegex(RuntimeError, "Dataset not found"):
                    getattr(datasets, c)(
                        tmpdir,
                        download=False,
                        basis_set="def2mTZVP",
                        functional="B973c",
                    )

        # these also have the wB97M-D3BJ/def2TZVPP LoT
        for c in ["ANI1x", "ANI2x", "COMP6v1", "COMP6v2"]:
            with tempfile.TemporaryDirectory() as tmpdir:
                with self.assertRaisesRegex(RuntimeError, "Dataset not found"):
                    getattr(datasets, c)(
                        tmpdir,
                        download=False,
                        basis_set="def2TZVPP",
                        functional="wB97MD3BJ",
                    )
                # Case insensitivity
                with self.assertRaisesRegex(RuntimeError, "Dataset not found"):
                    getattr(datasets, c)(
                        tmpdir,
                        download=False,
                        basis_set="DEF2tZvPp",
                        functional="Wb97md3Bj",
                    )
