import os
import tempfile
import warnings

import torch

from torchani.testing import TestCase, ANITest, expand
from torchani.sae import calculate_saes, sorted_gsaes
from torchani.datasets import create_batched_dataset, ANIBatchedDataset

path = os.path.dirname(os.path.realpath(__file__))
dataset_path_gdb = os.path.join(path, "../dataset/ani1-up_to_gdb4/ani_gdb_s02.h5")


@expand(device="cpu", jit=False)
class TestGSAES(ANITest):
    def testGSAES(self):
        gsaes = sorted_gsaes(("H", "C", "S"), "wB97X", "631Gd")
        self.assertEqual(gsaes, [-0.4993213, -37.8338334, -398.0814169])

        gsaes = sorted_gsaes(("H", "S", "C"), "wB97X", "631Gd")
        self.assertEqual(gsaes, [-0.4993213, -398.0814169, -37.8338334])

        # test case insensitivity
        gsaes = sorted_gsaes(("H", "S", "C"), "Wb97x", "631GD")
        self.assertEqual(gsaes, [-0.4993213, -398.0814169, -37.8338334])

        with self.assertRaises(KeyError):
            sorted_gsaes("wB97X", "631Gd", ("Pu"))


class TestEstimationSAE(TestCase):
    def setUp(self):
        # This test relies on legacy datasets for now, so we ignore this warning
        warnings.filterwarnings(
            "ignore",
            message=".*legacy dataset.*",
            category=UserWarning,
        )
        self.tmp_dir_batched = tempfile.TemporaryDirectory()
        self.batch_size = 2560
        create_batched_dataset(
            dataset_path_gdb,
            dest_path=self.tmp_dir_batched.name,
            splits={"training": 1.0},
            batch_size=self.batch_size,
            divs_seed=12345,
            batch_seed=12345,
            properties=("energies", "species", "coordinates"),
        )
        self.direct_cache = create_batched_dataset(
            dataset_path_gdb,
            splits={"training": 1.0},
            batch_size=self.batch_size,
            divs_seed=12345,
            batch_seed=12345,
            properties=("energies", "species", "coordinates"),
            direct_cache=True,
        )
        self.train = ANIBatchedDataset(self.tmp_dir_batched.name, split="training")

    def tearDown(self) -> None:
        warnings.resetwarnings()
        self.tmp_dir_batched.cleanup()

    def testExactSAE(self):
        self._testExactSAE(direct=False)

    def testStochasticSAE(self):
        self._testStochasticSAE(direct=False)

    def testExactSAEDirect(self):
        self._testExactSAE(direct=True)

    def testStochasticSAEDirect(self):
        self._testStochasticSAE(direct=True)

    def _testExactSAE(self, direct: bool = False):
        if direct:
            ds = self.direct_cache["training"]
        else:
            ds = self.train
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore",
                message=(
                    "Using all batches to estimate SAE,"
                    " this may take up a lot of memory."
                ),
            )
            saes, _ = calculate_saes(ds, ("H", "C", "N", "O"), mode="exact")
            torch.set_printoptions(precision=10)
        self.assertEqual(
            saes,
            torch.tensor(
                [-0.5983182192, -38.0726242065, -54.6750144958, -75.1433029175],
                dtype=torch.float,
            ),
            atol=2.5e-3,
            rtol=2.5e-3,
        )

    def _testStochasticSAE(self, direct: bool = False):
        if direct:
            ds = self.direct_cache["training"]
        else:
            ds = self.train
        saes, _ = calculate_saes(ds, ("H", "C", "N", "O"), mode="sgd")
        # in this specific case the sae difference is very large because it is a
        # very small sample, but for the full sample this imlementation is correct
        self.assertEqual(
            saes,
            torch.tensor([-20.4466, -0.3910, -8.8793, -11.4184], dtype=torch.float),
            atol=0.2,
            rtol=0.2,
        )
