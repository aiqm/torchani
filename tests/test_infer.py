import typing as tp
from pathlib import Path
import os
import warnings
import unittest

import torch

from torchani.models import ANI2x
from torchani._testing import ANITestCase, expand
from torchani.csrc import MNP_IS_INSTALLED, CUAEV_IS_INSTALLED
from torchani.grad import energies_and_forces
from torchani.io import read_xyz


if not MNP_IS_INSTALLED:
    warnings.warn("Skipping all MNP tests, install compiled extensions to run them")


@expand()
class TestInfer(ANITestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("MNPNetworks models need CUDA even when running on CPU")
        # MNPNetworks emits warnings due to its complexity, filter those here
        warnings.filterwarnings("ignore", message=".* MNP .*")
        self._saved_omp_num_threads = os.environ.get("OMP_NUM_THREADS", "")
        # set num threads for MNP
        os.environ["OMP_NUM_THREADS"] = "2"
        # Disable TensorFloat32, errors between two run of same model for large system
        # could reach 1e-3. However note that this error for large system is not that
        # big actually.
        self._saved_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False

    def tearDown(self) -> None:
        torch.backends.cuda.matmul.allow_tf32 = self._saved_tf32
        os.environ["OMP_NUM_THREADS"] = self._saved_omp_num_threads
        warnings.resetwarnings()

    def _build_ani2x(
        self, idx: tp.Optional[int] = None, mnp: bool = False, infer: bool = False
    ):
        if mnp and not MNP_IS_INSTALLED:
            raise unittest.SkipTest("MNP extension is not available, skipping MNP test")
        if self.device == "cuda" and CUAEV_IS_INSTALLED:
            strat = "cuaev-fused"
        else:
            strat = "pyaev"
        model = ANI2x(model_index=idx, strategy=strat)
        if infer:
            model = model.to_infer_model(mnp)
        model = self._setup(model)
        return model

    def _test(self, model_ref, model_infer):
        files = ["small.xyz", "1hz5.xyz", "6W8H.xyz"]
        for file in files:
            # Skip 6W8H.xyz (large, slow) if device is cpu
            if self.device == "cpu" and file.startswith("6W8H"):
                continue
            species, coordinates, _, _ = read_xyz(
                (Path(__file__).parent / "resources") / file,
                device=self.device,
                dtype=torch.float,
            )
            force1, energy1 = energies_and_forces(model_ref, species, coordinates)
            force2, energy2 = energies_and_forces(model_infer, species, coordinates)
            self.assertEqual(energy1, energy2, atol=5e-4, rtol=5e-4)
            self.assertEqual(force1, force2, atol=5e-4, rtol=5e-4)

    def testBmmEnsemble(self):
        self._test(
            self._build_ani2x(),
            self._build_ani2x(idx=None, infer=True),
        )

    def testBmmEnsemble_mnp(self):
        self._test(
            self._build_ani2x(),
            self._build_ani2x(mnp=True, idx=None, infer=True),
        )

    def testANIInfer(self):
        if self.jit:
            self.skipTest("MNPNetworks does not support JIT with pyMNP")
        self._test(
            self._build_ani2x(idx=0),
            self._build_ani2x(idx=0, infer=True),
        )

    def testANIInfer_mnp(self):
        self._test(
            self._build_ani2x(idx=0),
            self._build_ani2x(mnp=True, idx=0, infer=True),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
