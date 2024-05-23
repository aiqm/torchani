import os
from itertools import product
import unittest
from pathlib import Path

import torch
from parameterized import parameterized_class

import torchani
from torchani.csrc import CUAEV_IS_INSTALLED
from torchani.testing import TestCase
from torchani.benchmark import timeit
from torchani.grad import energies_and_forces
from torchani.io import read_xyz


devices = ["cuda", "cpu"]
ani2x = torchani.models.ANI2x()


@parameterized_class(("device"), product(devices))
@unittest.skipIf(not torch.cuda.is_available(), "Infer model needs cuda is available")
class TestInfer(TestCase):
    def setUp(self):
        # Disable Tensorfloat, errors between two run of same model for large system
        # could reach 1e-3. However note that this error for large system is not that
        # big actually.
        self._saved_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        # TODO: waiting for the NVFuser
        # [Bug](https://github.com/pytorch/pytorch/issues/84510) to be fixed
        torch._C._jit_set_nvfuser_enabled(False)
        self.ani2x = ani2x.to(self.device)
        self.path = os.path.dirname(os.path.realpath(__file__))

    def tearDown(self) -> None:
        torch.backends.cuda.matmul.allow_tf32 = self._saved_tf32
        torch._C._jit_set_nvfuser_enabled(True)

    def _test(self, model_ref, model_infer):
        files = ["small.xyz", "1hz5.xyz", "6W8H.xyz"]
        for file in files:
            # Skip 6W8H.pdb (large, slow) if device is cpu
            if self.device == "cpu" and file.startswith("6W8H"):
                continue
            species, coordinates, _ = read_xyz(
                (Path(__file__).parent / "test_data") / file,
                device=self.device,
                dtype=torch.float,
            )
            force1, energy1 = energies_and_forces(model_ref, species, coordinates)
            force2, energy2 = energies_and_forces(model_infer, species, coordinates)
            self.assertEqual(energy1, energy2, atol=1e-5, rtol=1e-5)
            self.assertEqual(force1, force2, atol=1e-5, rtol=1e-5)

    def testBmmEnsemble(self):
        ani2x_infer = torchani.models.ANI2x().to_infer_model().to(self.device)
        self._test(ani2x, ani2x_infer)

    def testBmmEnsembleJIT(self):
        ani2x_infer_jit = torchani.models.ANI2x().to_infer_model().to(self.device)
        ani2x_infer_jit = torch.jit.script(ani2x_infer_jit)
        self._test(ani2x, ani2x_infer_jit)

    def testBenchmarkJIT(self):
        """
        Sample benchmark result on 2080 Ti
        cuda:
            run_ani2x                          : 21.739 ms/step
            run_ani2x_infer                    : 9.630 ms/step
        cpu:
            run_ani2x                          : 756.459 ms/step
            run_ani2x_infer                    : 32.482 ms/step
        """

        def run(model, file):
            species, coordinates, _ = read_xyz(
                (Path(__file__).parent / "test_data") / file,
                device=self.device,
                dtype=torch.float,
            )
            _, _ = energies_and_forces(model, species, coordinates)

        use_cuaev = (self.device == "cuda") and CUAEV_IS_INSTALLED
        ani2x_jit = torch.jit.script(
            torchani.models.ANI2x(use_cuda_extension=use_cuaev).to(self.device)
        )
        ani2x_infer_jit = (
            torchani.models.ANI2x(use_cuda_extension=use_cuaev)
            .to_infer_model()
            .to(self.device)
        )
        ani2x_infer_jit = torch.jit.script(ani2x_infer_jit)

        file = "small.xyz"

        def run_ani2x():
            run(ani2x_jit, file)

        def run_ani2x_infer():
            run(ani2x_infer_jit, file)

        steps = 10 if self.device == "cpu" else 30
        print()
        timeit(run_ani2x, steps=steps)
        timeit(run_ani2x_infer, steps=steps)


if __name__ == "__main__":
    unittest.main(verbosity=2)
