import os
from itertools import product
import unittest

import torch
from ase.io import read
from parameterized import parameterized_class

import torchani
from torchani.testing import TestCase
from torchani.benchmark import timeit

# Disable Tensorfloat, errors between two run of same model for large system
# could reach 1e-3. However note that this error for large system is not that
# big actually.
torch.backends.cuda.matmul.allow_tf32 = False

# TODO: waiting for the NVFuser
# [Bug](https://github.com/pytorch/pytorch/issues/84510) to be fixed
torch._C._jit_set_nvfuser_enabled(False)

devices = ["cuda", "cpu"]
ani2x = torchani.models.ANI2x()


@parameterized_class(("device"), product(devices))
@unittest.skipIf(not torch.cuda.is_available(), "Infer model needs cuda is available")
class TestInfer(TestCase):
    def setUp(self):
        self.ani2x = ani2x.to(self.device)
        self.path = os.path.dirname(os.path.realpath(__file__))

    def _test(self, model_ref, model_infer):
        files = ["small.pdb", "1hz5.pdb", "6W8H.pdb"]
        # Skip 6W8H.pdb (slow on cpu) if device is cpu
        files = files[:-1] if self.device == "cpu" else files
        for file in files:
            filepath = os.path.join(self.path, f"../dataset/pdb/{file}")
            mol = read(filepath)
            species = torch.tensor(
                mol.get_atomic_numbers(), device=self.device
            ).unsqueeze(0)
            coordinates = torch.tensor(
                mol.get_positions(),
                dtype=torch.float32,
                requires_grad=True,
                device=self.device,
            ).unsqueeze(0)

            _, energy1 = model_ref((species, coordinates))
            force1 = torch.autograd.grad(energy1.sum(), coordinates)[0]
            _, energy2 = model_infer((species, coordinates))
            force2 = torch.autograd.grad(energy2.sum(), coordinates)[0]

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
            filepath = os.path.join(self.path, f"../dataset/pdb/{file}")
            mol = read(filepath)
            species = torch.tensor(
                mol.get_atomic_numbers(), device=self.device
            ).unsqueeze(0)
            coordinates = torch.tensor(
                mol.get_positions(),
                dtype=torch.float32,
                requires_grad=True,
                device=self.device,
            ).unsqueeze(0)

            _, energy1 = model((species, coordinates))
            _ = torch.autograd.grad(energy1.sum(), coordinates)[0]  # force

        use_cuaev = self.device == "cuda"
        ani2x_jit = torch.jit.script(
            torchani.models.ANI2x(use_cuda_extension=use_cuaev).to(self.device)
        )
        ani2x_infer_jit = (
            torchani.models.ANI2x(use_cuda_extension=use_cuaev)
            .to_infer_model()
            .to(self.device)
        )
        ani2x_infer_jit = torch.jit.script(ani2x_infer_jit)

        file = "small.pdb"

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
