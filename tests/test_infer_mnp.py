from pathlib import Path
import os
import torch
import unittest

from ase.io import read

import torchani
from torchani.testing import TestCase

# Disable Tensorfloat, errors between two run of same model for large system
# could reach 1e-3. However note that this error for large system is not that
# big actually.
torch.backends.cuda.matmul.allow_tf32 = False

# TODO: waiting for the NVFuser
# [Bug](https://github.com/pytorch/pytorch/issues/84510) to be fixed
torch._C._jit_set_nvfuser_enabled(False)

# set num threads for multi net parallel
os.environ["OMP_NUM_THREADS"] = "2"

PDB_PATH = (Path(__file__).resolve().parent.parent / "dataset") / "pdb"


@unittest.skipIf(
    not torchani.infer.mnp_is_installed,
    "Nvtx commands need the MNP extension",
)
class TestNvtx(TestCase):
    def testNVTX(self):
        torch.ops.mnp.nvtx_range_push("hello")
        torch.ops.mnp.nvtx_range_pop()


@unittest.skipIf(
    not torch.cuda.is_available(),
    "InferMNP model needs CUDA",
)
@unittest.skipIf(
    not torchani.infer.mnp_is_installed,
    "InferMNP model needs the MNP extension",
)
class TestCPUInferMNP(TestCase):
    def setUp(self):
        self.scripting = False
        self.device = "cpu"
        self.ani2x = self._build_ani2x()

    def _setup_model(self, model):
        return model

    def _build_ani2x(self):
        use_cuda = self.device == "cuda"
        return torchani.models.ANI2x(use_cuda_extension=use_cuda).to(self.device)

    def _test(self, model_ref, model_infer):
        files = ["small.pdb", "1hz5.pdb", "6W8H.pdb"]
        # Skip 6W8H.pdb (slow on cpu) if device is cpu
        files = files[:-1] if self.device == "cpu" else files
        for file in files:
            mol = read(str(PDB_PATH / file))
            species = torch.tensor(
                mol.get_atomic_numbers(),
                dtype=torch.int64,
                device=self.device,
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
            self.assertEqual(force1, force2, atol=5e-4, rtol=5e-4)

    def testBmmEnsemble(self):
        ani2x_infer = self._build_ani2x()
        ani2x_infer.neural_networks = ani2x_infer.neural_networks.to_infer_model(False)
        self._test(self.ani2x, self._setup_model(ani2x_infer))

    def testBmmEnsembleMNP(self):
        ani2x_infer = self._build_ani2x()
        ani2x_infer.neural_networks = ani2x_infer.neural_networks.to_infer_model(True)
        self._test(self.ani2x, self._setup_model(ani2x_infer))

    def testANIInfer(self):
        if self.scripting:
            return
        ani2x_infer = self._build_ani2x()[0]
        ani2x_infer.neural_networks = ani2x_infer.neural_networks.to_infer_model(False)
        self._test(self.ani2x[0], self._setup_model(ani2x_infer))

    def testANIInferMNP(self):
        ani2x_infer = self._build_ani2x()[0]
        ani2x_infer.neural_networks = ani2x_infer.neural_networks.to_infer_model(True)
        self._test(self.ani2x[0], self._setup_model(ani2x_infer))


@unittest.skipIf(
    not torch.cuda.is_available(),
    "InferMNP model needs CUDA",
)
@unittest.skipIf(
    not torchani.infer.mnp_is_installed,
    "InferMNP model needs the MNP extension",
)
class TestCUDAInferMNP(TestCPUInferMNP):
    def setUp(self):
        self.scripting = True
        self.device = "cuda"
        self.ani2x = torchani.models.ANI2x().to(self.device)
        self.path = os.path.dirname(os.path.realpath(__file__))

    def _setup_model(self, model):
        return torch.jit.script(model)


if __name__ == "__main__":
    unittest.main(verbosity=2)
