import os
import torch
import torchani
from ase.io import read
from itertools import product
import unittest
from torchani.testing import TestCase
from parameterized import parameterized_class

# Disable Tensorfloat, errors between two run of same model for large system could reach 1e-3.
# However note that this error for large system is not that big actually.
torch.backends.cuda.matmul.allow_tf32 = False

use_mnps = [True, False] if torchani.infer.mnp_is_installed else [False]
devices = ['cuda', 'cpu']
ani2x = torchani.models.ANI2x(periodic_table_index=True, model_index=None)


@parameterized_class(('device', 'use_mnp'), product(devices, use_mnps))
@unittest.skipIf(not torch.cuda.is_available(), "Infer model needs cuda is available")
class TestInfer(TestCase):

    def setUp(self):
        self.ani2x = ani2x
        self.path = os.path.dirname(os.path.realpath(__file__))

    def _test(self, model_ref, model_infer):
        files = ['small.pdb', '1hz5.pdb', '6W8H.pdb']
        # Skip 6W8H.pdb (slow on cpu) if device is cpu
        files = files[:-1] if self.device == 'cpu' else files
        for file in files:
            filepath = os.path.join(self.path, f'../dataset/pdb/{file}')
            mol = read(filepath)
            species = torch.tensor([mol.get_atomic_numbers()], device=self.device)
            positions = torch.tensor([mol.get_positions()], dtype=torch.float32, requires_grad=False, device=self.device)
            speciesPositions = self.ani2x.species_converter((species, positions))
            species, coordinates = speciesPositions
            coordinates.requires_grad_(True)

            _, energy1 = model_ref((species, coordinates))
            force1 = torch.autograd.grad(energy1.sum(), coordinates)[0]
            _, energy2 = model_infer((species, coordinates))
            force2 = torch.autograd.grad(energy2.sum(), coordinates)[0]

            self.assertEqual(energy1, energy2, atol=1e-5, rtol=1e-5)
            self.assertEqual(force1, force2, atol=1e-5, rtol=1e-5)

    def testBmmEnsemble(self):
        model_iterator = self.ani2x.neural_networks
        aev_computer = torchani.AEVComputer.like_2x(use_cuda_extension=(self.device == 'cuda'))
        ensemble = torchani.nn.Sequential(aev_computer, model_iterator).to(self.device)
        bmm_ensemble = torchani.nn.Sequential(aev_computer, self.ani2x.neural_networks.to_infer_model(use_mnp=self.use_mnp)).to(self.device)
        self._test(ensemble, bmm_ensemble)

    def testANIInferModel(self):
        model_iterator = self.ani2x.neural_networks
        aev_computer = torchani.AEVComputer.like_2x(use_cuda_extension=(self.device == 'cuda'))
        model_ref = torchani.nn.Sequential(aev_computer, model_iterator[0]).to(self.device)
        model_infer = torchani.nn.Sequential(aev_computer, model_iterator[0].to_infer_model(use_mnp=self.use_mnp)).to(self.device)
        self._test(model_ref, model_infer)

    def testBmmEnsembleJIT(self):
        model_iterator = self.ani2x.neural_networks
        aev_computer = torchani.AEVComputer.like_2x(use_cuda_extension=(self.device == 'cuda'))
        ensemble = torchani.nn.Sequential(aev_computer, model_iterator).to(self.device)
        # jit
        bmm_ensemble = torchani.nn.Sequential(aev_computer, self.ani2x.neural_networks.to_infer_model(use_mnp=self.use_mnp)).to(self.device)
        bmm_ensemble_jit = torch.jit.script(bmm_ensemble)
        if self.use_mnp:
            self._test(ensemble, bmm_ensemble_jit)
        else:
            with self.assertRaisesRegex(torch.jit.Error, "The following operation failed in the TorchScript interpreter."):
                # with error "RuntimeError: JIT Infer Model only support use_mnp=True"
                self._test(ensemble, bmm_ensemble_jit)

    def testANIInferModelJIT(self):
        model_iterator = self.ani2x.neural_networks
        aev_computer = torchani.AEVComputer.like_2x(use_cuda_extension=(self.device == 'cuda'))
        model_ref = torchani.nn.Sequential(aev_computer, model_iterator[0]).to(self.device)
        # jit
        model_infer = torchani.nn.Sequential(aev_computer, model_iterator[0].to_infer_model(use_mnp=self.use_mnp)).to(self.device)
        model_infer_jit = torch.jit.script(model_infer)
        if self.use_mnp:
            self._test(model_ref, model_infer_jit)
        else:
            with self.assertRaisesRegex(torch.jit.Error, "The following operation failed in the TorchScript interpreter."):
                # with error "RuntimeError: JIT Infer Model only support use_mnp=True"
                self._test(model_ref, model_infer_jit)


if __name__ == '__main__':
    unittest.main()
