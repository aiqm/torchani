import torchani
import unittest
import torch
import os
from torchani.testing import TestCase, make_tensor

skipIfNoGPU = unittest.skipIf(not torch.cuda.is_available(),
                              'There is no device to run this test')


@unittest.skipIf(not torchani.aev.has_cuaev, "only valid when cuaev is installed")
class TestCUAEVNoGPU(TestCase):

    def testSimple(self):
        def f(coordinates, species, Rcr: float, Rca: float, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species: int):
            return torch.ops.cuaev.cuComputeAEV(coordinates, species, Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species)
        s = torch.jit.script(f)
        self.assertIn("cuaev::cuComputeAEV", str(s.graph))

    def testAEVComputer(self):
        path = os.path.dirname(os.path.realpath(__file__))
        const_file = os.path.join(path, '../torchani/resources/ani-1x_8x/rHCNO-5.2R_16-3.5A_a4-8.params')  # noqa: E501
        consts = torchani.neurochem.Constants(const_file)
        aev_computer = torchani.AEVComputer(**consts, use_cuda_extension=True)
        s = torch.jit.script(aev_computer)
        # Computation of AEV using cuaev when there is no atoms does not require CUDA, and can be run without GPU
        species = make_tensor((8, 0), 'cpu', torch.int64, low=-1, high=4)
        coordinates = make_tensor((8, 0, 3), 'cpu', torch.float32, low=-5, high=5)
        self.assertIn("cuaev::cuComputeAEV", str(s.graph_for((species, coordinates))))


@unittest.skipIf(not torchani.aev.has_cuaev, "only valid when cuaev is installed")
@skipIfNoGPU
class TestCUAEV(TestCase):
    def testHello(self):
        pass


if __name__ == '__main__':
    unittest.main()
