import torchani
import unittest
import torch
import os

skipIfNoGPU = unittest.skipIf(not torch.cuda.is_available(),
                              'There is no device to run this test')


@unittest.skipIf(not torchani.aev.has_cuaev, "only valid when cuaev is installed")
class TestCUAEV(torchani.testing.TestCase):

    def testJITSimple(self):
        def f(coordinates, species, Rcr: float, Rca: float, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species: int):
            return torch.ops.cuaev.cuComputeAEV(coordinates, species, Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species)
        s = torch.jit.script(f)
        self.assertIn("cuaev::cuComputeAEV", str(s.graph))

    def testJITAEVComputer(self):
        path = os.path.dirname(os.path.realpath(__file__))
        const_file = os.path.join(path, '../torchani/resources/ani-1x_8x/rHCNO-5.2R_16-3.5A_a4-8.params')  # noqa: E501
        consts = torchani.neurochem.Constants(const_file)
        aev_computer = torchani.AEVComputer(**consts)
        s = torch.jit.script(aev_computer)
        self.assertIn("cuaev::cuComputeAEV", str(s.graph))

    @skipIfNoGPU
    def testHello(self):
        pass


if __name__ == '__main__':
    unittest.main()
