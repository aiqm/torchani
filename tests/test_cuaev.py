import torchani
import unittest
import torch

skipIfNoGPU = unittest.skipIf(not torch.cuda.is_available(),
                              'There is no device to run this test')


@unittest.skipIf(not torchani.has_cuaev, "only valid when cuaev is installed")
class TestCUAEV(unittest.TestCase):

    def testJIT(self):
        def f(coordinates, species, Rcr: float, Rca: float, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species: int):
            return torch.ops.cuaev.cuComputeAEV(coordinates, species, Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species)
        s = torch.jit.script(f)
        self.assertIn("cuaev::cuComputeAEV", str(s.graph))

    @skipIfNoGPU
    def testHello(self):
        pass


if __name__ == '__main__':
    unittest.main()
