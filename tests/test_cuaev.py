import torchani
import unittest
import torch

skipIfNoGPU = unittest.skipIf(not torch.cuda.is_available(),
                              'There is no device to run this test')


@unittest.skipIf(torchani.cuaev.is_installed, "only valid when cuaev not installed")
class TestCUAEVNotInstalled(unittest.TestCase):

    def testCuComputeAEV(self):
        self.assertRaisesRegex(RuntimeError, "cuaev is not installed", lambda: torchani.cuaev.cuComputeAEV())


@unittest.skipIf(not torchani.cuaev.is_installed, "only valid when cuaev is installed")
class TestCUAEV(unittest.TestCase):

    @skipIfNoGPU
    def testHello(self):
        # TODO: this should be removed when a real cuaev is merged
        self.assertEqual("Hello World!!!", torchani.cuaev.cuComputeAEV())


if __name__ == '__main__':
    unittest.main()
