import torchani
import unittest
import torch

skipIfNoGPU = unittest.skipIf(not torch.cuda.is_available(),
                              'There is no device to run this test')


class TestCUAEV(unittest.TestCase):

    def testInstalled(self):
        self.assertTrue(torchani.cuaev.is_installed())

    @skipIfNoGPU
    def testHello(self):
        # TODO: this should be removed when a real cuaev is merged
        self.assertEqual("Hello World!!!", torchani.cuaev.cuComputeAEV())


if __name__ == '__main__':
    unittest.main()
