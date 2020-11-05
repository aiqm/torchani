import torchani
import unittest

class TestCUAEV(unittest.TestCase):

    def testHello(self):
        self.assertEqual("Hello World!!!", torchani.cuaev.cuComputeAEV())
