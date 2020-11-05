import torchani
import unittest


class TestCUAEV(unittest.TestCase):

    def testInstalled(self):
        self.assertFalse(torchani.cuaev.is_installed())

    def testCuComputeAEV(self):
        # TODO: this should be removed when a real cuaev is merged
        self.assertRaisesRegex(RuntimeError, "cuaev is not installed", lambda: torchani.cuaev.cuComputeAEV())


if __name__ == '__main__':
    unittest.main()
