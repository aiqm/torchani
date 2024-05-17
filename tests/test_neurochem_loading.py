import unittest

from torchani.testing import TestCase
from torchani.neurochem import load_builtin
from torchani.models import ANI1x, ANI2x, ANI1ccx


class TestLoading(TestCase):
    # check that models loaded from a neurochem source are equal to models
    # loaded directly from a state_dict

    def testANI1x(self):
        model = ANI1x()
        model_nc = load_builtin("ani1x")
        self.assertEqual(model_nc.state_dict(), model.state_dict())

    def testANI1xSingle(self):
        for j in range(8):
            model = ANI1x(model_index=j)
            model_nc = load_builtin("ani1x", model_index=j)
            self.assertEqual(model_nc.state_dict(), model.state_dict())

    def testANI2x(self):
        model = ANI2x()
        model_nc = load_builtin("ani2x")
        self.assertEqual(model_nc.state_dict(), model.state_dict())

    def testANI2xSingle(self):
        for j in range(8):
            model = ANI2x(model_index=j)
            model_nc = load_builtin("ani2x", model_index=j)
            self.assertEqual(model_nc.state_dict(), model.state_dict())

    def testANI1ccx(self):
        model = ANI1ccx()
        model_nc = load_builtin("ani1ccx")
        self.assertEqual(model_nc.state_dict(), model.state_dict())


if __name__ == "__main__":
    unittest.main(verbosity=2)
