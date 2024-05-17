import unittest

from torchani.testing import ANITest, expand
from torchani.neurochem import load_builtin_from_name
from torchani.models import ANI1x, ANI2x, ANI1ccx


@expand(device="cpu", jit=False)
class TestLoading(ANITest):
    # check that models loaded from a neurochem source are equal to models
    # loaded directly from a state_dict

    def testANI1x(self):
        model = self._setup(ANI1x())
        model_nc = self._setup(load_builtin_from_name("ani1x"))
        self.assertEqual(model_nc.state_dict(), model.state_dict())

    def testANI1xSingle(self):
        for j in range(8):
            model = self._setup(ANI1x(model_index=j))
            model_nc = self._setup(load_builtin_from_name("ani1x", model_index=j))
            self.assertEqual(model_nc.state_dict(), model.state_dict())

    def testANI2x(self):
        model = self._setup(ANI2x())
        model_nc = self._setup(load_builtin_from_name("ani2x"))
        self.assertEqual(model_nc.state_dict(), model.state_dict())

    def testANI2xSingle(self):
        for j in range(8):
            model = self._setup(ANI2x(model_index=j))
            model_nc = self._setup(load_builtin_from_name("ani2x", model_index=j))
            self.assertEqual(model_nc.state_dict(), model.state_dict())

    def testANI1ccx(self):
        model = self._setup(ANI1ccx())
        model_nc = self._setup(load_builtin_from_name("ani1ccx"))
        self.assertEqual(model_nc.state_dict(), model.state_dict())


if __name__ == "__main__":
    unittest.main(verbosity=2)
