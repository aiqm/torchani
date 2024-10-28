import unittest
import os

from torchani._testing import ANITestCase, expand
from torchani.models import ANI1x, ANI2x, ANI1ccx
from torchani.aev import AEVComputer
from torchani.neurochem import load_model_from_name, load_aev_computer_and_symbols

path = os.path.dirname(os.path.realpath(__file__))
const_file_1x = os.path.join(path, "resources/rHCNO-5.2R_16-3.5A_a4-8.params")
const_file_1ccx = os.path.join(path, "resources/rHCNO-5.2R_16-3.5A_a4-8.params")
const_file_2x = os.path.join(path, "resources/rHCNOSFCl-5.1R_16-3.5A_a8-4.params")


@expand(device="cpu", jit=False)
class TestModelLoader(ANITestCase):
    # check that models loaded from a neurochem source are equal to models
    # loaded directly from a state_dict
    def testANI1x(self):
        model = self._setup(ANI1x())
        model_nc = self._setup(load_model_from_name("ani1x"))
        self.assertEqual(model_nc.state_dict(), model.state_dict())

    def testANI1xSingle(self):
        for j in range(8):
            model = self._setup(ANI1x(model_index=j))
            model_nc = self._setup(load_model_from_name("ani1x", model_index=j))
            self.assertEqual(model_nc.state_dict(), model.state_dict())

    def testANI2x(self):
        model = self._setup(ANI2x())
        model_nc = self._setup(load_model_from_name("ani2x"))
        self.assertEqual(model_nc.state_dict(), model.state_dict())

    def testANI2xSingle(self):
        for j in range(8):
            model = self._setup(ANI2x(model_index=j))
            model_nc = self._setup(load_model_from_name("ani2x", model_index=j))
            self.assertEqual(model_nc.state_dict(), model.state_dict())

    def testANI1ccx(self):
        model = self._setup(ANI1ccx())
        model_nc = self._setup(load_model_from_name("ani1ccx"))
        self.assertEqual(model_nc.state_dict(), model.state_dict())


@expand(device="cpu", jit=False)
class TestAEVLoader(ANITestCase):
    # Test that checks that inexact friendly constructor
    # reproduces the values from ANI1x with the correct parameters
    def testEqualNeurochem1x(self):
        aev_1x_nc, _ = load_aev_computer_and_symbols(const_file_1x)
        aev_1x = AEVComputer.like_1x()
        self._compare_constants(aev_1x_nc, aev_1x)

    def testEqualNeurochem2x(self):
        aev_2x_nc, _ = load_aev_computer_and_symbols(const_file_2x)
        aev_2x = AEVComputer.like_2x()
        self._compare_constants(aev_2x_nc, aev_2x)

    def testEqualNeurochem1ccx(self):
        aev_1ccx_nc, _ = load_aev_computer_and_symbols(const_file_1ccx)
        aev_1ccx = AEVComputer.like_1x()
        self._compare_constants(aev_1ccx_nc, aev_1ccx)

    def _compare_constants(self, aev_computer, aev_computer_alt):
        alt_state_dict = aev_computer_alt.state_dict()
        for k, v in aev_computer.state_dict().items():
            self.assertEqual(alt_state_dict[k], v, rtol=1e-17, atol=1e-17)


if __name__ == "__main__":
    unittest.main(verbosity=2)
