import unittest
import torchani
from torchani.testing import TestCase


class TestLoading(TestCase):
    # check that models loaded from a neurochem source are equal to models
    # loaded directly from a state_dict

    def testANI1x(self):
        model = torchani.models.ANI1x(use_neurochem_source=False)
        model_nc = torchani.models.ANI1x(use_neurochem_source=True)
        self.assertEqual(model_nc.state_dict(), model.state_dict())

    def testANI1xSingle(self):
        for j in range(8):
            model = torchani.models.ANI1x(model_index=j, use_neurochem_source=False)
            model_nc = torchani.models.ANI1x(model_index=j, use_neurochem_source=True)
            self.assertEqual(model_nc.state_dict(), model.state_dict())

    def testANI2x(self):
        model = torchani.models.ANI2x(use_neurochem_source=False)
        model_nc = torchani.models.ANI2x(use_neurochem_source=True)
        self.assertEqual(model_nc.state_dict(), model.state_dict())

    def testANI2xSingle(self):
        for j in range(8):
            model = torchani.models.ANI2x(model_index=j, use_neurochem_source=False)
            model_nc = torchani.models.ANI2x(model_index=j, use_neurochem_source=True)
            self.assertEqual(model_nc.state_dict(), model.state_dict())

    def testANI1ccx(self):
        model = torchani.models.ANI1ccx(use_neurochem_source=False)
        model_nc = torchani.models.ANI1ccx(use_neurochem_source=True)
        self.assertEqual(model_nc.state_dict(), model.state_dict())


if __name__ == "__main__":
    unittest.main(verbosity=2)
