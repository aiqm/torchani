import unittest
import torchani
from torchani.testing import TestCase
from torchani import assembler


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

    def testANI1ccx(self):
        model = torchani.models.ANI1ccx(use_neurochem_source=False)
        model_nc = torchani.models.ANI1ccx(use_neurochem_source=True)
        self.assertEqual(model_nc.state_dict(), model.state_dict())

    def testAssembler(self):
        model_1ccx = torchani.models.ANI1ccx(use_neurochem_source=False)
        model_1x = torchani.models.ANI1x(use_neurochem_source=False)
        model_2x = torchani.models.ANI2x(use_neurochem_source=False)
        model_dr = torchani.models.ANIdr(use_neurochem_source=False)

        asm_1x = assembler.ANI1x()
        asm_2x = assembler.ANI2x()
        asm_1ccx = assembler.ANI1ccx()
        asm_dr = assembler.ANIdr()
        self.assertEqual(model_1x.state_dict(), asm_1x.state_dict())
        self.assertEqual(model_dr.state_dict(), asm_dr.state_dict())
        self.assertEqual(model_1ccx.state_dict(), asm_1ccx.state_dict())
        self.assertEqual(model_2x.state_dict(), asm_2x.state_dict())


if __name__ == '__main__':
    unittest.main()
