import unittest
import torch
import torchani

# This test checks that building the model with the default constructor
# gives the same model as loading the model from the pytorch pickled file
# The test is done for models pickled with the periodic_table_index flag
# set and unset since these models are different.
class TestPickledModels(unittest.TestCase):

    @staticmethod
    def count_elements(model):
        n = 0
        for p in model.parameters():
            n += p.numel()
        return n

    @staticmethod
    def have_same_parameters(model1, model2):
        for param1, param2 in zip(model1.parameters(), model2.parameters()):
            if not torch.all(param1 == param2):
                return False
        return True

    def testANI1x_noPTI(self):
        model_init = torchani.models.ANI1x(periodic_table_index=False)
        model_from_pt = torchani.models.ANI1x.from_pt(periodic_table_index=False)
        self.assertEqual(self.count_elements(model_init), self.count_elements(model_from_pt))
        self.assertTrue(self.have_same_parameters(model_init, model_from_pt))

    def testANI1x_PTI(self):
        model_init = torchani.models.ANI1x(periodic_table_index=True)
        model_from_pt = torchani.models.ANI1x.from_pt(periodic_table_index=True)
        self.assertEqual(self.count_elements(model_init), self.count_elements(model_from_pt))
        self.assertTrue(self.have_same_parameters(model_init, model_from_pt))

    def testANI1ccx_noPTI(self):
        model_init = torchani.models.ANI1ccx(periodic_table_index=False)
        model_from_pt = torchani.models.ANI1ccx.from_pt(periodic_table_index=False)
        self.assertEqual(self.count_elements(model_init), self.count_elements(model_from_pt))
        self.assertTrue(self.have_same_parameters(model_init, model_from_pt))

    def testANI1ccx_PTI(self):
        model_init = torchani.models.ANI1ccx(periodic_table_index=True)
        model_from_pt = torchani.models.ANI1ccx.from_pt(periodic_table_index=True)
        self.assertEqual(self.count_elements(model_init), self.count_elements(model_from_pt))
        self.assertTrue(self.have_same_parameters(model_init, model_from_pt))


if __name__ == '__main__':
    unittest.main()
