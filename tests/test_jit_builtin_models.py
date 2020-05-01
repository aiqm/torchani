import torch
import torchani
import unittest
import os


path = os.path.dirname(os.path.realpath(__file__))
dspath = os.path.join(path, '../dataset/ani-1x/sample.h5')


class TestBuiltinModelsJIT(unittest.TestCase):

    def setUp(self):
        self.ani1ccx = torchani.models.ANI1ccx()
        self.ds = torchani.data.load(dspath).subtract_self_energies(self.ani1ccx.sae_dict).species_to_indices().shuffle().collate(256).cache()

    def _test_model(self, model):
        properties = next(iter(self.ds))
        input_ = (properties['species'], properties['coordinates'].float())
        _, e = model(input_)
        _, e2 = torch.jit.script(model)(input_)
        self.assertTrue(torch.allclose(e, e2))

    def _test_ensemble(self, ensemble):
        self._test_model(ensemble)
        for m in ensemble:
            self._test_model(m)

    def testANI1x(self):
        ani1x = torchani.models.ANI1x()
        self._test_ensemble(ani1x)

    def testANI1ccx(self):
        ani1ccx = torchani.models.ANI1ccx()
        self._test_ensemble(ani1ccx)


if __name__ == '__main__':
    unittest.main()
