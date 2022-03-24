import torch
import torchani
import unittest
import os
from torchani.testing import TestCase


path = os.path.dirname(os.path.realpath(__file__))
dspath = os.path.join(path, '../dataset/ani-1x/sample.h5')


class TestBuiltinModelsJIT(TestCase):
    # Tests if JIT compiled models have the same output energies
    # as eager (non JIT) models

    def setUp(self):
        # in general self energies should be subtracted, and shuffle should be
        # performed, but for these tests this is not important
        self.ds = torchani.data.load(dspath).species_to_indices().collate(256).cache()

    def _test_model(self, model):
        properties = next(iter(self.ds))
        input_ = (properties['species'], properties['coordinates'].float())
        _, e = model(input_)
        _, e2 = torch.jit.script(model)(input_)
        self.assertEqual(e, e2)

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
