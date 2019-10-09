import torch
import torchani
import unittest
import os
import pickle


path = os.path.dirname(os.path.realpath(__file__))
dspath = os.path.join(path, '../dataset/ani-1x/sample.h5')
batch_size = 256
chunk_threshold = 5
other_properties = {'properties': ['energies'],
                    'padding_values': [None],
                    'padded_shapes': [(batch_size, )],
                    'dtypes': [torch.float64],
                    }


class TestBuiltinModelsJIT(unittest.TestCase):

    def setUp(self):
        self.ds = torchani.data.CachedDataset(dspath, batch_size=batch_size, device='cpu',
                                              chunk_threshold=chunk_threshold,
                                              other_properties=other_properties,
                                              subtract_self_energies=True)
        self.ani1ccx = torchani.models.ANI1ccx()

    def _test_model(self, model):
        chunk = self.ds[0][0][0]
        _, e = model(chunk)
        _, e2 = torch.jit.script(model)(chunk)
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
