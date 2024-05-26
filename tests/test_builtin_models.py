import unittest

import torch

from torchani.testing import ANITest, expand
from torchani.models import ANI1x, ANI2x
from torchani.datasets import create_batched_dataset, TestData


@expand(jit=True, device="cpu")
class TestBuiltinModels(ANITest):
    # Tests if JIT compiled models have the same output energies
    # as eager (non JIT) models

    def setUp(self):
        # in general self energies should be subtracted, and shuffle should be
        # performed, but for these tests this is not important
        self.ds = create_batched_dataset(
            TestData(verbose=False, skip_check=True),
            splits={"training": 1.0},
            direct_cache=True,
            batch_size=256,
            verbose=False,
        )["training"]

    def _test_model(self, model):
        properties = next(iter(self.ds))
        input_ = (
            properties["species"].to(self.device),
            properties["coordinates"].to(self.device, dtype=torch.float),
        )
        _, e = model(input_)
        _, e2 = torch.jit.script(model)(input_)
        self.assertEqual(e, e2)

    def _test_ensemble(self, ensemble):
        self._test_model(ensemble)
        for m in ensemble:
            self._test_model(m)

    def testANI1x(self):
        ani1x = ANI1x().to(self.device)
        self._test_ensemble(ani1x)

    def testANI2x(self):
        ani1ccx = ANI2x().to(self.device)
        self._test_ensemble(ani1ccx)


if __name__ == "__main__":
    unittest.main(verbosity=2)
