import unittest

import torch

from torchani.testing import ANITest, expand
from torchani.models import ANI1x, ANI2x, ANIdr, ANImbis
from torchani.datasets import batch_all_in_ram, TestData


@expand()
class TestSinglePointEntry(ANITest):
    def setUp(self):
        # in general self energies should be subtracted, and shuffle should be
        # performed, but for these tests this is not important
        self.ds = batch_all_in_ram(
            TestData(verbose=False, skip_check=True),
            batch_size=256,
            verbose=False,
        )

    def testANI1x(self) -> None:
        self._test_model(self._setup(ANI1x()))

    def testANIdr(self) -> None:
        self._test_model(self._setup(ANIdr()))

    def testANImbis(self) -> None:
        self._test_model(self._setup(ANImbis()), charges=True)

    def _test_model(self, model, charges: bool = False):
        properties = next(iter(self.ds))
        species = properties["species"].to(self.device)
        coords = properties["coordinates"].to(self.device, dtype=torch.float)
        outputs = model.sp((species, coords))
        if not charges:
            _, e = model(
                (species, coords)
            )
        else:
            _, e, q = model.energies_and_atomic_charges(
                (species, coords)
            )
            self.assertEqual(q, outputs["atomic_charges"])
        self.assertEqual(e, outputs["energies"])


@expand(jit=True, device="cpu")
class TestBuiltinModels(ANITest):
    # Tests if JIT compiled models have the same output energies
    # as eager (non JIT) models

    def setUp(self):
        # in general self energies should be subtracted, and shuffle should be
        # performed, but for these tests this is not important
        self.ds = batch_all_in_ram(
            TestData(verbose=False, skip_check=True),
            batch_size=256,
            verbose=False,
        )

    def _test_model(self, model):
        properties = next(iter(self.ds))
        input_ = (
            properties["species"].to(self.device),
            properties["coordinates"].to(self.device, dtype=torch.float),
        )
        _, e = model(input_)
        _, e2 = torch.jit.script(model)(input_)
        self.assertEqual(e, e2)

    def _test_ensemble_charges(self, ensemble):
        self._test_model_charges(ensemble)
        for m in ensemble:
            self._test_model_charges(m)

    def _test_model_charges(self, model):
        properties = next(iter(self.ds))
        input_ = (
            properties["species"].to(self.device),
            properties["coordinates"].to(self.device, dtype=torch.float),
        )
        _, e, q = model.energies_and_atomic_charges(input_)
        _, e2, q2 = torch.jit.script(model).energies_and_atomic_charges(input_)
        self.assertEqual(e, e2)
        self.assertEqual(q, q2)

    def _test_ensemble(self, ensemble):
        self._test_model(ensemble)
        for m in ensemble:
            self._test_model(m)

    def testANI1x(self):
        self._test_ensemble(ANI1x().to(self.device))

    def testANI2x(self):
        self._test_ensemble(ANI2x().to(self.device))

    def testANImbis(self):
        self._test_ensemble(ANImbis().to(self.device))

    def testANImbis_charges(self):
        self._test_ensemble_charges(ANImbis().to(self.device))


if __name__ == "__main__":
    unittest.main(verbosity=2)
