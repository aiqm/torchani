import unittest

import torch

from torchani.testing import ANITest, expand
from torchani.models import ANI1x, ANIdr, ANImbis
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
        outputs = model.sp(species, coords)
        if not charges:
            _, e = model((species, coords))
        else:
            _, e, q = model.energies_and_atomic_charges((species, coords))
            self.assertEqual(q, outputs["atomic_charges"])
        self.assertEqual(e, outputs["energies"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
