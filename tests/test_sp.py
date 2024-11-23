import unittest

import torch

from torchani import single_point
from torchani._testing import ANITestCase, expand
from torchani.models import ANI1x, ANIdr, ANImbis
from torchani.datasets import batch_all_in_ram, TestData


@expand()
class TestSinglePointEntry(ANITestCase):
    def setUp(self):
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
        self._test_model(self._setup(ANImbis()))

    def _test_model(self, model):
        properties = next(iter(self.ds))
        species = properties["species"].to(self.device)
        coords = properties["coordinates"].to(self.device, dtype=torch.float)
        out = model((species, coords))

        try:
            result = single_point(model, species, coords, atomic_charges=True)
            self.assertTrue("atomic_charges" in result)
        except ValueError:
            result = single_point(model, species, coords)

        if "atomic_charges" in result:
            self.assertEqual(out.atomic_charges, result["atomic_charges"])
        self.assertEqual(out.energies, result["energies"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
