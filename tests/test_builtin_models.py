import unittest

import torch

from torchani.testing import ANITest, expand
from torchani.models import ANI1x, ANI2x, ANIdr, ANImbis
from torchani.datasets import batch_all_in_ram, TestData


@expand()
class TestActiveModelsPoints(ANITest):
    def setUp(self):
        # in general self energies should be subtracted, and shuffle should be
        # performed, but for these tests this is not important
        self.ds = batch_all_in_ram(
            TestData(verbose=False, skip_check=True),
            batch_size=256,
            verbose=False,
        )

    def testANI1x(self) -> None:
        self._test_model(ANI1x().to(self.device), self._setup(ANI1x()))

    def testANI2x(self) -> None:
        self._test_model(ANI2x().to(self.device), self._setup(ANI2x()))

    def _test_model(self, model, modifiable_model, charges: bool = False):
        properties = next(iter(self.ds))
        species = properties["species"].to(self.device)
        coords = properties["coordinates"].to(self.device, dtype=torch.float)
        for j, m in enumerate(model):
            _, e = m((species, coords))
            modifiable_model.set_active_members([j])
            _, e_active = modifiable_model((species, coords))
            self.assertEqual(e, e_active)


@expand()
class TestExternalEntryPoints(ANITest):
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
        self._test_model(self._setup(ANImbis()), charges=True)

    def _test_model(self, model, charges: bool = False):
        properties = next(iter(self.ds))
        species = properties["species"].to(self.device)
        coords = properties["coordinates"].to(self.device, dtype=torch.float)
        coords, cell = model.aev_computer.neighborlist.compute_bounding_cell(
            coords.detach(), eps=1e-3,
        )
        pbc = torch.tensor([True, True, True], dtype=torch.bool, device=self.device)
        if hasattr(model, "potentials"):
            cutoff = model.potentials[0].cutoff
        else:
            cutoff = model.aev_computer.radial_terms.cutoff
        neighbors = model.aev_computer.neighborlist(
            species, coords, cutoff, cell, pbc, return_shift_values=True
        )
        if not charges:
            _, e = model((species, coords), cell, pbc)
            _, e2 = model.from_neighborlist(
                (species, coords), neighbors.indices, neighbors.shift_values
            )
        else:
            _, e, q = model.energies_and_atomic_charges((species, coords), cell, pbc)
            _, e2, q2 = model.energies_and_atomic_charges_from_neighborlist(
                (species, coords), neighbors.indices, neighbors.shift_values
            )
            self.assertEqual(q, q2)
        self.assertEqual(e, e2)


@expand(jit=True, device="cpu")
class TestBuiltinModels(ANITest):
    # Tests if JIT compiled models have the same output energies
    # as eager (non JIT) models

    def setUp(self):
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
