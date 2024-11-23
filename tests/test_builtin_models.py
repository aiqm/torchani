import unittest

import torch

from torchani._testing import ANITestCase, expand
from torchani.models import ANI1x, ANI2x, ANIdr, ANImbis
from torchani.datasets import batch_all_in_ram, TestData
from torchani.neighbors import compute_bounding_cell, reconstruct_shifts


@expand()
class TestActiveModelsPoints(ANITestCase):
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

    def _test_model(self, model, modifiable_model):
        properties = next(iter(self.ds))
        species = properties["species"].to(self.device)
        coords = properties["coordinates"].to(self.device, dtype=torch.float)
        for j, m in enumerate(model):
            _, e = m((species, coords))
            modifiable_model.set_active_members([j])
            _, e_active = modifiable_model((species, coords))
            self.assertEqual(e, e_active)


@expand()
class TestExternalNeighborsEntryPoint(ANITestCase):
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
        coords, cell = compute_bounding_cell(coords.detach(), eps=1e-3)
        pbc = torch.tensor([True, True, True], dtype=torch.bool, device=self.device)
        # Emulate external neighbors:
        # - don't pass diff_vectors or distances.
        # - pass shift values
        neighbors = model.neighborlist(model.cutoff, species, coords, cell, pbc)
        shifts = reconstruct_shifts(coords, neighbors)
        result = model.compute_from_external_neighbors(
            species, coords, neighbors.indices, shifts
        )
        out = model((species, coords), cell, pbc)
        if hasattr(out, "atomic_charges"):
            self.assertEqual(out.atomic_charges, result.scalars)
        else:
            self.assertEqual(None, result.scalars)
        self.assertEqual(out.energies, result.energies)


@expand()
class TestInternalNeighborsEntryPoint(TestExternalNeighborsEntryPoint):
    def _test_model(self, model):
        properties = next(iter(self.ds))
        species = properties["species"].to(self.device)
        coords = properties["coordinates"].to(self.device, dtype=torch.float)
        coords, cell = compute_bounding_cell(coords.detach(), eps=1e-3)
        pbc = torch.tensor([True, True, True], dtype=torch.bool, device=self.device)
        elem_idxs = model.species_converter(species)
        neighbors = model.neighborlist(model.cutoff, elem_idxs, coords, cell, pbc)
        result = model.compute_from_neighbors(elem_idxs, coords, neighbors)
        out = model((species, coords), cell, pbc)
        if hasattr(out, "atomic_charges"):
            self.assertEqual(out.atomic_charges, result.scalars)
        else:
            self.assertEqual(None, result.scalars)
        self.assertEqual(out.energies, result.energies)


@expand(jit=True, device="cpu")
class TestBuiltinModels(ANITestCase):
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
        result = model(input_)
        jit_result = torch.jit.script(model)(input_)
        if hasattr(result, "atomic_charges"):
            self.assertEqual(result.atomic_charges, jit_result.atomic_charges)
        self.assertEqual(result.energies, jit_result.energies)

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
