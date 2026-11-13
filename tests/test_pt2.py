# NOTE: These tests are experimental
import os
import torch
from torch import Tensor
from torch.export import Dim
import unittest

import pytest

from torchani._testing import (
    ANITestCase,
    Molecs,
    expand,
    make_elem_idxs,
    make_molec,
    make_molecs,
    make_neighbors,
)
from torchani.arch import simple_ani, simple_aniq
from torchani.nn import ANINetworks
from torchani.neighbors import (
    discard_outside_cutoff,
    neighbors_to_triples,
    AdaptiveList,
    AllPairs,
    CellList,
    Neighbors,
    Triples,
)
from torchani.aev import ANIRadial, ANIAngular, AEVComputer

SYMBOLS = ("H", "C", "N", "O")

pytestmark = pytest.mark.pt2


class ANITestCasePT2(ANITestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        # Set up logging (high verbosity with +) for Dynamo
        os.environ["TORCH_LOGS"] = "+dynamo"
        os.environ["TORCHDYNAMO_VERBOSE"] = "1"  # (needed according to log msgs?)
        os.environ["TORCHDYNAMO_EXTENDED_DEBUG_CREATE_SYMBOL"] = "u2"
        # Many models use dynamic output shape ops (nonzero, unique, unique_consecutive)
        torch._dynamo.config.capture_dynamic_output_shape_ops = True
        # Some functions call .item() or int() on tensors (e.g. sizes of index tensors)
        torch._dynamo.config.capture_scalar_outputs = True

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass()
        os.environ["TORCH_LOGS"] = ""
        os.environ["TORCHDYNAMO_VERBOSE"] = "0"
        torch._dynamo.config.capture_dynamic_output_shape_ops = False
        torch._dynamo.config.capture_scalar_outputs = False

    def tearDown(self) -> None:
        torch.compiler.reset()  # needed?


@expand(jit=False)
class TestCompile(ANITestCasePT2):
    def _assert_compiled_matches(self, fn, *args) -> None:
        expect = fn(*args)
        actual = torch.compile(fn, fullgraph=True)(*args)
        self.assertEqual(actual, expect, rtol=1e-5, atol=1e-5)

    # Compares energies (and charges, if present) and forces
    def _assert_compiled_model_matches(self, model, molecs: Molecs) -> None:
        results = []
        for fn in (model, torch.compile(model, fullgraph=True)):
            coords = molecs.coords.clone().requires_grad_(True)
            out = fn((molecs.atomic_nums, coords), cell=molecs.cell, pbc=molecs.pbc)
            forces = -torch.autograd.grad(out.energies.sum(), coords)[0]
            results.append((tuple(out[1:]), forces))
        self.assertEqual(results[1], results[0], rtol=1e-5, atol=1e-5)

    # On CUDA the compiled and eager pair (or triple) ordering may differ, so pairs
    # and triples are compared after sorting rows of (idx, idx, distance, ...)
    @staticmethod
    def _sorted_rows(rows: Tensor) -> Tensor:
        for col in reversed(range(rows.shape[1])):
            rows = rows[torch.sort(rows[:, col], stable=True).indices]
        return rows

    def _canonical_neighbors(self, neighbors: Neighbors) -> Tensor:
        first = neighbors.indices.min(0).values.to(neighbors.distances.dtype)
        second = neighbors.indices.max(0).values.to(neighbors.distances.dtype)
        return self._sorted_rows(torch.stack((first, second, neighbors.distances), 1))

    def _canonical_triples(self, triples: Triples) -> Tensor:
        central = triples.central_idxs.to(triples.distances.dtype)
        dist_min = triples.distances.min(0).values
        dist_max = triples.distances.max(0).values
        return self._sorted_rows(torch.stack((central, dist_min, dist_max), 1))

    def _neighbors(self) -> Neighbors:
        return make_neighbors(10, seed=1234, device=self.device)

    def testRadial(self) -> None:
        radial = ANIRadial.like_2x().to(self.device)
        self._assert_compiled_matches(radial, self._neighbors().distances)

    def testAngular(self) -> None:
        angular = ANIAngular.like_2x().to(self.device)
        triples = neighbors_to_triples(self._neighbors())
        self._assert_compiled_matches(
            angular, triples.distances, triples.diff_vectors
        )

    def testDiscardOutsideCutoff(self) -> None:
        self._assert_compiled_matches(discard_outside_cutoff, self._neighbors(), 3.5)

    def testNeighborsToTriples(self) -> None:
        neighbors = self._neighbors()
        expect = neighbors_to_triples(neighbors)
        actual = torch.compile(neighbors_to_triples, fullgraph=True)(neighbors)
        self.assertEqual(
            self._canonical_triples(actual),
            self._canonical_triples(expect),
            rtol=1e-5,
            atol=1e-5,
        )

    def testAEVComputer(self) -> None:
        aev_computer = AEVComputer.like_2x().to(self.device)
        molec = make_molec(10, seed=1234, device=self.device)
        elem_idxs = make_elem_idxs(1, 10, seed=1234).to(self.device)
        neighbors = AllPairs()(5.2, elem_idxs, molec.coords)
        self._assert_compiled_matches(
            aev_computer.compute_from_neighbors, elem_idxs, molec.coords, neighbors
        )

    # Only some of the supported elements are present, and some atoms are dummy
    def testANINetworks(self) -> None:
        networks = ANINetworks.like_2x().to(self.device)
        elem_idxs = torch.tensor([[0, 1, 1, 0, -1], [3, 0, 0, -1, -1]])
        elem_idxs = elem_idxs.to(self.device)
        aevs = torch.rand(2, 5, 1008, device=self.device)
        results = []
        for fn in (networks, torch.compile(networks, fullgraph=True)):
            _aevs = aevs.clone().requires_grad_(True)
            energies = fn(elem_idxs, _aevs)
            grad = torch.autograd.grad(energies.sum(), _aevs)[0]
            results.append((energies, grad))
        self.assertEqual(results[1], results[0], rtol=1e-5, atol=1e-5)

    def testNeighborlists(self) -> None:
        # Sizes of 6 and 10 atoms test both branches of the adaptive list
        cases = (
            (AllPairs(), 6),
            (CellList(), 6),
            (AdaptiveList(threshold=8, threshold_nopbc=8), 6),
            (AdaptiveList(threshold=8, threshold_nopbc=8), 10),
        )
        for neighborlist, atoms in cases:
            for pbc in (False, True):
                with self.subTest(neighborlist=neighborlist, atoms=atoms, pbc=pbc):
                    torch.compiler.reset()
                    molec = make_molec(
                        atoms, cell_size=7.0, pbc=pbc, seed=1234, device=self.device
                    )
                    args = (5.2, molec.atomic_nums, molec.coords, molec.cell, molec.pbc)
                    expect = neighborlist(*args)
                    actual = torch.compile(neighborlist, fullgraph=True)(*args)
                    self.assertEqual(
                        self._canonical_neighbors(actual),
                        self._canonical_neighbors(expect),
                        rtol=1e-5,
                        atol=1e-5,
                    )

    def testModel(self) -> None:
        model = simple_ani(SYMBOLS, "wb97x-631gd", repulsion=False, strategy="pyaev")
        molecs = make_molecs(2, 6, seed=1234, device=self.device)
        molecs.atomic_nums[1, -2:] = -1
        self._assert_compiled_model_matches(model.to(self.device), molecs)

    def testModelCharges(self) -> None:
        model = simple_aniq(SYMBOLS, "wb97x-631gd", repulsion=False, strategy="pyaev")
        molecs = make_molecs(2, 6, seed=1234, device=self.device)
        molecs.atomic_nums[1, -2:] = -1
        self._assert_compiled_model_matches(model.to(self.device), molecs)

    def testModelRepulsionPBC(self) -> None:
        model = simple_ani(SYMBOLS, "wb97x-631gd", repulsion=True, strategy="pyaev")
        molec = make_molec(6, cell_size=6.0, pbc=True, seed=1234, device=self.device)
        self._assert_compiled_model_matches(model.to(self.device), molec)

    @unittest.skip("Data-dependent branching and module state mutation")
    def testVerletCellList(self) -> None:
        pass

    @unittest.skip("C++ ops have no fake kernels, and TorchBind classes no fake class")
    def testCppExtensions(self) -> None:
        pass


class TestExport(ANITestCasePT2):

    def testRadial(self) -> None:
        neighbors = make_neighbors(10, seed=1234)
        mod = ANIRadial.like_1x()
        pairs_dim = Dim("pairs")
        _ = torch.export.export(
            mod,
            args=(neighbors.distances,),
            dynamic_shapes={"distances": (pairs_dim,)},
        )

    def testAngular(self) -> None:
        neighbors = make_neighbors(10, seed=1234)
        triples = neighbors_to_triples(neighbors)
        mod = ANIAngular.like_1x()
        stat = Dim.STATIC  # type: ignore
        triples_dim = Dim("triples")
        _ = torch.export.export(
            mod,
            args=(triples.distances, triples.diff_vectors),
            dynamic_shapes={
                "tri_distances": (stat, triples_dim),
                "tri_vectors": (stat, triples_dim, stat),
            },
        )

    def testDiscardNeighbors(self) -> None:
        neighbors = make_neighbors(10, seed=1234)

        class Mod(torch.nn.Module):
            def forward(
                self, neighbors: tuple[Tensor, Tensor, Tensor], cutoff: float
            ) -> Neighbors:
                indices, distances, diff_vectors = neighbors
                _neighbors = Neighbors(indices, distances, diff_vectors)
                return discard_outside_cutoff(_neighbors, cutoff)

        m = Mod()

        stat = Dim.STATIC  # type: ignore
        pairs = Dim("pairs")
        _ = torch.export.export(
            m,
            args=(tuple(neighbors), 5.2),
            dynamic_shapes={
                "neighbors": ((stat, pairs), (pairs,), (pairs, stat)),
                "cutoff": None,  # cutoff is fixed on export
            },
        )

    # NOTE: This is top priority to export with compute_from_external_neighbors
    @unittest.skipIf(True, "Fails in pytorch 2.5, due to m: int = int(counts.max())")
    def testNeighborsToTriples(self) -> None:
        neighbors = make_neighbors(10, seed=1234)

        class Mod(torch.nn.Module):
            def forward(self, neighbors: tuple[Tensor, Tensor, Tensor]) -> Triples:
                indices, distances, diff_vectors = neighbors
                _neighbors = Neighbors(indices, distances, diff_vectors)
                return neighbors_to_triples(_neighbors)

        m = Mod()
        stat = Dim.STATIC  # type: ignore
        pairs = torch.export.Dim("pairs")
        _ = torch.export.export(
            m,
            args=(tuple(neighbors),),
            dynamic_shapes={
                "neighbors": ((stat, pairs), (pairs,), (pairs, stat))
            },
        )

    @unittest.skipIf(True, "Currently fails due to pbc.any() control flow")
    def testCellList(self) -> None:
        molec = make_molec(10, seed=1234)
        stat = Dim.STATIC  # type: ignore
        atoms_d = Dim("atoms")
        _ = torch.export.export(
            CellList(),
            args=(molec.atomic_nums, molec.coords, 5.2, None, None),
            dynamic_shapes={
                # specialize to 1 molecule
                "cutoff": stat,  # cutoff is fixed on export
                "species": (stat, atoms_d),
                "coords": (stat, atoms_d, stat),
                "cell": (stat, stat),
                "pbc": (stat,),
            },
        )

    def _testAllPairs(self, static: bool = False, pbc: bool = False) -> None:
        molec = make_molec(10, seed=1234)
        stat = Dim.STATIC  # type: ignore
        atoms = torch.export.Dim("atoms") if not static else stat
        if pbc:
            cell_dim = (stat, stat)
            pbc_dim = (stat,)
            cell_ten = molec.cell
            pbc_ten = molec.pbc
        else:
            cell_dim = None
            pbc_dim = None
            pbc_ten = None
            cell_ten = None
        _ = torch.export.export(
            AllPairs(),
            args=(5.2, molec.atomic_nums, molec.coords, cell_ten, pbc_ten),
            dynamic_shapes={
                # specialize to 1 molecule
                "cutoff": None,  # cutoff is fixed on export
                "species": (stat, atoms),
                "coords": (stat, atoms, stat),
                "cell": cell_dim,
                "pbc": pbc_dim,
            },
        )

    def testAllPairsStaticNoPbc(self) -> None:
        self._testAllPairs(static=True, pbc=False)

    @unittest.skipIf(True, "Currently fails due to cartesian prod")
    def testAllPairsStaticPbc(self) -> None:
        self._testAllPairs(static=True, pbc=True)

    @unittest.skipIf(True, "Currently fails due to atoms being inferred to be 10?")
    def testAllPairsDynamicNoPbc(self) -> None:
        self._testAllPairs(static=False, pbc=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
