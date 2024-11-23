# NOTE: These tests are experimental
import typing as tp
import os
import torch
from torch import Tensor
from torch.export import Dim
import unittest

from torchani._testing import ANITestCase, make_neighbors, make_molec
from torchani.utils import SYMBOLS_2X
from torchani.nn import ANINetworks, make_2x_network
from torchani.neighbors import (
    discard_outside_cutoff,
    neighbors_to_triples,
    AllPairs,
    CellList,
    Neighbors,
    Triples,
)
from torchani.aev import ANIRadial, ANIAngular, AEVComputer


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

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass()
        os.environ["TORCH_LOGS"] = ""
        os.environ["TORCHDYNAMO_VERBOSE"] = "0"
        torch._dynamo.config.capture_dynamic_output_shape_ops = False

    def tearDown(self) -> None:
        torch.compiler.reset()  # needed?


class TestCompile(ANITestCasePT2):
    def testRadial(self) -> None:
        _ = ANIRadial.like_2x().compile(fullgraph=True)

    def testAngular(self) -> None:
        _ = ANIAngular.like_2x().compile(fullgraph=True)

    def testDiscardNeighbors(self) -> None:
        _ = torch.compile(discard_outside_cutoff, fullgraph=True)

    def testNeighborsToTriples(self) -> None:
        _ = torch.compile(neighbors_to_triples, fullgraph=True)

    def testAEVComputer(self) -> None:
        _ = torch.compile(AEVComputer.like_2x().compute_from_neighbors, fullgraph=True)

    def testAtomicNetwork(self) -> None:
        _ = make_2x_network("H").compile(fullgraph=True)

    def testANINetworks(self) -> None:
        ANINetworks({s: make_2x_network(s) for s in SYMBOLS_2X}).compile(fullgraph=True)

    def tearDown(self) -> None:
        torch.compiler.reset()


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
                self, neighbors: tp.Tuple[Tensor, Tensor, Tensor], cutoff: float
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
            def forward(self, neighbors: tp.Tuple[Tensor, Tensor, Tensor]) -> Triples:
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
