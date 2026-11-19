# NOTE: These tests are experimental
import os
import tempfile
import zipfile
from pathlib import Path
import torch
from torch import Tensor
from torch.export import Dim
from torch._inductor.package import load_package
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
from torchani.compile import compile_model, _example_inputs
from torchani.nn import ANINetworks
from torchani.neighbors import (
    discard_outside_cutoff,
    neighbors_to_triples,
    AdaptiveList,
    AllPairs,
    CellList,
    FastAdaptiveList,
    FastCellList,
    Neighborlist,
    Neighbors,
    Triples,
)
from torchani.aev import ANIRadial, ANIAngular, AEVComputer
from torchani.potentials import Coulomb
from torchani.csrc import CUAEV_IS_INSTALLED, CLIST_IS_INSTALLED

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
        self._assert_compiled_neighborlists_match(cases)

    def testCppNeighborlists(self) -> None:
        self._skip_if_no_cell_list()
        cases = (
            (FastCellList(), 6),
            (FastAdaptiveList(threshold=8, threshold_nopbc=8), 6),
            (FastAdaptiveList(threshold=8, threshold_nopbc=8), 10),
        )
        self._assert_compiled_neighborlists_match(cases)

    def _assert_compiled_neighborlists_match(
        self, cases: tuple[tuple[Neighborlist, int], ...]
    ) -> None:
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

    def testCoulomb(self) -> None:
        molec = make_molec(10, seed=1234, device=self.device)
        qs = torch.rand(molec.atomic_nums.shape, device=self.device) - 0.5
        for damp_fn in ("", "erf", "tanh"):
            with self.subTest(damp_fn=damp_fn):
                torch.compiler.reset()
                pot = Coulomb(SYMBOLS, damp_fn=damp_fn).to(self.device)
                self._assert_compiled_matches(
                    pot, molec.atomic_nums, molec.coords, None, None, qs
                )

    def testModelChargesCoulomb(self) -> None:
        molecs = make_molecs(2, 6, seed=1234, device=self.device)
        molecs.atomic_nums[1, -2:] = -1
        for coulomb in ("full", "erf", "tanh"):
            with self.subTest(coulomb=coulomb):
                torch.compiler.reset()
                model = simple_aniq(
                    SYMBOLS,
                    "wb97x-631gd",
                    repulsion=False,
                    strategy="pyaev",
                    coulomb=coulomb,
                )
                self._assert_compiled_model_matches(model.to(self.device), molecs)

    def testModelChargesCoulombPBC(self) -> None:
        model = simple_aniq(
            SYMBOLS, "wb97x-631gd", repulsion=False, strategy="pyaev", coulomb="erf"
        )
        molec = make_molec(6, cell_size=6.0, pbc=True, seed=1234, device=self.device)
        self._assert_compiled_model_matches(model.to(self.device), molec)

    def testModelRepulsionPBC(self) -> None:
        model = simple_ani(SYMBOLS, "wb97x-631gd", repulsion=True, strategy="pyaev")
        molec = make_molec(6, cell_size=6.0, pbc=True, seed=1234, device=self.device)
        self._assert_compiled_model_matches(model.to(self.device), molec)

    def _skip_if_no_cell_list(self) -> None:
        if not CLIST_IS_INSTALLED:
            self.skipTest("Cell list extension is not installed")

    def testModelCppCellList(self) -> None:
        self._skip_if_no_cell_list()
        model = simple_ani(
            SYMBOLS,
            "wb97x-631gd",
            repulsion=False,
            strategy="pyaev",
            neighborlist="fast_cell_list",
        )
        for pbc in (False, True):
            with self.subTest(pbc=pbc):
                torch.compiler.reset()
                molec = make_molec(
                    6, cell_size=6.0, pbc=pbc, seed=1234, device=self.device
                )
                self._assert_compiled_model_matches(model.to(self.device), molec)

    def _skip_if_no_cuaev(self) -> None:
        if not CUAEV_IS_INSTALLED or self.device.type != "cuda":
            self.skipTest("cuAEV is only compiled for CUDA")

    def testModelCUAEVFused(self) -> None:
        self._skip_if_no_cuaev()
        model = simple_ani(
            SYMBOLS, "wb97x-631gd", repulsion=False, strategy="cuaev-fused"
        )
        molecs = make_molecs(2, 6, seed=1234, device=self.device)
        molecs.atomic_nums[1, -2:] = -1
        self._assert_compiled_model_matches(model.to(self.device), molecs)

    def testModelCUAEVHalfNbrlistPBC(self) -> None:
        self._skip_if_no_cuaev()
        model = simple_ani(SYMBOLS, "wb97x-631gd", repulsion=False, strategy="cuaev")
        molec = make_molec(6, cell_size=6.0, pbc=True, seed=1234, device=self.device)
        self._assert_compiled_model_matches(model.to(self.device), molec)

    def testCUAEVFullNbrlist(self) -> None:
        self._skip_if_no_cuaev()
        aev_computer = AEVComputer.like_2x(strategy="cuaev").to(self.device)
        molec = make_molec(10, seed=1234, device=self.device)
        elem_idxs = make_elem_idxs(1, 10, seed=1234).to(self.device)
        neighbors = AllPairs()(5.1, elem_idxs, molec.coords)
        full_nbrlist = AEVComputer._half_to_full_nbrlist(neighbors.indices)
        self._assert_compiled_matches(
            aev_computer._compute_cuaev_with_full_nbrlist,
            elem_idxs,
            molec.coords,
            *full_nbrlist,
        )

    @unittest.skip("Data-dependent branching and module state mutation")
    def testVerletCellList(self) -> None:
        pass

    @unittest.skip("mnp ops have no fake kernels, and TorchBind classes no fake class")
    def testCppExtensions(self) -> None:
        pass


class _CoulombMLMM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = simple_aniq(
            SYMBOLS, "wb97x-631gd", repulsion=False, strategy="pyaev"
        )

    def compute_mlmm(
        self,
        species_coordinates: tuple[Tensor, Tensor],
        mm_coords: Tensor,
        mm_charges: Tensor,
        charge: Tensor | None = None,
        spin_mult: Tensor | None = None,
        mm_cutoff: Tensor | None = None,
        pairs_12: Tensor | None = None,
        pairs_13: Tensor | None = None,
        pairs_14: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        coords = species_coordinates[1]
        output = self.model(species_coordinates, charge=charge)
        # cdist branches on the number of points, so it can't have dynamic shapes
        delta = mm_coords.unsqueeze(2) - coords.unsqueeze(1)
        distances = torch.linalg.norm(delta, dim=-1)
        pair_coulomb = (
            mm_charges.unsqueeze(2) * output.atomic_charges.unsqueeze(1) / distances
        )
        return output.energies + pair_coulomb.sum((1, 2)), output.atomic_charges

    def compute_multipole_mlmm(
        self,
        species_coordinates: tuple[Tensor, Tensor],
        mm_coords: Tensor,
        mm_charges: Tensor,
        mm_dipoles: Tensor,
        mm_quads: Tensor,
        charge: Tensor | None = None,
        spin_mult: Tensor | None = None,
        mm_cutoff: Tensor | None = None,
        pairs_12: Tensor | None = None,
        pairs_13: Tensor | None = None,
        pairs_14: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        energies, atomic_charges = self.compute_mlmm(
            species_coordinates, mm_coords, mm_charges, charge
        )
        delta = mm_coords.unsqueeze(2) - species_coordinates[1].unsqueeze(1)
        distances = torch.linalg.norm(delta, dim=-1)
        dipole_potential = -(mm_dipoles.unsqueeze(2) * delta).sum(-1) / distances**3
        delta_outer = delta.unsqueeze(-1) * delta.unsqueeze(-2)
        quad_contraction = (mm_quads.unsqueeze(2) * delta_outer).sum((-1, -2))
        quad_potential = 1.5 * quad_contraction / distances**5
        mm_potential = dipole_potential + quad_potential
        pair_energies = mm_potential * atomic_charges.unsqueeze(1)
        return energies + pair_energies.sum((1, 2)), atomic_charges


class TestCompileModel(ANITestCasePT2):
    # Compiling is slow (~30 s per package), so only a few configs are compiled

    def setUp(self) -> None:
        super().setUp()
        self._tmp_dir = tempfile.TemporaryDirectory()
        self._package_path = Path(self._tmp_dir.name, "model.pt2")

    def tearDown(self) -> None:
        super().tearDown()
        self._tmp_dir.cleanup()

    def _model_names(self, path: Path) -> set[str]:
        names = set()
        for record in zipfile.ZipFile(path).namelist():
            parts = record.split("/")
            if "aotinductor" in parts:
                start = parts.index("aotinductor") + 1
                names.add("/".join(parts[start:-1]))
        return names

    def _assert_package_matches(
        self,
        name: str,
        args: tuple[Tensor, ...],
        expect_energies: Tensor,
        expect_forces: Tensor,
    ) -> Tensor:
        energies, forces, atomic_charges = load_package(
            str(self._package_path), name
        )(*args)
        self.assertEqual(energies, expect_energies)
        self.assertEqual(forces, expect_forces)
        return atomic_charges

    def _eager_energies_forces(
        self, model: torch.nn.Module, species: Tensor, coords: Tensor, **kwargs
    ) -> tuple[Tensor, Tensor]:
        coords = coords.clone().requires_grad_(True)
        energies = model((species, coords), **kwargs).energies
        forces = -torch.autograd.grad(energies.sum(), coords)[0]
        return energies.detach(), forces

    def testSinglePointEntrypoints(self) -> None:
        # AllPairs is swapped internally for CellList, the input model is unchanged
        model = simple_ani(SYMBOLS, "wb97x-631gd", repulsion=False, strategy="pyaev")
        compile_model(
            model,
            self._package_path,
            device="cpu",
            precision="f64",
            configs=("single_point", "single_point_pbc"),
        )
        self.assertIs(type(model.neighborlist), AllPairs)
        self.assertEqual(
            self._model_names(self._package_path),
            {"single_point/cpu/f64", "single_point_pbc/cpu/f64"},
        )
        model = model.double()
        pbc = torch.ones(3, dtype=torch.bool)
        net_charge = torch.tensor([0])
        # Different sizes check that the number of atoms is dynamic
        for atoms in (20, 31):
            species, coords, cell, _, _ = _example_inputs(model, atoms=atoms)
            expect = self._eager_energies_forces(model, species, coords)
            atomic_charges = self._assert_package_matches(
                "single_point/cpu/f64", (species, coords, net_charge), *expect
            )
            self.assertEqual(atomic_charges.shape, (1, 0))
            expect = self._eager_energies_forces(
                model, species, coords, cell=cell, pbc=pbc
            )
            self._assert_package_matches(
                "single_point_pbc/cpu/f64",
                (species, coords, cell, pbc, net_charge),
                *expect,
            )

    def testExternalNeighborsCharges(self) -> None:
        model = simple_aniq(SYMBOLS, "wb97x-631gd", repulsion=False, strategy="pyaev")
        compile_model(
            model,
            self._package_path,
            device="cpu",
            precision="f32",
            configs="external_neighbors",
        )
        self.assertEqual(
            self._model_names(self._package_path), {"external_neighbors/cpu/f32"}
        )
        species, coords, _, neighbor_idxs, shifts = _example_inputs(model)
        _coords = coords.clone().requires_grad_(True)
        result = model.compute_from_external_neighbors(
            species, _coords, neighbor_idxs, shifts
        )
        forces = -torch.autograd.grad(result["energies"].sum(), _coords)[0]
        atomic_charges = self._assert_package_matches(
            "external_neighbors/cpu/f32",
            (species, coords, neighbor_idxs, shifts, torch.tensor([0])),
            result["energies"].detach(),
            forces,
        )
        self.assertEqual(atomic_charges, result["atomic_charges"].detach())

    def testCustomMLMM(self) -> None:
        model = _CoulombMLMM()
        compile_model(
            model,
            self._package_path,
            device="cpu",
            precision="f64",
            configs="custom_mlmm",
            symbols=SYMBOLS,
            cutoff=model.model.cutoff,
        )
        self.assertEqual(
            self._model_names(self._package_path), {"custom_mlmm/cpu/f64"}
        )
        model = model.double()
        runner = load_package(str(self._package_path), "custom_mlmm/cpu/f64")
        net_charge = torch.tensor([0])
        spin_mult = torch.tensor([1])
        mm_cutoff = torch.tensor([0.0], dtype=torch.double)
        # Different sizes check that the number of ML and MM atoms, and pairs is dynamic
        for atoms, mm_atoms, pairs_num in ((20, 7, 0), (31, 25, 5)):
            species, coords, cell, _, _ = _example_inputs(model.model, atoms=atoms)
            gen = torch.Generator().manual_seed(atoms)
            pairs_12 = torch.randint(0, atoms, (2, pairs_num), generator=gen)
            pairs_13 = torch.randint(0, atoms, (2, pairs_num + 1), generator=gen)
            pairs_14 = torch.randint(0, atoms, (2, pairs_num + 2), generator=gen)
            mm_coords = torch.rand(1, mm_atoms, 3, generator=gen, dtype=torch.double)
            mm_coords = (mm_coords + 1.0) * cell.diagonal()
            mm_charges = torch.rand(1, mm_atoms, generator=gen, dtype=torch.double)
            _coords = coords.clone().requires_grad_(True)
            _mm_coords = mm_coords.clone().requires_grad_(True)
            energies, atomic_charges = model.compute_mlmm(
                (species, _coords), _mm_coords, mm_charges
            )
            grads = torch.autograd.grad(energies.sum(), (_coords, _mm_coords))
            output = runner(
                species,
                coords,
                mm_coords,
                mm_charges,
                net_charge,
                spin_mult,
                mm_cutoff,
                pairs_12,
                pairs_13,
                pairs_14,
            )
            self.assertEqual(output[0], energies.detach())
            self.assertEqual(output[1], -grads[0])
            self.assertEqual(output[2], -grads[1])
            self.assertEqual(output[3], atomic_charges.detach())

    def testMultipoleMLMM(self) -> None:
        model = _CoulombMLMM()
        compile_model(
            model,
            self._package_path,
            device="cpu",
            precision="f64",
            configs="multipole_mlmm",
            symbols=SYMBOLS,
            cutoff=model.model.cutoff,
        )
        self.assertEqual(
            self._model_names(self._package_path), {"multipole_mlmm/cpu/f64"}
        )
        model = model.double()
        runner = load_package(str(self._package_path), "multipole_mlmm/cpu/f64")
        net_charge = torch.tensor([0])
        spin_mult = torch.tensor([1])
        mm_cutoff = torch.tensor([0.0], dtype=torch.double)
        # Different sizes check that the number of ML and MM atoms, and pairs is dynamic
        for atoms, mm_atoms, pairs_num in ((20, 7, 0), (31, 25, 5)):
            species, coords, cell, _, _ = _example_inputs(model.model, atoms=atoms)
            gen = torch.Generator().manual_seed(atoms)
            pairs_12 = torch.randint(0, atoms, (2, pairs_num), generator=gen)
            pairs_13 = torch.randint(0, atoms, (2, pairs_num + 1), generator=gen)
            pairs_14 = torch.randint(0, atoms, (2, pairs_num + 2), generator=gen)
            mm_coords = torch.rand(1, mm_atoms, 3, generator=gen, dtype=torch.double)
            mm_coords = (mm_coords + 1.0) * cell.diagonal()
            mm_charges = torch.rand(1, mm_atoms, generator=gen, dtype=torch.double)
            mm_dipoles = torch.rand(1, mm_atoms, 3, generator=gen, dtype=torch.double)
            mm_quads = torch.rand(1, mm_atoms, 3, 3, generator=gen, dtype=torch.double)
            _coords = coords.clone().requires_grad_(True)
            _mm_coords = mm_coords.clone().requires_grad_(True)
            energies, atomic_charges = model.compute_multipole_mlmm(
                (species, _coords), _mm_coords, mm_charges, mm_dipoles, mm_quads
            )
            grads = torch.autograd.grad(energies.sum(), (_coords, _mm_coords))
            output = runner(
                species,
                coords,
                mm_coords,
                mm_charges,
                mm_dipoles,
                mm_quads,
                net_charge,
                spin_mult,
                mm_cutoff,
                pairs_12,
                pairs_13,
                pairs_14,
            )
            self.assertEqual(output[0], energies.detach())
            self.assertEqual(output[1], -grads[0])
            self.assertEqual(output[2], -grads[1])
            self.assertEqual(output[3], atomic_charges.detach())

    def testCudaCuaev(self) -> None:
        if not CUAEV_IS_INSTALLED or not torch.cuda.is_available():
            self.skipTest("cuAEV compilation requires CUDA and the cuAEV extension")
        # The input strategy is ignored, CUDA uses cuAEV if the model supports it
        model = simple_ani(
            SYMBOLS,
            "wb97x-631gd",
            repulsion=False,
            strategy="pyaev",
            neighborlist="cell_list",
        )
        compile_model(
            model,
            self._package_path,
            device="cuda",
            precision="f32",
            configs="single_point",
        )
        # Ops without a C shim, such as cuAEV's, are listed in the package json files
        archive = zipfile.ZipFile(self._package_path)
        archive_text = ""
        for record in archive.namelist():
            if record.endswith((".json", ".cpp")):
                archive_text += archive.read(record).decode()
        self.assertTrue("cuaev::run" in archive_text)
        model = model.to("cuda")
        species, coords, _, _, _ = _example_inputs(model)
        expect = self._eager_energies_forces(model, species, coords)
        net_charge = torch.tensor([0], device="cuda")
        runner = load_package(str(self._package_path), "single_point/cuda/f32")
        energies, forces, _ = runner(species, coords, net_charge)
        self.assertEqual(energies, expect[0], rtol=1e-4, atol=1e-4)
        self.assertEqual(forces, expect[1], rtol=1e-4, atol=1e-4)

    def testInvalidArgs(self) -> None:
        model = simple_ani(SYMBOLS, "wb97x-631gd", repulsion=False, strategy="pyaev")
        with self.assertRaisesRegex(ValueError, "compute_mlmm"):
            compile_model(model, self._package_path, configs="custom_mlmm")
        with self.assertRaisesRegex(ValueError, "compute_multipole_mlmm"):
            compile_model(model, self._package_path, configs="multipole_mlmm")
        with self.assertRaisesRegex(ValueError, "compute_from_external_neighbors"):
            compile_model(
                torch.nn.Linear(1, 1),  # type: ignore[arg-type]
                self._package_path,
                configs="external_neighbors",
            )
        model = simple_ani(
            SYMBOLS,
            "wb97x-631gd",
            repulsion=False,
            strategy="pyaev",
            neighborlist="verlet_cell_list",
        )
        with self.assertRaisesRegex(ValueError, "VerletCellList"):
            compile_model(model, self._package_path)


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
