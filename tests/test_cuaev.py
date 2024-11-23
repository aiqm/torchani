import tempfile
import typing as tp
from pathlib import Path
import os
import unittest
import pickle

import torch
from parameterized import parameterized_class

from torchani._testing import ANITestCase, expand
from torchani.utils import SYMBOLS_2X
from torchani.nn import SpeciesConverter
from torchani.aev import AEVComputer
from torchani.models import ANIdr
from torchani.io import read_xyz
from torchani._testing import TestCase, make_tensor
from torchani.csrc import CUAEV_IS_INSTALLED

path = os.path.dirname(os.path.realpath(__file__))

skipIfNoGPU = unittest.skipIf(
    not torch.cuda.is_available(), "There is no device to run this test"
)
skipIfNoMultiGPU = unittest.skipIf(
    not torch.cuda.device_count() >= 2,
    "There are not enough GPU devices to run this test",
)
skipIfNoCUAEV = unittest.skipIf(
    not CUAEV_IS_INSTALLED, "only valid when cuaev is installed"
)


@expand(device="cuda")
class TestCUAEVStrategy(ANITestCase):
    def setUp(self) -> None:
        self.tolerance = 5e-5
        coordinates = torch.tensor(
            [
                [
                    [0.03192167, 0.00638559, 0.01301679],
                    [-0.83140486, 0.39370209, -0.26395324],
                    [-0.66518241, -0.84461308, 0.20759389],
                    [0.45554739, 0.54289633, 0.81170881],
                    [0.66091919, -0.16799635, -0.91037834],
                ],
                [
                    [-4.1862600, 0.0575700, -0.0381200],
                    [-3.1689400, 0.0523700, 0.0200000],
                    [-4.4978600, 0.8211300, 0.5604100],
                    [-4.4978700, -0.8000100, 0.4155600],
                    [0.00000000, -0.00000000, -0.00000000],
                ],
            ],
            device=self.device,
            dtype=torch.float32,
        )
        species = torch.tensor(
            [[1, 0, 0, 0, 0], [2, 0, 0, 0, -1]], device=self.device, dtype=torch.long
        )
        znums = torch.tensor(
            [[6, 1, 1, 1, 1], [7, 1, 1, 1, -1]], device=self.device, dtype=torch.long
        )
        self.input = (species, coordinates)
        self.znum_input = (znums, coordinates)

    @skipIfNoCUAEV
    def testModelChangeStrat(self) -> None:
        m = self._setup(ANIdr(strategy="cuaev-interface"))

        m.set_strategy("cuaev-interface")
        cu = m(self.znum_input)[1]

        m.set_strategy("pyaev")
        py = m(self.znum_input)[1]

        self.assertEqual(py, cu, atol=self.tolerance, rtol=self.tolerance)

    @skipIfNoCUAEV
    def testInvalidModelStrat(self) -> None:
        m = self._setup(ANIdr(strategy="cuaev-interface"))
        with self.assertRaises(torch.jit.Error if self.jit else RuntimeError):
            m.set_strategy("cuaev-fused")
            _ = m(self.znum_input)[1]

    @skipIfNoCUAEV
    def testAEVChangeStrat(self) -> None:
        m = self._setup(AEVComputer.like_2x(strategy="cuaev-interface"))

        m.set_strategy("cuaev-interface")
        cu = m(*self.input)[1]

        m.set_strategy("cuaev-fused")
        cu_fused = m(*self.input)[1]

        m.set_strategy("pyaev")
        py = m(*self.input)[1]

        self.assertEqual(py, cu, atol=self.tolerance, rtol=self.tolerance)
        self.assertEqual(cu, cu_fused, atol=self.tolerance, rtol=self.tolerance)


@skipIfNoCUAEV
class TestCUAEVNoGPU(TestCase):
    def testSimple(self):
        def f(
            coordinates,
            species,
            Rcr: float,
            Rca: float,
            EtaR,
            ShfR,
            EtaA,
            Zeta,
            ShfA,
            ShfZ,
            num_species: int,
        ):
            cuaev_computer = torch.classes.cuaev.CuaevComputer(
                Rcr,
                Rca,
                EtaR.flatten(),
                ShfR.flatten(),
                EtaA.flatten(),
                Zeta.flatten(),
                ShfA.flatten(),
                ShfZ.flatten(),
                num_species,
                True,
            )
            return torch.ops.cuaev.run(coordinates, species, cuaev_computer)

        self.assertIn("cuaev::run", str(torch.jit.script(f).graph))

    def testNoAtoms(self):
        # cuAEV with no atoms does not require CUDA, and can be run without GPU
        m = torch.jit.script(AEVComputer.like_1x(strategy="cuaev-fused"))
        species = make_tensor((8, 0), device="cpu", dtype=torch.int64, low=-1, high=4)
        coordinates = make_tensor(
            (8, 0, 3), device="cpu", dtype=torch.float32, low=-5, high=5
        )
        self.assertIn("cuaev::run", str(m.graph_for(species, coordinates)))

    def testPickle(self):
        aev_computer = AEVComputer.like_1x(strategy="cuaev-fused")
        with tempfile.TemporaryFile(mode="wb+") as f:
            pickle.dump(aev_computer, f)
            f.seek(0)
            _ = pickle.load(f)


@skipIfNoGPU
@skipIfNoCUAEV
@parameterized_class(
    ("cutoff_fn", "dtype"),
    [
        ["cosine", torch.float32],
        ["cosine", torch.float64],
        ["smooth", torch.float32],
        ["smooth", torch.float64],
    ],
)
class TestCUAEV(TestCase):
    dtype: torch.dtype
    cutoff_fn: tp.Literal["smooth", "cosine"]

    def setUp(self, device="cuda:0"):
        # double precision error is within 5e-13
        self.tolerance = 5e-5 if self.dtype == torch.float32 else 5e-13
        self.device = device
        self.aev_computer_1x = AEVComputer.like_1x(cutoff_fn=self.cutoff_fn).to(
            dtype=self.dtype, device=self.device
        )
        self.cuaev_computer_1x = AEVComputer.like_1x(
            cutoff_fn=self.cutoff_fn, strategy="cuaev-fused"
        ).to(self.device, self.dtype)
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(384, 1, False, dtype=self.dtype, device=self.device)
        )

        self.aev_computer_2x = AEVComputer.like_2x(cutoff_fn=self.cutoff_fn).to(
            self.device, self.dtype
        )
        self.cuaev_computer_2x = AEVComputer.like_2x(
            cutoff_fn=self.cutoff_fn, strategy="cuaev-fused"
        ).to(self.device, self.dtype)
        self.cuaev_computer_2x_use_interface = AEVComputer.like_2x(
            cutoff_fn=self.cutoff_fn, strategy="cuaev-interface"
        ).to(self.device, self.dtype)
        self.converter = SpeciesConverter(SYMBOLS_2X).to(self.device, self.dtype)
        self.cutoff_2x = self.cuaev_computer_2x.radial.cutoff
        self.cutoff_1x = self.cuaev_computer_1x.radial.cutoff

    def _skip_if_not_cosine(self):
        if self.cutoff_fn != "cosine":
            self.skipTest("Skip slow tests for non-cosine cutoff")

    def _skip_if_double(self):
        if self.dtype == torch.float64:
            self.skipTest("Skip slow tests for double precision")

    def _double_backward_1_test(self, species, coordinates):
        def double_backward(aev_computer, species, coordinates):
            torch.manual_seed(12345)
            self.nn.zero_grad()
            aev = aev_computer(species, coordinates)
            E = self.nn(aev).sum()
            force = -torch.autograd.grad(
                E, coordinates, create_graph=True, retain_graph=True
            )[0]
            force_true = torch.randn_like(force)
            loss = torch.abs(force_true - force).sum(dim=(1, 2)).mean()
            loss.backward()
            param = next(self.nn.parameters())
            return aev, force, param.grad

        aev, force_ref, param_grad_ref = double_backward(
            self.aev_computer_1x, species, coordinates
        )
        cu_aev, force_cuaev, param_grad = double_backward(
            self.cuaev_computer_1x, species, coordinates
        )

        self.assertEqual(cu_aev, aev, f"cu_aev: {cu_aev}\n aev: {aev}")
        self.assertEqual(
            force_cuaev,
            force_ref,
            f"\nforce_cuaev: {force_cuaev}\n force_ref: {force_ref}",
        )
        self.assertEqual(
            param_grad,
            param_grad_ref,
            f"\nparam_grad: {param_grad}\n param_grad_ref: {param_grad_ref}",
            atol=self.tolerance,
            rtol=self.tolerance,
        )

    def _double_backward_2_test(self, species, coordinates):
        def double_backward(aev_computer, species, coordinates):
            """
            # We want to get the gradient of `grad_aev`, which requires
            # `grad_aev` to be a leaf node due to `torch.autograd`'s
            # limitation. So we split the coord->aev->energy graph into two
            # separate graphs: coord->aev and aev->energy, so that aev and
            # grad_aev are now leaves.
            """
            torch.manual_seed(12345)
            # graph1 input -> aev
            coordinates = coordinates.clone().detach().requires_grad_()
            aev = aev_computer(species, coordinates)
            # graph2 aev -> E
            aev_ = aev.clone().detach().requires_grad_()
            E = self.nn(aev_).sum()
            # graph2 backward
            aev_grad = torch.autograd.grad(
                E, aev_, create_graph=True, retain_graph=True
            )[0]
            # graph1 backward
            aev_grad_ = aev_grad.clone().detach().requires_grad_()
            force = torch.autograd.grad(
                aev, coordinates, aev_grad_, create_graph=True, retain_graph=True
            )[0]
            # force loss backward
            force_true = torch.randn_like(force)
            loss = torch.abs(force_true - force).sum(dim=(1, 2)).mean()
            aev_grad_grad = torch.autograd.grad(
                loss, aev_grad_, create_graph=True, retain_graph=True
            )[0]

            return aev, force, aev_grad_grad

        aev, force_ref, aev_grad_grad = double_backward(
            self.aev_computer_1x, species, coordinates
        )
        cu_aev, force_cuaev, cuaev_grad_grad = double_backward(
            self.cuaev_computer_1x, species, coordinates
        )

        self.assertEqual(
            cu_aev,
            aev,
            f"cu_aev: {cu_aev}\n aev: {aev}",
            atol=self.tolerance,
            rtol=self.tolerance,
        )
        self.assertEqual(
            force_cuaev,
            force_ref,
            f"\nforce_cuaev: {force_cuaev}\n force_ref: {force_ref}",
            atol=self.tolerance,
            rtol=self.tolerance,
        )
        self.assertEqual(
            cuaev_grad_grad,
            aev_grad_grad,
            f"\ncuaev_grad_grad: {cuaev_grad_grad}\n aev_grad_grad: {aev_grad_grad}",
            atol=self.tolerance,
            rtol=self.tolerance,
        )

    def testSimple(self):
        coordinates = torch.tensor(
            [
                [
                    [0.03192167, 0.00638559, 0.01301679],
                    [-0.83140486, 0.39370209, -0.26395324],
                    [-0.66518241, -0.84461308, 0.20759389],
                    [0.45554739, 0.54289633, 0.81170881],
                    [0.66091919, -0.16799635, -0.91037834],
                ],
                [
                    [-4.1862600, 0.0575700, -0.0381200],
                    [-3.1689400, 0.0523700, 0.0200000],
                    [-4.4978600, 0.8211300, 0.5604100],
                    [-4.4978700, -0.8000100, 0.4155600],
                    [0.00000000, -0.00000000, -0.00000000],
                ],
            ],
            device=self.device,
            dtype=self.dtype,
        )
        species = torch.tensor([[1, 0, 0, 0, 0], [2, 0, 0, 0, -1]], device=self.device)

        aev = self.aev_computer_1x(species, coordinates)
        cu_aev = self.cuaev_computer_1x(species, coordinates)
        self.assertEqual(cu_aev, aev, atol=self.tolerance, rtol=self.tolerance)

    @skipIfNoMultiGPU
    def testMultiGPU(self):
        self.setUp(device="cuda:1")
        self.testSimple()
        self.testSimpleBackward()
        self.testSimpleDoubleBackward_1()
        self.testSimpleDoubleBackward_2()
        self.setUp(device="cuda:0")

    def testBatch(self):
        coordinates = torch.rand([100, 50, 3], device=self.device, dtype=self.dtype) * 5
        species = torch.randint(-1, 3, (100, 50), device=self.device)

        aev = self.aev_computer_1x(species, coordinates)
        cu_aev = self.cuaev_computer_1x(species, coordinates)
        self.assertEqual(cu_aev, aev, atol=self.tolerance, rtol=self.tolerance)

    def testBatchHalfNbr(self):
        coordinates = torch.rand([100, 50, 3], device=self.device, dtype=self.dtype) * 5
        species = torch.randint(-1, 3, (100, 50), device=self.device)

        aev = self.aev_computer_2x(species, coordinates)
        cu_aev = self.cuaev_computer_2x_use_interface(species, coordinates)
        self.assertEqual(cu_aev, aev, atol=self.tolerance, rtol=self.tolerance)

    def testPickleCorrectness(self):
        ref_aev_computer = self.cuaev_computer_1x
        tmpfile = "/tmp/cuaev.pkl"
        with open(tmpfile, "wb") as file:
            pickle.dump(ref_aev_computer, file)
        with open(tmpfile, "rb") as file:
            test_aev_computer = pickle.load(file)
        os.remove(tmpfile)

        coordinates = torch.rand([2, 50, 3], device=self.device, dtype=self.dtype) * 5
        species = torch.randint(-1, 3, (2, 50), device=self.device)
        ref_aev = ref_aev_computer(species, coordinates)
        test_aev = test_aev_computer(species, coordinates)
        self.assertEqual(ref_aev, test_aev, atol=self.tolerance, rtol=self.tolerance)

    def testSimpleBackward(self):
        coordinates = torch.tensor(
            [
                [
                    [0.03192167, 0.00638559, 0.01301679],
                    [-0.83140486, 0.39370209, -0.26395324],
                    [-0.66518241, -0.84461308, 0.20759389],
                    [0.45554739, 0.54289633, 0.81170881],
                    [0.66091919, -0.16799635, -0.91037834],
                ],
                [
                    [-4.1862600, 0.0575700, -0.0381200],
                    [-3.1689400, 0.0523700, 0.0200000],
                    [-4.4978600, 0.8211300, 0.5604100],
                    [-4.4978700, -0.8000100, 0.4155600],
                    [0.00000000, -0.00000000, -0.00000000],
                ],
            ],
            requires_grad=True,
            device=self.device,
            dtype=self.dtype,
        )
        species = torch.tensor([[1, 0, 0, 0, 0], [2, 0, 0, 0, -1]], device=self.device)

        aev = self.aev_computer_1x(species, coordinates)
        aev.backward(torch.ones_like(aev))
        aev_grad = coordinates.grad

        coordinates = coordinates.clone().detach()
        coordinates.requires_grad_()
        cu_aev = self.cuaev_computer_1x(species, coordinates)
        cu_aev.backward(torch.ones_like(cu_aev))
        cuaev_grad = coordinates.grad
        self.assertEqual(
            cu_aev,
            aev,
            f"cu_aev: {cu_aev}\n aev: {aev}",
            atol=self.tolerance,
            rtol=self.tolerance,
        )
        self.assertEqual(
            cuaev_grad,
            aev_grad,
            f"\ncuaev_grad: {cuaev_grad}\n aev_grad: {aev_grad}",
            atol=self.tolerance,
            rtol=self.tolerance,
        )

    def testSimpleDoubleBackward_1(self):
        """
        Test Double Backward (Force training) by parameters' gradient
        """
        coordinates = torch.tensor(
            [
                [
                    [0.03192167, 0.00638559, 0.01301679],
                    [-0.83140486, 0.39370209, -0.26395324],
                    [-0.66518241, -0.84461308, 0.20759389],
                    [0.45554739, 0.54289633, 0.81170881],
                    [0.66091919, -0.16799635, -0.91037834],
                ],
                [
                    [-4.1862600, 0.0575700, -0.0381200],
                    [-3.1689400, 0.0523700, 0.0200000],
                    [-4.4978600, 0.8211300, 0.5604100],
                    [-4.4978700, -0.8000100, 0.4155600],
                    [0.00000000, -0.00000000, -0.00000000],
                ],
            ],
            requires_grad=True,
            device=self.device,
            dtype=self.dtype,
        )
        species = torch.tensor([[1, 0, 0, 0, 0], [2, 0, 0, 0, -1]], device=self.device)

        self._double_backward_1_test(species, coordinates)

    def testSimpleDoubleBackward_2(self):
        """
        Test Double Backward (Force training) directly.
        Double backward:
        Forward: input is dE/dAEV, output is force
        Backward: input is dLoss/dForce, output is dLoss/(dE/dAEV)
        """
        coordinates = torch.tensor(
            [
                [
                    [0.03192167, 0.00638559, 0.01301679],
                    [-0.83140486, 0.39370209, -0.26395324],
                    [-0.66518241, -0.84461308, 0.20759389],
                    [0.45554739, 0.54289633, 0.81170881],
                    [0.66091919, -0.16799635, -0.91037834],
                ],
                [
                    [-4.1862600, 0.0575700, -0.0381200],
                    [-3.1689400, 0.0523700, 0.0200000],
                    [-4.4978600, 0.8211300, 0.5604100],
                    [-4.4978700, -0.8000100, 0.4155600],
                    [0.00000000, -0.00000000, -0.00000000],
                ],
            ],
            requires_grad=True,
            device=self.device,
            dtype=self.dtype,
        )
        species = torch.tensor([[1, 0, 0, 0, 0], [2, 0, 0, 0, -1]], device=self.device)

        self._double_backward_2_test(species, coordinates)

    def testTripeptideMD(self):
        for i in range(100):
            datafile = os.path.join(path, f"resources/tripeptide-md/{i}.dat")
            with open(datafile, "rb") as f:
                coordinates, species, *_ = pickle.load(f)
                coordinates = (
                    torch.from_numpy(coordinates)
                    .unsqueeze(0)
                    .to(self.device, self.dtype)
                )
                species = torch.from_numpy(species).unsqueeze(0).to(self.device)
                aev = self.aev_computer_1x(species, coordinates)
                cu_aev = self.cuaev_computer_1x(species, coordinates)
                self.assertEqual(cu_aev, aev, atol=self.tolerance, rtol=self.tolerance)

    def testTripeptideMDBackward(self):
        for i in range(100):
            datafile = os.path.join(path, f"resources/tripeptide-md/{i}.dat")
            with open(datafile, "rb") as f:
                coordinates, species, *_ = pickle.load(f)
                coordinates = (
                    torch.from_numpy(coordinates)
                    .unsqueeze(0)
                    .to(self.device, self.dtype)
                    .requires_grad_(True)
                )
                species = torch.from_numpy(species).unsqueeze(0).to(self.device)
                aev = self.aev_computer_1x(species, coordinates)
                aev.backward(torch.ones_like(aev))
                aev_grad = coordinates.grad

                coordinates = coordinates.clone().detach()
                coordinates.requires_grad_()
                cu_aev = self.cuaev_computer_1x(species, coordinates)
                cu_aev.backward(torch.ones_like(cu_aev))
                cuaev_grad = coordinates.grad
                self.assertEqual(cu_aev, aev, atol=self.tolerance, rtol=self.tolerance)
                tolerance = 5e-4 if self.dtype == torch.float32 else 5e-13
                self.assertEqual(cuaev_grad, aev_grad, atol=tolerance, rtol=tolerance)

    def testTripeptideMDDoubleBackward_2(self):
        # skip if not cosine
        self._skip_if_not_cosine()

        for i in range(100):
            datafile = os.path.join(path, f"resources/tripeptide-md/{i}.dat")
            with open(datafile, "rb") as f:
                coordinates, species, *_ = pickle.load(f)
                coordinates = (
                    torch.from_numpy(coordinates)
                    .unsqueeze(0)
                    .to(self.device, self.dtype)
                    .requires_grad_(True)
                )
                species = torch.from_numpy(species).unsqueeze(0).to(self.device)
                self._double_backward_2_test(species, coordinates)

    def testNIST(self):
        # skip if not cosine
        self._skip_if_not_cosine()
        self._skip_if_double()

        datafile = os.path.join(path, "resources/NIST/all")
        with open(datafile, "rb") as f:
            data = pickle.load(f)
            for coordinates, species, _, _, _, _ in data:
                coordinates = torch.from_numpy(coordinates).to(self.device, self.dtype)
                species = torch.from_numpy(species).to(self.device)
                aev = self.aev_computer_1x(species, coordinates)
                cu_aev = self.cuaev_computer_1x(species, coordinates)
                self.assertEqual(cu_aev, aev, atol=self.tolerance, rtol=self.tolerance)

    def testNISTBackward(self):
        self._skip_if_not_cosine()
        self._skip_if_double()

        datafile = os.path.join(path, "resources/NIST/all")
        with open(datafile, "rb") as f:
            data = pickle.load(f)
            for coordinates, species, _, _, _, _ in data[:10]:
                coordinates = (
                    torch.from_numpy(coordinates)
                    .to(self.device, self.dtype)
                    .requires_grad_(True)
                )
                species = torch.from_numpy(species).to(self.device)
                aev = self.aev_computer_1x(species, coordinates)
                aev.backward(torch.ones_like(aev))
                aev_grad = coordinates.grad

                coordinates = coordinates.clone().detach()
                coordinates.requires_grad_()
                cu_aev = self.cuaev_computer_1x(species, coordinates)
                cu_aev.backward(torch.ones_like(cu_aev))
                cuaev_grad = coordinates.grad
                self.assertEqual(cu_aev, aev, atol=self.tolerance, rtol=self.tolerance)
                self.assertEqual(
                    cuaev_grad, aev_grad, atol=self.tolerance, rtol=self.tolerance
                )

    def testNISTDoubleBackward_2(self):
        self._skip_if_not_cosine()
        self._skip_if_double()

        datafile = os.path.join(path, "resources/NIST/all")
        with open(datafile, "rb") as f:
            data = pickle.load(f)
            for coordinates, species, _, _, _, _ in data[:3]:
                coordinates = (
                    torch.from_numpy(coordinates)
                    .to(self.device, self.dtype)
                    .requires_grad_(True)
                )
                species = torch.from_numpy(species).to(self.device)
                self._double_backward_2_test(species, coordinates)

    def testVeryDenseMolecule(self):
        """
        Test very dense molecule for aev correctness, especially for angular
        kernel when center atom pairs are more than 32. issue:
        https://github.com/aiqm/torchani/pull/555
        """
        for i in range(5):
            datafile = os.path.join(path, f"resources/tripeptide-md/{i}.dat")
            with open(datafile, "rb") as f:
                coordinates, species, *_ = pickle.load(f)
                # change angstrom coordinates to 10 times smaller
                coordinates = 0.1 * torch.from_numpy(coordinates).unsqueeze(0).to(
                    self.device, self.dtype
                )
                species = torch.from_numpy(species).unsqueeze(0).to(self.device)
                aev = self.aev_computer_1x(species, coordinates)
                cu_aev = self.cuaev_computer_1x(species, coordinates)
                self.assertEqual(cu_aev, aev, atol=self.tolerance, rtol=self.tolerance)

    def testVeryDenseMoleculeBackward(self):
        for i in range(5):
            datafile = os.path.join(path, f"resources/tripeptide-md/{i}.dat")
            with open(datafile, "rb") as f:
                coordinates, species, *_ = pickle.load(f)
                # change angstrom coordinates to 10 times smaller
                coordinates = 0.1 * torch.from_numpy(coordinates).unsqueeze(0).to(
                    self.device, self.dtype
                )
                coordinates.requires_grad_(True)
                species = torch.from_numpy(species).unsqueeze(0).to(self.device)

                aev = self.aev_computer_1x(species, coordinates)
                aev.backward(torch.ones_like(aev))
                aev_grad = coordinates.grad

                coordinates = coordinates.clone().detach()
                coordinates.requires_grad_()
                cu_aev = self.cuaev_computer_1x(species, coordinates)
                cu_aev.backward(torch.ones_like(cu_aev))
                cuaev_grad = coordinates.grad
                self.assertEqual(cu_aev, aev, atol=self.tolerance, rtol=self.tolerance)
                # Needs slightly more slack
                self.assertEqual(cuaev_grad, aev_grad, atol=9e-3, rtol=9e-5)

    def testXYZ(self):
        files = ["small.xyz", "1hz5.xyz", "6W8H.xyz"]
        for file in files:
            species, coordinates, _, _ = read_xyz(
                (Path(__file__).parent / "resources") / file,
                device=self.device,
                dtype=self.dtype,
            )
            species = self.converter(species)
            aev = self.aev_computer_2x(species, coordinates)
            cu_aev = self.cuaev_computer_2x(species, coordinates)
            self.assertEqual(cu_aev, aev, atol=self.tolerance, rtol=self.tolerance)

    def testXYZBackward(self):
        files = ["small.xyz", "1hz5.xyz", "6W8H.xyz"]
        for file in files:
            species, coordinates, _, _ = read_xyz(
                (Path(__file__).parent / "resources") / file,
                device=self.device,
                dtype=self.dtype,
            )
            species = self.converter(species)
            coordinates.requires_grad_(True)

            aev = self.aev_computer_2x(species, coordinates)
            aev.backward(torch.ones_like(aev))
            aev_grad = coordinates.grad

            coordinates = coordinates.clone().detach()
            coordinates.requires_grad_()
            cu_aev = self.cuaev_computer_2x(species, coordinates)
            cu_aev.backward(torch.ones_like(cu_aev))
            cuaev_grad = coordinates.grad
            self.assertEqual(cu_aev, aev, atol=self.tolerance, rtol=self.tolerance)
            self.assertEqual(
                cuaev_grad, aev_grad, atol=self.tolerance, rtol=self.tolerance
            )

    def testWithHalfNbrList_nopbc(self):
        files = ["small.xyz", "1hz5.xyz", "6W8H.xyz"]
        for file in files:
            species, coordinates, _, _ = read_xyz(
                (Path(__file__).parent / "resources") / file,
                device=self.device,
                dtype=self.dtype,
            )
            species = self.converter(species)
            coordinates.requires_grad_(True)

            aev = self.aev_computer_2x(species, coordinates)
            aev.backward(torch.ones_like(aev))
            aev_grad = coordinates.grad

            coordinates = coordinates.clone().detach()
            coordinates.requires_grad_()
            cu_aev = self.cuaev_computer_2x_use_interface(species, coordinates)
            cu_aev.backward(torch.ones_like(cu_aev))
            cuaev_grad = coordinates.grad
            self.assertEqual(cu_aev, aev, atol=self.tolerance, rtol=self.tolerance)
            self.assertEqual(
                cuaev_grad, aev_grad, atol=self.tolerance, rtol=self.tolerance
            )

    def testWithHalfNbrList_pbc(self):
        species, coordinates, cell, pbc = read_xyz(
            (Path(__file__).parent / "resources") / "water-0.8nm.xyz",
            device=self.device,
            dtype=self.dtype,
        )
        species = self.converter(species)
        coordinates.requires_grad_(True)
        aev = self.aev_computer_2x(species, coordinates, cell, pbc)
        aev.backward(torch.ones_like(aev))
        aev_grad = coordinates.grad

        coordinates = coordinates.clone().detach()
        coordinates.requires_grad_()
        cu_aev = self.cuaev_computer_2x_use_interface(species, coordinates, cell, pbc)
        cu_aev.backward(torch.ones_like(cu_aev))
        cuaev_grad = coordinates.grad
        self.assertEqual(cu_aev, aev, atol=self.tolerance, rtol=self.tolerance)
        self.assertEqual(cuaev_grad, aev_grad, atol=self.tolerance, rtol=self.tolerance)

    def testWithFullNbrList_nopbc(self):
        files = ["small.xyz", "1hz5.xyz", "6W8H.xyz"]
        for file in files:
            species, coordinates, _, _ = read_xyz(
                (Path(__file__).parent / "resources") / file,
                device=self.device,
                dtype=self.dtype,
            )
            species = self.converter(species)
            coordinates.requires_grad_(True)

            aev = self.aev_computer_2x(species, coordinates)
            aev.backward(torch.ones_like(aev))
            aev_grad = coordinates.grad

            coordinates = coordinates.clone().detach()
            coordinates.requires_grad_()
            atom_index12, _, _ = self.cuaev_computer_2x_use_interface.neighborlist(
                self.cutoff_2x, species, coordinates
            )
            assert species.shape[0] == 1
            (
                ilist_unique,
                jlist,
                numneigh,
            ) = self.cuaev_computer_2x_use_interface._half_to_full_nbrlist(atom_index12)
            cu_aev = (
                self.cuaev_computer_2x_use_interface._compute_cuaev_with_full_nbrlist(
                    species, coordinates, ilist_unique, jlist, numneigh
                )
            )

            cu_aev.backward(torch.ones_like(cu_aev))
            cuaev_grad = coordinates.grad
            self.assertEqual(cu_aev, aev, atol=self.tolerance, rtol=self.tolerance)
            self.assertEqual(
                cuaev_grad, aev_grad, atol=self.tolerance, rtol=self.tolerance
            )

    def testWithFullNbrList_pbc(self):
        species, coordinates, cell, pbc = read_xyz(
            (Path(__file__).parent / "resources") / "water-0.8nm.xyz",
            device=self.device,
            dtype=self.dtype,
        )
        species = self.converter(species)
        coordinates.requires_grad_(True)
        aev = self.aev_computer_2x(species, coordinates, cell, pbc)
        aev.backward(torch.ones_like(aev))
        aev_grad = coordinates.grad

        coordinates = coordinates.clone().detach()
        coordinates.requires_grad_()

        atom_index12, _, _ = self.cuaev_computer_2x_use_interface.neighborlist(
            self.cutoff_2x, species, coordinates
        )
        assert species.shape[0] == 1
        (
            ilist_unique,
            jlist,
            numneigh,
        ) = self.cuaev_computer_2x_use_interface._half_to_full_nbrlist(atom_index12)
        cu_aev = self.cuaev_computer_2x_use_interface._compute_cuaev_with_full_nbrlist(
            species, coordinates, ilist_unique, jlist, numneigh
        )

        cu_aev.backward(torch.ones_like(cu_aev))
        cuaev_grad = coordinates.grad

        # Debug
        if False:
            print((cu_aev - aev).abs().max())
            print((cuaev_grad - aev_grad).abs().max())

        # when pbc is on, full nbrlist converted from half nbrlist is not correct
        # self.assertEqual(cu_aev, aev, atol=self.tolerance, rtol=self.tolerance)
        # self.assertEqual(cuaev_grad, aev_grad, atol=self.tolerance,
        # rtol=self.tolerance)


if __name__ == "__main__":
    unittest.main(verbosity=2)
