import os
import torch
import torchani
import unittest
import pickle
from ase.io import read
from torchani.testing import TestCase, make_tensor
from parameterized import parameterized_class

path = os.path.dirname(os.path.realpath(__file__))

skipIfNoGPU = unittest.skipIf(not torch.cuda.is_available(), 'There is no device to run this test')
skipIfNoMultiGPU = unittest.skipIf(not torch.cuda.device_count() >= 2, 'There is not enough GPU devices to run this test')
skipIfNoCUAEV = unittest.skipIf(not torchani.aev.cuaev_is_installed, "only valid when cuaev is installed")


@skipIfNoCUAEV
class TestCUAEVNoGPU(TestCase):

    def testSimple(self):
        def f(coordinates, species, Rcr: float, Rca: float, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species: int):
            cuaev_computer = torch.classes.cuaev.CuaevComputer(Rcr, Rca, EtaR.flatten(), ShfR.flatten(), EtaA.flatten(), Zeta.flatten(), ShfA.flatten(), ShfZ.flatten(), num_species, True)
            return torch.ops.cuaev.run(coordinates, species, cuaev_computer)
        s = torch.jit.script(f)
        self.assertIn("cuaev::run", str(s.graph))

    def testAEVComputer(self):
        aev_computer = torchani.AEVComputer.like_1x(use_cuda_extension=True)
        s = torch.jit.script(aev_computer)
        # Computation of AEV using cuaev when there is no atoms does not require CUDA, and can be run without GPU
        species = make_tensor((8, 0), 'cpu', torch.int64, low=-1, high=4)
        coordinates = make_tensor((8, 0, 3), 'cpu', torch.float32, low=-5, high=5)
        self.assertIn("cuaev::run", str(s.graph_for((species, coordinates))))

    def testPickle(self):
        aev_computer = torchani.AEVComputer.like_1x(use_cuda_extension=True)
        tmpfile = '/tmp/cuaev.pkl'
        with open(tmpfile, 'wb') as file:
            pickle.dump(aev_computer, file)
        with open(tmpfile, 'rb') as file:
            aev_computer = pickle.load(file)
        os.remove(tmpfile)


@skipIfNoGPU
@skipIfNoCUAEV
@parameterized_class(('cutoff_fn'), [['cosine'], ['smooth']])
class TestCUAEV(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ani2x = torchani.models.ANI2x(periodic_table_index=True, model_index=None)

    def setUp(self, device='cuda:0'):
        self.tolerance = 5e-5
        self.device = device
        self.aev_computer_1x = torchani.AEVComputer.like_1x(cutoff_fn=self.cutoff_fn).to(self.device)
        self.cuaev_computer_1x = torchani.AEVComputer.like_1x(cutoff_fn=self.cutoff_fn, use_cuda_extension=True).to(self.device)
        self.nn = torch.nn.Sequential(torch.nn.Linear(384, 1, False)).to(self.device)

        self.aev_computer_2x = torchani.AEVComputer.like_2x(cutoff_fn=self.cutoff_fn).to(self.device)
        self.cuaev_computer_2x = torchani.AEVComputer.like_2x(cutoff_fn=self.cutoff_fn, use_cuda_extension=True).to(self.device)
        self.ani2x = self.__class__.ani2x

    def _skip_if_not_cosine(self):
        if self.cutoff_fn != "cosine":
            self.skipTest("Skip slow tests for non-cosine cutoff")

    def _double_backward_1_test(self, species, coordinates):

        def double_backward(aev_computer, species, coordinates):
            torch.manual_seed(12345)
            self.nn.zero_grad()
            _, aev = aev_computer((species, coordinates))
            E = self.nn(aev).sum()
            force = -torch.autograd.grad(E, coordinates, create_graph=True, retain_graph=True)[0]
            force_true = torch.randn_like(force)
            loss = torch.abs(force_true - force).sum(dim=(1, 2)).mean()
            loss.backward()
            param = next(self.nn.parameters())
            return aev, force, param.grad

        aev, force_ref, param_grad_ref = double_backward(self.aev_computer_1x, species, coordinates)
        cu_aev, force_cuaev, param_grad = double_backward(self.cuaev_computer_1x, species, coordinates)

        self.assertEqual(cu_aev, aev, f'cu_aev: {cu_aev}\n aev: {aev}')
        self.assertEqual(force_cuaev, force_ref, f'\nforce_cuaev: {force_cuaev}\n force_ref: {force_ref}')
        self.assertEqual(param_grad, param_grad_ref, f'\nparam_grad: {param_grad}\n param_grad_ref: {param_grad_ref}', atol=5e-5, rtol=5e-5)

    def _double_backward_2_test(self, species, coordinates):

        def double_backward(aev_computer, species, coordinates):
            """
            # We want to get the gradient of `grad_aev`, which requires `grad_aev` to be a leaf node
            # due to `torch.autograd`'s limitation. So we split the coord->aev->energy graph into two separate
            # graphs: coord->aev and aev->energy, so that aev and grad_aev are now leaves.
            """
            torch.manual_seed(12345)
            # graph1 input -> aev
            coordinates = coordinates.clone().detach().requires_grad_()
            _, aev = aev_computer((species, coordinates))
            # graph2 aev -> E
            aev_ = aev.clone().detach().requires_grad_()
            E = self.nn(aev_).sum()
            # graph2 backward
            aev_grad = torch.autograd.grad(E, aev_, create_graph=True, retain_graph=True)[0]
            # graph1 backward
            aev_grad_ = aev_grad.clone().detach().requires_grad_()
            force = torch.autograd.grad(aev, coordinates, aev_grad_, create_graph=True, retain_graph=True)[0]
            # force loss backward
            force_true = torch.randn_like(force)
            loss = torch.abs(force_true - force).sum(dim=(1, 2)).mean()
            aev_grad_grad = torch.autograd.grad(loss, aev_grad_, create_graph=True, retain_graph=True)[0]

            return aev, force, aev_grad_grad

        aev, force_ref, aev_grad_grad = double_backward(self.aev_computer_1x, species, coordinates)
        cu_aev, force_cuaev, cuaev_grad_grad = double_backward(self.cuaev_computer_1x, species, coordinates)

        self.assertEqual(cu_aev, aev, f'cu_aev: {cu_aev}\n aev: {aev}', atol=5e-5, rtol=5e-5)
        self.assertEqual(force_cuaev, force_ref, f'\nforce_cuaev: {force_cuaev}\n force_ref: {force_ref}', atol=5e-5, rtol=5e-5)
        self.assertEqual(cuaev_grad_grad, aev_grad_grad, f'\ncuaev_grad_grad: {cuaev_grad_grad}\n aev_grad_grad: {aev_grad_grad}', atol=5e-5, rtol=5e-5)

    def testSimple(self):
        coordinates = torch.tensor([
            [[0.03192167, 0.00638559, 0.01301679],
             [-0.83140486, 0.39370209, -0.26395324],
             [-0.66518241, -0.84461308, 0.20759389],
             [0.45554739, 0.54289633, 0.81170881],
             [0.66091919, -0.16799635, -0.91037834]],
            [[-4.1862600, 0.0575700, -0.0381200],
             [-3.1689400, 0.0523700, 0.0200000],
             [-4.4978600, 0.8211300, 0.5604100],
             [-4.4978700, -0.8000100, 0.4155600],
             [0.00000000, -0.00000000, -0.00000000]]
        ], device=self.device)
        species = torch.tensor([[1, 0, 0, 0, 0], [2, 0, 0, 0, -1]], device=self.device)

        _, aev = self.aev_computer_1x((species, coordinates))
        _, cu_aev = self.cuaev_computer_1x((species, coordinates))
        self.assertEqual(cu_aev, aev)

    @skipIfNoMultiGPU
    def testMultiGPU(self):
        self.setUp(device='cuda:1')
        self.testSimple()
        self.testSimpleBackward()
        self.testSimpleDoubleBackward_1()
        self.testSimpleDoubleBackward_2()
        self.setUp(device='cuda:0')

    def testBatch(self):
        coordinates = torch.rand([100, 50, 3], device=self.device) * 5
        species = torch.randint(-1, 3, (100, 50), device=self.device)

        _, aev = self.aev_computer_1x((species, coordinates))
        _, cu_aev = self.cuaev_computer_1x((species, coordinates))
        self.assertEqual(cu_aev, aev)

    def testPickleCorrectness(self):
        ref_aev_computer = self.cuaev_computer_1x
        tmpfile = '/tmp/cuaev.pkl'
        with open(tmpfile, 'wb') as file:
            pickle.dump(ref_aev_computer, file)
        with open(tmpfile, 'rb') as file:
            test_aev_computer = pickle.load(file)
        os.remove(tmpfile)

        coordinates = torch.rand([2, 50, 3], device=self.device) * 5
        species = torch.randint(-1, 3, (2, 50), device=self.device)
        _, ref_aev = ref_aev_computer((species, coordinates))
        _, test_aev = test_aev_computer((species, coordinates))
        self.assertEqual(ref_aev, test_aev)

    def testSimpleBackward(self):
        coordinates = torch.tensor([
            [[0.03192167, 0.00638559, 0.01301679],
             [-0.83140486, 0.39370209, -0.26395324],
             [-0.66518241, -0.84461308, 0.20759389],
             [0.45554739, 0.54289633, 0.81170881],
             [0.66091919, -0.16799635, -0.91037834]],
            [[-4.1862600, 0.0575700, -0.0381200],
             [-3.1689400, 0.0523700, 0.0200000],
             [-4.4978600, 0.8211300, 0.5604100],
             [-4.4978700, -0.8000100, 0.4155600],
             [0.00000000, -0.00000000, -0.00000000]]
        ], requires_grad=True, device=self.device)
        species = torch.tensor([[1, 0, 0, 0, 0], [2, 0, 0, 0, -1]], device=self.device)

        _, aev = self.aev_computer_1x((species, coordinates))
        aev.backward(torch.ones_like(aev))
        aev_grad = coordinates.grad

        coordinates = coordinates.clone().detach()
        coordinates.requires_grad_()
        _, cu_aev = self.cuaev_computer_1x((species, coordinates))
        cu_aev.backward(torch.ones_like(cu_aev))
        cuaev_grad = coordinates.grad
        self.assertEqual(cu_aev, aev, f'cu_aev: {cu_aev}\n aev: {aev}')
        self.assertEqual(cuaev_grad, aev_grad, f'\ncuaev_grad: {cuaev_grad}\n aev_grad: {aev_grad}', atol=5e-5, rtol=5e-5)

    def testSimpleDoubleBackward_1(self):
        """
        Test Double Backward (Force training) by parameters' gradient
        """
        coordinates = torch.tensor([
            [[0.03192167, 0.00638559, 0.01301679],
             [-0.83140486, 0.39370209, -0.26395324],
             [-0.66518241, -0.84461308, 0.20759389],
             [0.45554739, 0.54289633, 0.81170881],
             [0.66091919, -0.16799635, -0.91037834]],
            [[-4.1862600, 0.0575700, -0.0381200],
             [-3.1689400, 0.0523700, 0.0200000],
             [-4.4978600, 0.8211300, 0.5604100],
             [-4.4978700, -0.8000100, 0.4155600],
             [0.00000000, -0.00000000, -0.00000000]]
        ], requires_grad=True, device=self.device)
        species = torch.tensor([[1, 0, 0, 0, 0], [2, 0, 0, 0, -1]], device=self.device)

        self._double_backward_1_test(species, coordinates)

    def testSimpleDoubleBackward_2(self):
        """
        Test Double Backward (Force training) directly.
        Double backward:
        Forward: input is dE/dAEV, output is force
        Backward: input is dLoss/dForce, output is dLoss/(dE/dAEV)
        """
        coordinates = torch.tensor([
            [[0.03192167, 0.00638559, 0.01301679],
             [-0.83140486, 0.39370209, -0.26395324],
             [-0.66518241, -0.84461308, 0.20759389],
             [0.45554739, 0.54289633, 0.81170881],
             [0.66091919, -0.16799635, -0.91037834]],
            [[-4.1862600, 0.0575700, -0.0381200],
             [-3.1689400, 0.0523700, 0.0200000],
             [-4.4978600, 0.8211300, 0.5604100],
             [-4.4978700, -0.8000100, 0.4155600],
             [0.00000000, -0.00000000, -0.00000000]]
        ], requires_grad=True, device=self.device)
        species = torch.tensor([[1, 0, 0, 0, 0], [2, 0, 0, 0, -1]], device=self.device)

        self._double_backward_2_test(species, coordinates)

    def testTripeptideMD(self):
        for i in range(100):
            datafile = os.path.join(path, 'test_data/tripeptide-md/{}.dat'.format(i))
            with open(datafile, 'rb') as f:
                coordinates, species, *_ = pickle.load(f)
                coordinates = torch.from_numpy(coordinates).float().unsqueeze(0).to(self.device)
                species = torch.from_numpy(species).unsqueeze(0).to(self.device)
                _, aev = self.aev_computer_1x((species, coordinates))
                _, cu_aev = self.cuaev_computer_1x((species, coordinates))
                self.assertEqual(cu_aev, aev)

    def testTripeptideMDBackward(self):
        for i in range(100):
            datafile = os.path.join(path, 'test_data/tripeptide-md/{}.dat'.format(i))
            with open(datafile, 'rb') as f:
                coordinates, species, *_ = pickle.load(f)
                coordinates = torch.from_numpy(coordinates).float().unsqueeze(0).to(self.device).requires_grad_(True)
                species = torch.from_numpy(species).unsqueeze(0).to(self.device)
                _, aev = self.aev_computer_1x((species, coordinates))
                aev.backward(torch.ones_like(aev))
                aev_grad = coordinates.grad

                coordinates = coordinates.clone().detach()
                coordinates.requires_grad_()
                _, cu_aev = self.cuaev_computer_1x((species, coordinates))
                cu_aev.backward(torch.ones_like(cu_aev))
                cuaev_grad = coordinates.grad
                self.assertEqual(cu_aev, aev)
                self.assertEqual(cuaev_grad, aev_grad, atol=5e-5, rtol=5e-5)

    def testTripeptideMDDoubleBackward_2(self):
        # skip if not cosine
        self._skip_if_not_cosine()

        for i in range(100):
            datafile = os.path.join(path, 'test_data/tripeptide-md/{}.dat'.format(i))
            with open(datafile, 'rb') as f:
                coordinates, species, *_ = pickle.load(f)
                coordinates = torch.from_numpy(coordinates).float().unsqueeze(0).to(self.device).requires_grad_(True)
                species = torch.from_numpy(species).unsqueeze(0).to(self.device)
                self._double_backward_2_test(species, coordinates)

    def testNIST(self):
        # skip if not cosine
        self._skip_if_not_cosine()

        datafile = os.path.join(path, 'test_data/NIST/all')
        with open(datafile, 'rb') as f:
            data = pickle.load(f)
            for coordinates, species, _, _, _, _ in data:
                coordinates = torch.from_numpy(coordinates).to(torch.float).to(self.device)
                species = torch.from_numpy(species).to(self.device)
                _, aev = self.aev_computer_1x((species, coordinates))
                _, cu_aev = self.cuaev_computer_1x((species, coordinates))
                self.assertEqual(cu_aev, aev)

    def testNISTBackward(self):
        datafile = os.path.join(path, 'test_data/NIST/all')
        with open(datafile, 'rb') as f:
            data = pickle.load(f)
            for coordinates, species, _, _, _, _ in data[:10]:
                coordinates = torch.from_numpy(coordinates).to(torch.float).to(self.device).requires_grad_(True)
                species = torch.from_numpy(species).to(self.device)
                _, aev = self.aev_computer_1x((species, coordinates))
                aev.backward(torch.ones_like(aev))
                aev_grad = coordinates.grad

                coordinates = coordinates.clone().detach()
                coordinates.requires_grad_()
                _, cu_aev = self.cuaev_computer_1x((species, coordinates))
                cu_aev.backward(torch.ones_like(cu_aev))
                cuaev_grad = coordinates.grad
                self.assertEqual(cu_aev, aev)
                self.assertEqual(cuaev_grad, aev_grad, atol=5e-5, rtol=5e-5)

    def testNISTDoubleBackward_2(self):
        datafile = os.path.join(path, 'test_data/NIST/all')
        with open(datafile, 'rb') as f:
            data = pickle.load(f)
            for coordinates, species, _, _, _, _ in data[:3]:
                coordinates = torch.from_numpy(coordinates).to(torch.float).to(self.device).requires_grad_(True)
                species = torch.from_numpy(species).to(self.device)
                self._double_backward_2_test(species, coordinates)

    def testVeryDenseMolecule(self):
        """
        Test very dense molecule for aev correctness, especially for angular kernel when center atom pairs are more than 32.
        issue: https://github.com/aiqm/torchani/pull/555
        """
        for i in range(5):
            datafile = os.path.join(path, 'test_data/tripeptide-md/{}.dat'.format(i))
            with open(datafile, 'rb') as f:
                coordinates, species, *_ = pickle.load(f)
                # change angstrom coordinates to 10 times smaller
                coordinates = 0.1 * torch.from_numpy(coordinates).float().unsqueeze(0).to(self.device)
                species = torch.from_numpy(species).unsqueeze(0).to(self.device)
                _, aev = self.aev_computer_1x((species, coordinates))
                _, cu_aev = self.cuaev_computer_1x((species, coordinates))
                self.assertEqual(cu_aev, aev, atol=5e-5, rtol=5e-5)

    def testVeryDenseMoleculeBackward(self):
        for i in range(5):
            datafile = os.path.join(path, 'test_data/tripeptide-md/{}.dat'.format(i))
            with open(datafile, 'rb') as f:
                coordinates, species, *_ = pickle.load(f)
                # change angstrom coordinates to 10 times smaller
                coordinates = 0.1 * torch.from_numpy(coordinates).float().unsqueeze(0).to(self.device)
                coordinates.requires_grad_(True)
                species = torch.from_numpy(species).unsqueeze(0).to(self.device)

                _, aev = self.aev_computer_1x((species, coordinates))
                aev.backward(torch.ones_like(aev))
                aev_grad = coordinates.grad

                coordinates = coordinates.clone().detach()
                coordinates.requires_grad_()
                _, cu_aev = self.cuaev_computer_1x((species, coordinates))
                cu_aev.backward(torch.ones_like(cu_aev))
                cuaev_grad = coordinates.grad
                self.assertEqual(cu_aev, aev, atol=5e-5, rtol=5e-5)
                self.assertEqual(cuaev_grad, aev_grad, atol=5e-4, rtol=5e-4)

    def testPDB(self):
        files = ['small.pdb', '1hz5.pdb', '6W8H.pdb']
        for file in files:
            filepath = os.path.join(path, f'../dataset/pdb/{file}')
            mol = read(filepath)
            species = torch.tensor([mol.get_atomic_numbers()], device=self.device)
            positions = torch.tensor([mol.get_positions()], dtype=torch.float32, requires_grad=False, device=self.device)
            speciesPositions = self.ani2x.species_converter((species, positions))
            species, coordinates = speciesPositions

            _, aev = self.aev_computer_2x((species, coordinates))
            _, cu_aev = self.cuaev_computer_2x((species, coordinates))
            self.assertEqual(cu_aev, aev)

    def testPDBBackward(self):
        files = ['small.pdb', '1hz5.pdb', '6W8H.pdb']
        for file in files:
            filepath = os.path.join(path, f'../dataset/pdb/{file}')
            mol = read(filepath)
            species = torch.tensor([mol.get_atomic_numbers()], device=self.device)
            positions = torch.tensor([mol.get_positions()], dtype=torch.float32, requires_grad=False, device=self.device)
            speciesPositions = self.ani2x.species_converter((species, positions))
            species, coordinates = speciesPositions
            coordinates.requires_grad_(True)

            _, aev = self.aev_computer_2x((species, coordinates))
            aev.backward(torch.ones_like(aev))
            aev_grad = coordinates.grad

            coordinates = coordinates.clone().detach()
            coordinates.requires_grad_()
            _, cu_aev = self.cuaev_computer_2x((species, coordinates))
            cu_aev.backward(torch.ones_like(cu_aev))
            cuaev_grad = coordinates.grad
            self.assertEqual(cu_aev, aev, atol=5e-5, rtol=5e-5)
            self.assertEqual(cuaev_grad, aev_grad, atol=5e-4, rtol=5e-4)


if __name__ == '__main__':
    unittest.main()
