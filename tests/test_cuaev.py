import os
import torch
import torchani
import unittest
import pickle
from torchani.testing import TestCase, make_tensor


path = os.path.dirname(os.path.realpath(__file__))

skipIfNoGPU = unittest.skipIf(not torch.cuda.is_available(),
                              'There is no device to run this test')
skipIfNoCUAEV = unittest.skipIf(not torchani.aev.has_cuaev, "only valid when cuaev is installed")


@skipIfNoCUAEV
class TestCUAEVNoGPU(TestCase):

    def testSimple(self):
        def f(coordinates, species, Rcr: float, Rca: float, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species: int):
            return torch.ops.cuaev.cuComputeAEV(coordinates, species, Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species)
        s = torch.jit.script(f)
        self.assertIn("cuaev::cuComputeAEV", str(s.graph))

    def testAEVComputer(self):
        path = os.path.dirname(os.path.realpath(__file__))
        const_file = os.path.join(path, '../torchani/resources/ani-1x_8x/rHCNO-5.2R_16-3.5A_a4-8.params')  # noqa: E501
        consts = torchani.neurochem.Constants(const_file)
        aev_computer = torchani.AEVComputer(**consts, use_cuda_extension=True)
        s = torch.jit.script(aev_computer)
        # Computation of AEV using cuaev when there is no atoms does not require CUDA, and can be run without GPU
        species = make_tensor((8, 0), 'cpu', torch.int64, low=-1, high=4)
        coordinates = make_tensor((8, 0, 3), 'cpu', torch.float32, low=-5, high=5)
        self.assertIn("cuaev::cuComputeAEV", str(s.graph_for((species, coordinates))))


@skipIfNoGPU
@skipIfNoCUAEV
class TestCUAEV(TestCase):

    def setUp(self):
        self.tolerance = 5e-5
        self.device = 'cuda'
        Rcr = 5.2000e+00
        Rca = 3.5000e+00
        EtaR = torch.tensor([1.6000000e+01], device=self.device)
        ShfR = torch.tensor([9.0000000e-01, 1.1687500e+00, 1.4375000e+00, 1.7062500e+00, 1.9750000e+00, 2.2437500e+00, 2.5125000e+00, 2.7812500e+00, 3.0500000e+00, 3.3187500e+00, 3.5875000e+00, 3.8562500e+00, 4.1250000e+00, 4.3937500e+00, 4.6625000e+00, 4.9312500e+00], device=self.device)
        Zeta = torch.tensor([3.2000000e+01], device=self.device)
        ShfZ = torch.tensor([1.9634954e-01, 5.8904862e-01, 9.8174770e-01, 1.3744468e+00, 1.7671459e+00, 2.1598449e+00, 2.5525440e+00, 2.9452431e+00], device=self.device)
        EtaA = torch.tensor([8.0000000e+00], device=self.device)
        ShfA = torch.tensor([9.0000000e-01, 1.5500000e+00, 2.2000000e+00, 2.8500000e+00], device=self.device)
        num_species = 4
        self.aev_computer = torchani.AEVComputer(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species)
        self.cuaev_computer = torchani.AEVComputer(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species, use_cuda_extension=True)

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
        ], requires_grad=True, device=self.device)
        species = torch.tensor([[1, 0, 0, 0, 0], [2, 0, 0, 0, -1]], device=self.device)

        _, aev = self.aev_computer((species, coordinates))
        _, cu_aev = self.cuaev_computer((species, coordinates))
        self.assertEqual(cu_aev, aev)

    def testTripeptideMD(self):
        for i in range(100):
            datafile = os.path.join(path, 'test_data/tripeptide-md/{}.dat'.format(i))
            with open(datafile, 'rb') as f:
                coordinates, species, _, _, _, _, _, _ = pickle.load(f)
                coordinates = torch.from_numpy(coordinates).float().unsqueeze(0).to(self.device)
                species = torch.from_numpy(species).unsqueeze(0).to(self.device)
                _, aev = self.aev_computer((species, coordinates))
                _, cu_aev = self.cuaev_computer((species, coordinates))
                self.assertEqual(cu_aev, aev)

    def testNIST(self):
        datafile = os.path.join(path, 'test_data/NIST/all')
        with open(datafile, 'rb') as f:
            data = pickle.load(f)
            for coordinates, species, _, _, _, _ in data:
                coordinates = torch.from_numpy(coordinates).to(torch.float).to(self.device)
                species = torch.from_numpy(species).to(self.device)
                _, aev = self.aev_computer((species, coordinates))
                _, cu_aev = self.cuaev_computer((species, coordinates))
                self.assertEqual(cu_aev, aev)

    def testVeryDenseMolecule(self):
        for i in range(100):
            datafile = os.path.join(path, 'test_data/tripeptide-md/{}.dat'.format(i))
            with open(datafile, 'rb') as f:
                coordinates, species, _, _, _, _, _, _ = pickle.load(f)
                # change angstrom coordinates to 10 times smaller
                coordinates = 0.1 * torch.from_numpy(coordinates).float().unsqueeze(0).to(self.device)
                species = torch.from_numpy(species).unsqueeze(0).to(self.device)
                _, aev = self.aev_computer((species, coordinates))
                _, cu_aev = self.cuaev_computer((species, coordinates))
                self.assertEqual(cu_aev, aev, atol=5e-5, rtol=5e-5)


if __name__ == '__main__':
    unittest.main()
