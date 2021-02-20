import os
import torch
import torchani
import unittest
from torchani.testing import TestCase
import torchsnooper
import snoop

# verbose = True
verbose = False
torchsnooper.register_snoop(verbose)

path = os.path.dirname(os.path.realpath(__file__))
skip = unittest.skipIf(True, "Skip")


class TestBackward(TestCase):

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
        self.aev_computer = torchani.gaev.AEVComputer(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species)
        self.coordinates = torch.tensor([
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
        self.species = torch.tensor([[1, 0, 0, 0, 0], [2, 0, 0, -1, -1]], device=self.device)

    # @snoop()
    # @skip
    def testCutoff(self):
        distances = torch.rand([16, 1, 1]) * 10
        distances.requires_grad_()
        cutoff = 5.2

        res1 = torchani.gaev.cutoff_cosine(distances, cutoff)
        res1.backward(torch.ones_like(res1))
        grad1 = distances.grad

        distances.grad.data.zero_()
        op = torchani.gaev.Cutoff_cosine.apply
        res2 = op(distances, cutoff)
        res2.backward(torch.ones_like(res2))
        grad2 = distances.grad
        self.assertEqual(grad1, grad2, f'grad1: {grad1}\n grad2: {grad2}')

        op = torchani.gaev.MyCutoff_cosine()
        res3 = op.forward(distances, cutoff)
        grad3, _ = op.backward(torch.ones_like(res3))
        self.assertEqual(grad1, grad3, f'grad1: {grad1}\n grad3: {grad3}')

    # @snoop()
    # @skip
    def testRadial_terms(self):
        torchani.gaev.return_radial_input = True
        torchani.gaev.return_angular_input = False
        torchani.gaev.return_computeaev_input = False
        Rcr, EtaR, ShfR, distances = self.aev_computer((self.species, self.coordinates))

        distances = distances.clone().detach()
        distances.requires_grad_()
        res1 = torchani.gaev.radial_terms(Rcr, EtaR, ShfR, distances)
        res1.backward(torch.ones_like(res1))
        grad1 = distances.grad

        op = torchani.gaev.MyRadial_terms()
        res3 = op.forward(Rcr, EtaR, ShfR, distances)
        _, _, _, grad3 = op.backward(torch.ones_like(res3))
        self.assertEqual(grad1, grad3, f'grad1: {grad1}\n grad3: {grad3}')

    # @snoop()
    # @skip
    def testAngular_terms(self):
        torchani.gaev.return_radial_input = False
        torchani.gaev.return_angular_input = True
        torchani.gaev.return_computeaev_input = False
        Rca, ShfZ, EtaA, Zeta, ShfA, vec12 = self.aev_computer((self.species, self.coordinates))

        vec12 = vec12.clone().detach()
        vec12.requires_grad_()
        res1 = torchani.gaev.angular_terms(Rca, ShfZ, EtaA, Zeta, ShfA, vec12)
        res1.backward(torch.ones_like(res1))
        grad1 = vec12.grad

        op = torchani.gaev.MyAngular_terms()
        res3 = op.forward(Rca, ShfZ, EtaA, Zeta, ShfA, vec12)
        _, _, _, _, _, grad3 = op.backward(torch.ones_like(res3))
        self.assertEqual(res1, res3, f'res1: {res1}\n res3: {res3}')
        self.assertEqual(grad1, grad3, f'grad1: {grad1}\n grad3: {grad3}')

    # @snoop()
    # @skip
    def testCompute_aev(self):
        torchani.gaev.return_radial_input = False
        torchani.gaev.return_angular_input = False
        torchani.gaev.return_computeaev_input = True
        species, coordinates, triu_index, constants, sizes, cell_shifts = self.aev_computer((self.species, self.coordinates))

        coordinates = coordinates.clone().detach()
        coordinates.requires_grad_()
        res1 = torchani.gaev.compute_aev(species, coordinates, triu_index, constants, sizes, cell_shifts)
        res1.backward(torch.ones_like(res1))
        grad1 = coordinates.grad

        op = torchani.gaev.MyCompute_aev()
        res3 = op.forward(species, coordinates, triu_index, constants, sizes, cell_shifts)
        _, grad3, _, _, _, _ = op.backward(torch.ones_like(res3))
        self.assertEqual(res1, res3, f'res1: {res1}\n res3: {res3}')
        self.assertEqual(grad1, grad3, f'\ngrad1: {grad1}\n grad3: {grad3}')

    @skip
    def testSimple(self):
        torchani.gaev.return_radial_input = False
        torchani.gaev.return_angular_input = False
        torchani.gaev.return_computeaev_input = False
        aev = self.aev_computer((self.species, self.coordinates))
        # self.assertEqual(cu_aev, aev)


if __name__ == '__main__':
    unittest.main()
