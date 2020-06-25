import torch
import torchani
import unittest
import os
import pickle

path = os.path.dirname(os.path.realpath(__file__))


class _TestGrad(unittest.TestCase):

    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = torchani.models.ANI1x(model_index=0).to(device=self.device, dtype=torch.double)
        datafile = os.path.join(path, 'test_data/NIST/all')

        # Some small molecules are selected to make the tests faster
        self.data = pickle.load(open(datafile, 'rb'))[1243:1250]

    def testGrad(self):
        for coordinates, species, _, _, _, _ in self.data:

            coordinates = torch.from_numpy(coordinates).to(device=self.device, dtype=torch.float64)
            coordinates.requires_grad_(True)

            species = torch.from_numpy(species).to(self.device)

            # forward call is wrapped with a lambda so that gradcheck gets a
            # function with only one tensor input and tensor output.

            # nondet_tol is necessary since some operations are
            # nondeterministic which makes two equal inputs have different
            # outputs
            torch.autograd.gradcheck(lambda x: self.model((species, x)).energies, coordinates, nondet_tol=1e-13)

    def testGradGrad(self):
        for coordinates, species, _, _, _, _ in self.data:

            coordinates = torch.from_numpy(coordinates).to(device=self.device, dtype=torch.float64)
            coordinates.requires_grad_(True)

            species = torch.from_numpy(species).to(self.device)

            # forward call is wrapped with a lambda so that gradcheck gets a
            # function with only one tensor input and tensor output.

            # nondet_tol is necessary since some operations are
            # nondeterministic which makes two equal inputs have different
            # outputs
            torch.autograd.gradgradcheck(lambda x: self.model((species, x)).energies, coordinates, nondet_tol=1e-13)
