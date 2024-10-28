import unittest
import os
import pickle

import torch
from torch.autograd import gradcheck, gradgradcheck

import torchani
from torchani._testing import ANITestCase, expand

path = os.path.dirname(os.path.realpath(__file__))


@expand()
class TestTorchNumericalCheck(ANITestCase):
    # torch.autograd.gradcheck (and torch.autograd.gradgradcheck) verify that
    # the numerical and analytical gradient (and hessian respectively) of a
    # function match within a given tolerance.
    #
    # The forward call of the function is wrapped with a lambda so that
    # gradcheck gets a function with only one tensor input and tensor output.
    #
    # nondet_tol is necessarily greater than zero since some operations are
    # nondeterministic which makes two equal inputs have different outputs
    def setUp(self):
        self.model = self._setup(
            torchani.models.ANI1x(model_index=0, periodic_table_index=False).double()
        )
        datafile = os.path.join(path, "resources/NIST/all")
        # Some small molecules are selected to make the tests faster
        with open(datafile, mode="rb") as fb:
            data = pickle.load(fb)[1243:1250]
        self.data = data

    def testAutograd(self):
        for coordinates, species, _, _, _, _ in self.data:
            coordinates = torch.tensor(
                coordinates, device=self.device, dtype=torch.double, requires_grad=True
            )
            species = torch.tensor(species, device=self.device, dtype=torch.long)

            gradcheck(
                lambda x: self.model((species, x)).energies,
                coordinates,
                nondet_tol=1e-13,
            )

    def testDoubleAutograd(self):
        for coordinates, species, _, _, _, _ in self.data:
            coordinates = torch.tensor(
                coordinates, device=self.device, dtype=torch.double, requires_grad=True
            )
            species = torch.tensor(species, device=self.device, dtype=torch.long)
            gradgradcheck(
                lambda x: self.model((species, x)).energies,
                coordinates,
                nondet_tol=1e-13,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
