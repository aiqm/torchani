"""Modular AEV usage Example"""
import torch
import torchani
import math
from torch import Tensor
from torchani.compat import Final

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# this example is meant to show how to take advantage of the modular AEV implementation
# We will use these coordinates and species:
coordinates = torch.tensor([[[0.03192167, 0.00638559, 0.01301679],
                             [-0.83140486, 0.39370209, -0.26395324],
                             [-0.66518241, -0.84461308, 0.20759389],
                             [0.45554739, 0.54289633, 0.81170881],
                             [0.66091919, -0.16799635, -0.91037834]]],
                           device=device)
species = torch.tensor([[1, 0, 0, 0, 0]], device=device)

# suppose that we want to make exactly the same aev computer as ANI 2x:
aev_computer = torchani.AEVComputer.like_2x().to(device)
species, aevs = aev_computer((species, coordinates))
radial_length = aev_computer.radial_length
print('AEV computer like 2x')
print('for first atom, first 10 terms of radial:', aevs[0, 0, :10])
print('for first atom, first 10 terms of angular:', aevs[0, 0, radial_length:radial_length + 10])
print()

# suppose we want to make an AEV computer essentially like the 1x aev computer,
# but using a different cutoff function, such as a smooth cutoff

# WARNING: Be very careful, if a model has not been trained using this cutoff function
# then using this aev computer with it will give nonsensical results

# we can then call:
aev_computer_smooth = torchani.AEVComputer.like_1x(cutoff_fn='smooth').to(device)
radial_length = aev_computer_smooth.radial_length
species, aevs = aev_computer_smooth((species, coordinates))
print('AEV computer like 1x, but with a smooth cutoff')
print('for first atom, first 10 terms of radial:', aevs[0, 0, :10])
print('for first atom, first 10 terms of angular:', aevs[0, 0, radial_length:radial_length + 10])
print()

# Lets say now we want to experiment with a different cutoff function, such as
# a biweight cutoff (WARNING: biweight does not have a continuous
# second derivative at the cutoff radius value, this is done just as an example)

# Since biweight is not coded in Torchani we can code it ourselves and pass it
# to the AEVComputer, as long as the forward method has this form, it will work!

# the same cutoff function will be used for both radial and angular terms


class CutoffBiweight(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, distances: Tensor, cutoff: float) -> Tensor:
        # assuming all elements in distances are smaller than cutoff
        return (cutoff**2 - distances**2)**2 / cutoff**4


cutoff_fn = CutoffBiweight()
aev_computer_bw = torchani.AEVComputer.like_1x(cutoff_fn=cutoff_fn).to(device)
radial_length = aev_computer_bw.radial_length
species, aevs = aev_computer_smooth((species, coordinates))
print('AEV computer like 1x, but with a custom cutoff function')
print('for first atom, first 10 terms of radial:', aevs[0, 0, :10])
print('for first atom, first 10 terms of angular:', aevs[0, 0, radial_length:radial_length + 10])
print()


# Now lets try something a bit more complicated. I want to experiment with
# different angular terms that have a form of exp(-gamma * (cos(theta) -
# cos(theta0))**2) how can I do that? I can pass this function to
# torchani, as long as it exposes the same API as StandardAngular
# (it has to have a *sublength*, a *cutoff*, and a *forward method* with the same signature)

class AngularCosDiff(torch.nn.Module):

    ShfZ: Tensor
    Gamma: Tensor
    ShfA: Tensor
    EtaA: Tensor
    cutoff: Final[float]

    def __init__(self, EtaA, ShfA, Gamma, ShfZ, cutoff, cutoff_fn='cosine'):
        super().__init__()
        self.register_buffer('Gamma', Gamma.view(1, 1, -1))
        self.register_buffer('EtaA', EtaA.view(-1, 1, 1))
        self.register_buffer('ShfA', ShfA.view(1, -1, 1))
        self.register_buffer('ShfZ', ShfZ.view(1, 1, -1))
        self.cutoff_fn = torchani.aev.cutoffs._parse_cutoff_fn(cutoff_fn)
        self.cutoff = cutoff

        assert self.ShfZ.numel() == self.Gamma.numel()

        self.sublength = self.EtaA.numel() * self.ShfA.numel() * self.ShfZ.numel()
        self.cutoff = cutoff

    def forward(self, vectors12: Tensor) -> Tensor:
        vectors12 = vectors12.view(2, -1, 3, 1, 1, 1)
        distances12 = vectors12.norm(2, dim=2)
        cos_angles = vectors12.prod(0).sum(1) / torch.clamp(
            distances12.prod(0), min=1e-10)

        fcj12 = self.cutoff_fn(distances12, self.cutoff)
        term1 = self.Gamma * (cos_angles - torch.cos(self.ShfZ))**2
        term2 = self.EtaA * (distances12.sum(0) / 2 - self.ShfA)**2
        exponent = term1 + term2
        ret = 4 * torch.exp(-exponent) * fcj12.prod(0)
        out = ret.flatten(start_dim=1)
        return out


# Now to initialize this function I want EtaA and ShfA to be the same as in
# standard ANI 1x, but I want Gamma and ShfZ to be different.
# note that here I have one Gamma for each ShfZ, and they come in pairs and
# belong together, for this reason the shapes of the tensors and sublengths
# are a bit different than in standard angular terms
EtaA = torchani.aev.StandardAngular.like_1x().EtaA
ShfA = torchani.aev.StandardAngular.like_1x().ShfA
cutoff = torchani.aev.StandardAngular.like_1x().cutoff
ShfZ = torch.linspace(0, math.pi, 9)
Gamma = torch.tensor([1023, 146.5, 36, 18.6, 15.5, 18.6, 36, 146.5, 1023], dtype=torch.float)

custom_terms = AngularCosDiff(EtaA, ShfA, Gamma, ShfZ, cutoff, cutoff_fn='cosine')

aev_computer_cosdiff = torchani.aev.AEVComputer(radial_terms='ani1x', angular_terms=custom_terms, num_species=4).to(device)
radial_length = aev_computer_cosdiff.radial_length
species, aevs = aev_computer_cosdiff((species, coordinates))
print('AEV computer like 1x, but with custom angular terms')
print('for first atom, first 10 terms of radial:', aevs[0, 0, :10])
print('for first atom, first 10 terms of angular:', aevs[0, 0, radial_length:radial_length + 10])
print()
