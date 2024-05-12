r"""
Modifying and Customizing the AEV
=================================

TorchANI allows for modification and customization of the AEV features
"""
# To begin with, let's first import the modules and setup devices we will use:
import math

import torch
from torch import Tensor

import torchani
from torchani.cutoffs import Cutoff
from torchani.aev import StandardRadial, AEVComputer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# This example is meant to show how to take advantage of the modular AEV implementation
# We will use these coordinates and species:
coordinates = torch.tensor(
    [
        [
            [0.03192167, 0.00638559, 0.01301679],
            [-0.83140486, 0.39370209, -0.26395324],
            [-0.66518241, -0.84461308, 0.20759389],
            [0.45554739, 0.54289633, 0.81170881],
            [0.66091919, -0.16799635, -0.91037834],
        ]
    ],
    device=device,
)
species = torch.tensor([[1, 0, 0, 0, 0]], device=device)

# Suppose that we want to make an aev computer in the ANI 2x style:
aev_computer = AEVComputer.style_2x().to(device)
species, aevs = aev_computer((species, coordinates))
radial_length = aev_computer.radial_length
print("AEV computer like 2x")
print("for first atom, first 10 terms of radial:", aevs[0, 0, :10])
print(
    "for first atom, first 10 terms of angular:",
    aevs[0, 0, radial_length: radial_length + 10],
)
print()

# suppose we want to make an AEV computer essentially like the 1x aev computer,
# but using a different cutoff function, such as a smooth cutoff
#
# WARNING: Be very careful, if a model has not been trained using this cutoff function
# then using this aev computer with it will give nonsensical results

aev_computer_smooth = AEVComputer.style_1x(cutoff_fn="smooth").to(device)
radial_length = aev_computer_smooth.radial_length
species, aevs = aev_computer_smooth((species, coordinates))
print("AEV computer like 1x, but with a smooth cutoff")
print("for first atom, first 10 terms of radial:", aevs[0, 0, :10])
print(
    "for first atom, first 10 terms of angular:",
    aevs[0, 0, radial_length: radial_length + 10],
)
print()

# Lets say now we want to experiment with a different cutoff function, such as
# a biweight cutoff (WARNING: biweight does not have a continuous
# second derivative at the cutoff radius value, this is done just as an example)

# Since biweight is not coded in Torchani we can code it ourselves and pass it
# to the AEVComputer, as long as the forward method has this form, it will work!

# The same cutoff function will be used for both radial and angular terms


class CutoffBiweight(Cutoff):
    def __init__(self):
        super().__init__()

    def forward(self, distances: Tensor, cutoff: float) -> Tensor:
        # assuming all elements in distances are smaller than cutoff
        return (cutoff**2 - distances**2) ** 2 / cutoff**4


aev_computer_bw = AEVComputer.style_1x(cutoff_fn=CutoffBiweight()).to(device)
radial_length = aev_computer_bw.radial_length
species, aevs = aev_computer_smooth((species, coordinates))
print("AEV computer like 1x, but with a custom cutoff function")
print("for first atom, first 10 terms of radial:", aevs[0, 0, :10])
print(
    "for first atom, first 10 terms of angular:",
    aevs[0, 0, radial_length: radial_length + 10],
)
print()


# Now lets try something a bit more complicated. I want to experiment with
# different angular terms that have a form of exp(-gamma * (cos(theta) -
# cos(theta0))**2) how can I do that? I can pass this function to torchani, as
# long as it exposes the same API as StandardAngular (it has to have a
# *sublength*, a *cutoff*, and a *forward method* with the same signature)


class AngularCosDiff(torch.nn.Module):
    cutoff: float
    sublength: int
    ShfZ: Tensor
    Gamma: Tensor
    ShfA: Tensor
    EtaA: Tensor

    def __init__(self, EtaA, ShfA, Gamma, ShfZ, cutoff, cutoff_fn="cosine"):
        super().__init__()
        self.register_buffer("Gamma", Gamma.view(1, 1, -1))
        self.register_buffer("EtaA", EtaA.view(-1, 1, 1))
        self.register_buffer("ShfA", ShfA.view(1, -1, 1))
        self.register_buffer("ShfZ", ShfZ.view(1, 1, -1))
        self.cutoff_fn = torchani.cutoffs.parse_cutoff_fn(cutoff_fn)
        self.cutoff = cutoff

        assert self.ShfZ.numel() == self.Gamma.numel()

        self.sublength = self.EtaA.numel() * self.ShfA.numel() * self.ShfZ.numel()

    def forward(self, vectors12: Tensor) -> Tensor:
        vectors12 = vectors12.view(2, -1, 3, 1, 1, 1)
        distances12 = vectors12.norm(2, dim=2)
        cos_angles = vectors12.prod(0).sum(1) / torch.clamp(
            distances12.prod(0), min=1e-10
        )

        fcj12 = self.cutoff_fn(distances12, self.cutoff)
        term1 = self.Gamma * (cos_angles - torch.cos(self.ShfZ)) ** 2
        term2 = self.EtaA * (distances12.sum(0) / 2 - self.ShfA) ** 2
        exponent = term1 + term2
        ret = 4 * torch.exp(-exponent) * (fcj12[0] * fcj12[1])
        out = ret.flatten(start_dim=1)
        return out


# Now lets initialize this function with some parameters
EtaA = torch.tensor([8.0])
ShfA = torch.tensor([0.9000, 1.5500, 2.2000, 2.8500], dtype=torch.float,)
ShfZ = torch.linspace(0, math.pi, 9)
cutoff = 3.5
Gamma = torch.tensor(
    [1023, 146.5, 36, 18.6, 15.5, 18.6, 36, 146.5, 1023],
    dtype=torch.float,
)

# We will use standard radial terms in the ani-1x style but our custom angular terms
aev_computer_cosdiff = AEVComputer(
    radial_terms=StandardRadial.style_1x(),
    angular_terms=AngularCosDiff(EtaA, ShfA, Gamma, ShfZ, cutoff, cutoff_fn="cosine"),
    num_species=4,
).to(device)

radial_length = aev_computer_cosdiff.radial_length
species, aevs = aev_computer_cosdiff((species, coordinates))
print("AEV computer similar to 1x, but with custom angular terms")
print("for first atom, first 10 terms of radial:", aevs[0, 0, :10])
print(
    "for first atom, first 10 terms of angular:",
    aevs[0, 0, radial_length: radial_length + 10],
)
print()
