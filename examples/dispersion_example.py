"""
Using the Dispersion D3 module
==========================================================

TorchANI can use a built-in implementation of D3 dispersion for dispersion
interactions. This is meant for using with functionals that are parametrized
with dispersion in the first place.
"""
import torch
from torchani.potentials import StandaloneTwoBodyDispersionD3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# By default the dispersion interactions have no cutoff
disp = StandaloneTwoBodyDispersionD3().to(device)

coordinates = torch.tensor([[[0.03192167, 0.00638559, 0.01301679],
                             [-0.83140486, 0.39370209, -0.26395324],
                             [-0.66518241, -0.84461308, 0.20759389],
                             [0.45554739, 0.54289633, 0.81170881],
                             [0.66091919, -0.16799635, -0.91037834]]],
                           requires_grad=True, device=device)
znumbers = torch.tensor([[6, 1, 1, 1, 1]], device=device)
disp_energy = disp((znumbers, coordinates)).energies
derivative = torch.autograd.grad(disp_energy.sum(), coordinates)[0]
force = -derivative
print('Dispersion Energy:', disp_energy)
print('Force:', force.squeeze())

# dispersion can also be calculated for batches of coordinates
# (here we just repeat the methanes as an example, but different molecules
# can be passed by using dummy "-1" atoms in the species)
r = 4
coordinates = coordinates.repeat(r, 1, 1)
znumbers = znumbers.repeat(r, 1)

disp_energy = disp((znumbers, coordinates)).energies
derivative = torch.autograd.grad(disp_energy.sum(), coordinates)[0]
force = -derivative
print('Dispersion Energy:', disp_energy)
print('Force:', force.squeeze())

# By default the supported species are H C N O, but different supported species
# can also be passed down to the constructor
disp = StandaloneTwoBodyDispersionD3(symbols=('H', 'C', 'N', 'O', 'S')).to(device)
# here I changed the species a bit to make a nonesense molecules
coordinates = torch.tensor([[[0.03192167, 0.00638559, 0.01301679],
                             [-0.83140486, 0.39370209, -0.26395324],
                             [-0.66518241, -0.84461308, 0.20759389],
                             [0.45554739, 0.54289633, 0.81170881],
                             [0.66091919, -0.16799635, -0.91037834]]],
                           requires_grad=True, device=device)
znumbers = torch.tensor([[6, 16, 1, 8, 1]], device=device)
disp_energy = disp((znumbers, coordinates)).energies
derivative = torch.autograd.grad(disp_energy.sum(), coordinates)[0]
force = -derivative
print('Dispersion Energy:', disp_energy)
print('Force:', force.squeeze())

# Another possibility (note that no network supports Fe at the moment, but
# StandaloneTwoBodyDispersionD3 supports elements 1-93
disp = StandaloneTwoBodyDispersionD3(symbols=('H', 'N', 'Fe')).to(device)
znumbers = torch.tensor([[26, 1, 7, 26, 1]], device=device)
disp_energy = disp((znumbers, coordinates)).energies
derivative = torch.autograd.grad(disp_energy.sum(), coordinates)[0]
force = -derivative
print('Dispersion Energy:', disp_energy)
print('Force:', force.squeeze())
