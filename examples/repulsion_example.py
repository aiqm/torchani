"""
Using the XTB-Repulsion module
==========================================================

TorchANI can use semiempirical XTB repulsion to improve the description of the
potential energy surface at short distances
"""
import torch
from torchani.repulsion import RepulsionXTB, StandaloneRepulsionXTB

# This is an example of how to use the repulsion interactions coded in torchani
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# by default the dispersion interactions have no cutoff
rep = StandaloneRepulsionXTB().to(device)

coordinates = torch.tensor([[[0.03192167, 0.00638559, 0.01301679],
                             [-0.83140486, 0.39370209, -0.26395324],
                             [-0.66518241, -0.84461308, 0.20759389],
                             [0.45554739, 0.54289633, 0.81170881],
                             [0.66091919, -0.16799635, -0.91037834]]],
                           requires_grad=True, device=device)
znumbers = torch.tensor([[6, 1, 1, 1, 1]], device=device)
rep_energy = rep((znumbers, coordinates)).energies
derivative = torch.autograd.grad(rep_energy.sum(), coordinates)[0]
force = -derivative
print('Repulsion Energy:', rep_energy)
print('Force:', force.squeeze())

# repulsion can also be calculated for batches of coordinates
# (here we just repeat the methanes as an example, but different molecules
# can be passed by using dummy "-1" atoms in the species)
r = 4
coordinates = coordinates.repeat(r, 1, 1)
znumbers = znumbers.repeat(r, 1)

rep_energy = rep((znumbers, coordinates)).energies
derivative = torch.autograd.grad(rep_energy.sum(), coordinates)[0]
force = -derivative
print('Repulsion Energy:', rep_energy)
print('Force:', force.squeeze())

# By default the supported species are H C N O, but different supported species
# can also be passed down to the constructor
rep = StandaloneRepulsionXTB(symbols=('H', 'C', 'N', 'O', 'S')).to(device)
# here I changed the species a bit to make a nonesense molecules
coordinates = torch.tensor([[[0.03192167, 0.00638559, 0.01301679],
                             [-0.83140486, 0.39370209, -0.26395324],
                             [-0.66518241, -0.84461308, 0.20759389],
                             [0.45554739, 0.54289633, 0.81170881],
                             [0.66091919, -0.16799635, -0.91037834]]],
                           requires_grad=True, device=device)
znumbers = torch.tensor([[6, 16, 1, 8, 1]], device=device)
rep_energy = rep((znumbers, coordinates)).energies
derivative = torch.autograd.grad(rep_energy.sum(), coordinates)[0]
force = -derivative
print('Repulsion Energy:', rep_energy)
print('Force:', force.squeeze())

# another possibility (note that no network supports Fe at the moment, but
# StandaloneRepulsionXTB supports elements 1-86
rep = StandaloneRepulsionXTB(symbols=('H', 'N', 'Fe')).to(device)
znumbers = torch.tensor([[26, 1, 7, 26, 1]], device=device)
rep_energy = rep((znumbers, coordinates)).energies
derivative = torch.autograd.grad(rep_energy.sum(), coordinates)[0]
force = -derivative
print('Repulsion Energy:', rep_energy)
print('Force:', force.squeeze())


# Internally torchani's models should NOT use StandaloneRepulsionXTB,
# since it recalculates the neighborlist and all pairwise distances,
# internally ani models use RepulsionXTB instead, which also
# takes in distances and a neighborlist, for example, this means
# atom 0 is a neighbor of atoms 1 and 2, and atom 1 is a neighbor of atom 2
neighbor_idxs = torch.tensor([[0, 0, 1],
                             [1, 2, 2]], dtype=torch.long, device=device)
distances = torch.tensor([0.1, 0.2, 0.5], dtype=torch.double, device=device)
# since this is internal to the models instead of atomic numbers we pass
# indices
atomic_idxs = torch.tensor([[0, 0, 1]], device=device, dtype=torch.long)
rep_ = RepulsionXTB(symbols=("H", "O")).to(device)

rep_energy = rep_(
    atomic_idxs,
    neighbor_idxs,
    distances
)
print('Repulsion energy', rep_energy)
