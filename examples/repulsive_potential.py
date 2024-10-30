r"""
Using xTB repulsion
===================

TorchANI can use semiempirical XTB repulsion to improve the description of the
potential energy surface at short distances
"""
# To begin with, let's first import the modules and setup devices we will use:
import torch

from torchani.potentials import RepulsionXTB
from torchani.grad import energies_and_forces

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rep = RepulsionXTB(cutoff=5.2, symbols=("H", "C", "N", "O")).to(device)

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
znumbers = torch.tensor([[6, 1, 1, 1, 1]], device=device)
rep_energy, force = energies_and_forces(rep, znumbers, coordinates)
print("Repulsion Energy:", rep_energy)
print("Force:", force.squeeze())

# Repulsion can also be calculated for batches of coordinates
# (here we just repeat the methanes as an example, but different molecules
# can be passed by using dummy "-1" atoms in the species)
r = 4
coordinates = coordinates.repeat(r, 1, 1)
znumbers = znumbers.repeat(r, 1)
rep_energy, force = energies_and_forces(rep, znumbers, coordinates)
print("Repulsion Energy:", rep_energy)
print("Force:", force.squeeze())

# Different supported species can be passed down to the constructor
rep = RepulsionXTB(cutoff=5.2, symbols=("H", "C", "N", "O", "S", "Fe")).to(
    device
)
# Here we change the species a bit to make a nonesense molecule
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
znumbers = torch.tensor([[6, 16, 1, 8, 26]], device=device)
rep_energy, force = energies_and_forces(rep, znumbers, coordinates)
print("Repulsion Energy:", rep_energy)
print("Force:", force.squeeze())
