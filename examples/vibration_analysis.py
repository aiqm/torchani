"""
Computing Vibrational Frequencies Using Analytical Hessian
==========================================================

TorchANI is able to use ASE interface to do structure optimization and
vibration analysis, but the Hessian in ASE's vibration analysis is computed
numerically, which is slow and less accurate.

TorchANI therefore provide an interface to compute the Hessian matrix and do
vibration analysis analytically, thanks to the super power of `torch.autograd`.
"""
import ase
from ase.optimize import LBFGS

import torch
import torchani


###############################################################################
# Let's now manually specify the device we want TorchANI to run:
device = torch.device("cpu")
model = torchani.models.ANI1x().to(device).double()

###############################################################################
# Let's first construct a water molecule and do structure optimization:
molecule = ase.Atoms(
    symbols=("H", "H", "O"),
    positions=[
        [0.9575, 0.0, 0.0],
        [-0.23990064, 0.92695951, 0.0],
        [0.0, 0.0, 0.0],
    ],
    calculator=model.ase(),
)
opt = LBFGS(molecule)
opt.run(fmax=1e-6)

###############################################################################
# Now let's extract coordinates and species from ASE to use it directly with
# TorchANI:
species = torch.tensor(
    molecule.get_atomic_numbers(),
    device=device,
    dtype=torch.long,
).unsqueeze(0)
coordinates = torch.tensor(
    molecule.get_positions(),
    device=device,
    dtype=torch.float,
    requires_grad=True,
).unsqueeze(0)

###############################################################################
# TorchANI needs the masses of elements in AMU to compute vibrations. The
# masses in AMU can be obtained from a tensor with atomic numbers by using
# this utility:
masses = torchani.utils.get_atomic_masses(species, dtype=torch.float)

###############################################################################
# To do vibration analysis, we first need to generate a graph that computes
# energies from species and coordinates. The code to generate a graph of energy
# is the same as the code to compute energy:
energies = model((species, coordinates)).energies

###############################################################################
# We can now use the energy graph to compute analytical Hessian matrix:
hessian = torchani.utils.hessian(coordinates, energies=energies)

###############################################################################
# The Hessian matrix should have shape `(1, 9, 9)`, where 1 means there is only
# one molecule to compute, 9 means `3 atoms * 3D space = 9 degree of freedom`.
print(hessian.shape)

###############################################################################
# We are now ready to compute vibrational frequencies. The output has unit
# cm^-1. Since there are in total 9 degree of freedom, there are in total 9
# frequencies. Only the frequencies of the 3 vibrational modes are interesting.
# We output the modes as MDU (mass deweighted unnormalized), to compare with ASE.
freq, modes, fconstants, rmasses = torchani.utils.vibrational_analysis(
    masses, hessian, mode_type="MDU"
)
torch.set_printoptions(precision=3, sci_mode=False)

print("Frequencies (cm^-1):", freq[6:])
print("Force Constants (mDyne/A):", fconstants[6:])
print("Reduced masses (AMU):", rmasses[6:])
print("Modes:", modes[6:])
