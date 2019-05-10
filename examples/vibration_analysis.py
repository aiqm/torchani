# -*- coding: utf-8 -*-
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
import ase.optimize
import torch
import torchani
import math


###############################################################################
# Let's now manually specify the device we want TorchANI to run:
model = torchani.models.ANI1x().double()

###############################################################################
# Let's first construct a water molecule and do structure optimization:
d = 0.9575
t = math.pi / 180 * 104.51
molecule = ase.Atoms('H2O', positions=[
    (d, 0, 0),
    (d * math.cos(t), d * math.sin(t), 0),
    (0, 0, 0),
], calculator=model.ase())
opt = ase.optimize.BFGS(molecule)
opt.run(fmax=1e-6)

###############################################################################
# Now let's extract coordinates and species from ASE to use it directly with
# TorchANI:
species = model.species_to_tensor(molecule.get_chemical_symbols()).unsqueeze(0)
coordinates = torch.from_numpy(molecule.get_positions()).unsqueeze(0).requires_grad_(True)

###############################################################################
# TorchANI needs to know the mass of each atom in amu in order to do vibration
# analysis:
element_masses = torch.tensor([
    1.008,  # H
    12.011,  # C
    14.007,  # N
    15.999,  # O
], dtype=torch.double)
masses = element_masses[species]

###############################################################################
# To do vibration analysis, we first need to generate a graph that computes
# energies from species and coordinates. The code to generate a graph of energy
# is the same as the code to compute energy:
_, energies = model((species, coordinates))

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
freq = torchani.utils.vibrational_analysis(masses, hessian)[-3:]
print(freq)
