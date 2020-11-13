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
device = torch.device('cpu')
model = torchani.models.ANI1x(periodic_table_index=True).to(device).double()

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
species = torch.tensor(molecule.get_atomic_numbers(), device=device, dtype=torch.long).unsqueeze(0)
coordinates = torch.from_numpy(molecule.get_positions()).unsqueeze(0).requires_grad_(True)

###############################################################################
# TorchANI needs the masses of elements in AMU to compute vibrations. The
# masses in AMU can be obtained from a tensor with atomic numbers by using
# this utility:
masses = torchani.utils.get_atomic_masses(species)

###############################################################################
# We can use :func:`torch.autograd.functional.hessian` to compute hessian:
hessian = torch.autograd.functional.hessian(lambda x: model((species, x)).energies, coordinates)

###############################################################################
# The Hessian matrix should have shape `(1, 3, 3, 1, 3, 3)`, where 1 means there
# is only one molecule to compute, 3 means 3 atoms and 3D space.
print(hessian.shape)

###############################################################################
# We are now ready to compute vibrational frequencies. The output has unit
# cm^-1. Since there are in total 9 degree of freedom, there are in total 9
# frequencies. Only the frequencies of the 3 vibrational modes are interesting.
# We output the modes as MDU (mass deweighted unnormalized), to compare with ASE.
freq, modes, fconstants, rmasses = torchani.utils.vibrational_analysis(masses, hessian, mode_type='MDU')
torch.set_printoptions(precision=3, sci_mode=False)

print('Frequencies (cm^-1):', freq[6:])
print('Force Constants (mDyne/A):', fconstants[6:])
print('Reduced masses (AMU):', rmasses[6:])
print('Modes:', modes[6:])
