r"""
Computing vibrational frequencies
=================================

TorchANI is able to use ASE interface to do structure optimization and
vibration analysis, but the Hessian in ASE's vibration analysis is computed
numerically, which is slow and less accurate.

TorchANI therefore provide an interface to compute the Hessian matrix and do
vibration analysis analytically, thanks to the power of `torch.autograd`.
"""

# %%
# As always, we start by importing the modules we need
import ase
from ase.optimize import LBFGS

import torch
from torchani.models import ANI1x
from torchani.grad import energies_forces_and_hessians, vibrational_analysis
from torchani.utils import get_atomic_masses

# %%
# Let's now manually specify the device we want TorchANI to run:
device = torch.device("cpu")
model = ANI1x(device=device, dtype=torch.double)
# %%
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
# %%
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
).unsqueeze(0)
# %%
# To do vibrational analysis, we need the hessian matrix. and the masses
# of the elements in AMU. The
# masses in AMU can be obtained from a tensor with atomic numbers by using
# the `torchani.utils.get_atomic_masses`, and the hessian can be calculated together
# with the energies and forces (note that it is no loss to also calculate energies and
# forces, even when we don't use them, since TorchANI uses `torch.autograd` to do this,
# which would internally calculate energies and forces anyways).
masses = get_atomic_masses(species, dtype=torch.float)
hessian, _, _ = energies_forces_and_hessians(model, species, coordinates)
# %%
# The Hessian matrix should have shape ``(1, 9, 9)``, where 1 means there is only
# one molecule to compute, 9 means "3 atoms * 3D space = 9 degree of freedom".
hessian.shape
# %%
# We are now ready to compute vibrational frequencies. The output has unit
# cm^-1. Since there are in total 9 degree of freedom, there are in total 9
# frequencies. Only the frequencies of the 3 vibrational modes are interesting.
# We output the modes as MDU (mass deweighted unnormalized), to compare with ASE.
freq, modes, fconstants, rmasses = vibrational_analysis(
    masses, hessian, mode_kind="mdu"
)
torch.set_printoptions(precision=3, sci_mode=False)
print("Frequencies (cm^-1):", freq[6:])
print("Force Constants (mDyne/A):", fconstants[6:])
print("Reduced masses (AMU):", rmasses[6:])
print("Modes:", modes[6:])
