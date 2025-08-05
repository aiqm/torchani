r"""
Molecular dynamics using ASE
============================

This example is modified from the official `home page`_ and `Constant temperature MD`_
to use the ASE interface of TorchANI as energy calculator.

.. _home page:
    https://wiki.fysik.dtu.dk/ase/
.. _Constant temperature MD:
    https://wiki.fysik.dtu.dk/ase/tutorials/md/md.html#constant-temperature-md
"""
# %%
# As always, we start by importing the modules we need
import ase
from ase.lattice.cubic import Diamond
from ase.md.langevin import Langevin
from ase.optimize import LBFGS

import torchani
# %%
# First we set up our system (in this case a diamond crystal, with PBC enabled)
atoms = Diamond(symbol="C", pbc=True)
len(atoms)  # The number of atoms in the system
# %%
# After, we create a calculator from an ANI model and attach it to our atoms
atoms.calc = torchani.models.ANI2x().ase()
# %%
# Then we minimize our system using the
# `L-BFGS <https://en.wikipedia.org/wiki/Limited-memory_BFGS>`_ optimizer,
# which is included in ASE, under `ase.optimize.LBFGS`.
opt = LBFGS(atoms)
opt.run(fmax=0.0002)
# %%
# We want to run constant temperature MD, and print some quantities throughout the MD.
# For this we need to create a callback (function) that prints the quantities we are
# interested in. For example, to print the energy of the system we can use:


def print_energy(atoms: ase.Atoms):
    pot_energy = atoms.get_potential_energy() / len(atoms)
    kin_energy = atoms.get_kinetic_energy() / len(atoms)
    temperature = kin_energy / (1.5 * ase.units.kB)
    tot_energy = pot_energy + kin_energy
    print(
        "Energy per atom: \n"
        f"    E_pot = {pot_energy:.3f} eV\n"
        f"    E_kin = {kin_energy:.3f} eV (T = {temperature:.1f} K)\n"
        f"    E_tot = {tot_energy:.3f} eV\n"
    )


# %%
# We will use the Langevin thermostat to control the temperature. To do this we need to
# first construct an `ase.md.langevin.Langevin` object. Here we use with a time step of
# 1 fs, a temperature of 300 K and a friction coefficient of 0.2
dyn = Langevin(
    atoms,
    timestep=1 * ase.units.fs,
    temperature_K=300,
    friction=0.2,
)
dyn.attach(print_energy, interval=5, atoms=atoms)
# %%
# Finally we run the dynamics using ``dyn.run(steps)``
print_energy(atoms)  # Print the initial energy
dyn.run(50)
