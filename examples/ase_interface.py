"""
Structure minimization and constant temperature MD using ASE interface
======================================================================

This example is modified from the official `home page` and
`Constant temperature MD`_ to use the ASE interface of TorchANI as energy
calculator.

.. _home page:
    https://wiki.fysik.dtu.dk/ase/
.. _Constant temperature MD:
    https://wiki.fysik.dtu.dk/ase/tutorials/md/md.html#constant-temperature-md
"""
# To begin with, let's import the modules we will use:
import ase
from ase.lattice.cubic import Diamond
from ase.md.langevin import Langevin
from ase.optimize import LBFGS

import torchani

###############################################################################
# First we set up our system (in this case a diamond crystal, with PBC enabled)
atoms = Diamond(symbol="C", pbc=True)
print(f"Num atoms in cell: {len(atoms)}")

###############################################################################
# Afterwards we create a calculator from an ANI model and attach it to our atoms
atoms.calc = torchani.models.ANI2x().ase()

###############################################################################
# Now we minimize our system
print("Starting minimization...")
opt = LBFGS(atoms)
opt.run(fmax=0.0002)
print()

###############################################################################
# We want to run constant temperature MD, and print some quantities throughout
# the MD. For this we need to create a callback function that prints
# the quantities we are interested in


def print_energy(atoms: ase.Atoms):
    r"""Function to print the potential, kinetic and total energies"""
    epot = atoms.get_potential_energy() / len(atoms)
    ekin = atoms.get_kinetic_energy() / len(atoms)
    temperature = ekin / (1.5 * ase.units.kB)
    etot = epot + ekin
    print(
        "Energy per atom: \n"
        f"    E_pot = {epot:.3f} eV\n"
        f"    E_kin = {ekin:.3f} eV (T = {temperature:.1f} K)\n"
        f"    E_tot = {etot:.3f} eV\n"
    )

###############################################################################
# We then set up the Langevin thermostat with a time step of 1 fs, a
# temperature of 300 K and a friction coefficient of 0.2


dyn = Langevin(
    atoms,
    timestep=1 * ase.units.fs,
    temperature_K=300,
    friction=0.2,
)
dyn.attach(print_energy, interval=5, atoms=atoms)

###############################################################################
# Finally we run the dynamics
print("Running dynamics...")
print_energy(atoms)
dyn.run(50)
