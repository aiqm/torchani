# -*- coding: utf-8 -*-
"""
Constant temperature MD using ASE interface
===========================================

This example is modified from the official `Constant temperature MD`_ to use
the ASE interface of TorchANI as energy calculator.

.. _Constant temperature MD:
    https://wiki.fysik.dtu.dk/ase/tutorials/md/md.html#constant-temperature-md
"""


###############################################################################
# To begin with, let's first import the modules we will use:
from ase.lattice.cubic import Diamond
from ase.md.langevin import Langevin
from ase import units
import torchani


###############################################################################
# Now let's set up a crystal
atoms = Diamond(symbol="C", pbc=True)

###############################################################################
# Now let's create a calculator from builtin models:
builtin = torchani.neurochem.Builtins()
calculator = torchani.ase.Calculator(builtin.species, builtin.aev_computer,
                                     builtin.models, builtin.energy_shifter)
atoms.set_calculator(calculator)

###############################################################################
# We want to run MD with constant energy using the Langevin algorithm
# with a time step of 5 fs, the temperature 50K and the friction
# coefficient to 0.02 atomic units.
dyn = Langevin(atoms, 5 * units.fs, 50 * units.kB, 0.002)


###############################################################################
# Let's print energies every 50 steps:
def printenergy(a=atoms):  # store a reference to atoms in the definition.
    """Function to print the potential, kinetic and total energy."""
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    print('Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
          'Etot = %.3feV' % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin))


dyn.attach(printenergy, interval=50)

###############################################################################
# Now run the dynamics:
printenergy()
dyn.run(500)
