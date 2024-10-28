Atomic Environment Vectors (AEV)
================================

.. automodule:: torchani.aev
.. autoclass:: torchani.aev.AEVComputer
    :members:
.. autoclass:: torchani.aev.AngularTerm
    :members:
.. autoclass:: torchani.aev.RadialTerm
    :members:
.. autoclass:: torchani.aev.StandardRadial
    :members:
.. autoclass:: torchani.aev.StandardAngular
    :members:

Cutoff Functions
================

.. automodule:: torchani.cutoffs
.. autoclass:: torchani.cutoffs.Cutoff
    :members:
.. autoclass:: torchani.cutoffs.CutoffSmooth
    :members:
.. autoclass:: torchani.cutoffs.CutoffCosine
    :members:

Neighborlists
=============

.. automodule:: torchani.neighbors
.. autoclass:: torchani.neighbors.Neighborlist
    :members:
.. autoclass:: torchani.neighbors.FullPairwise
    :members:
.. autoclass:: torchani.neighbors.CellList
    :members:

Atomic Networks and Containers
==============================

.. automodule:: torchani.nn
.. autoclass:: torchani.nn.ANINetworks
    :members:
.. autoclass:: torchani.nn.ANIEnsemble
    :members:
.. autoclass:: torchani.nn.BmmEnsemble
    :members:
.. autoclass:: torchani.nn.AtomicNetwork
    :members:
.. autoclass:: torchani.nn.BmmAtomicNetwork
    :members:
.. autoclass:: torchani.nn.BmmLinear
    :members:
.. autoclass:: torchani.nn.MNPNetworks
    :members:
.. autofunction:: torchani.nn.make_1x_network
.. autofunction:: torchani.nn.make_2x_network
.. autofunction:: torchani.nn.make_dr_network

Assembly of custom ANI-style Models
===================================

.. automodule:: torchani.assembly
.. autoclass:: torchani.assembly.ANI
    :members:
.. autoclass:: torchani.assembly.ANIq
    :members:
.. function:: torchani.assembly.simple_ani
.. function:: torchani.assembly.simple_aniq

Built-in ANI-style Models
=========================

.. automodule:: torchani.models
.. autofunction:: torchani.models.ANI1x
.. autofunction:: torchani.models.ANI1ccx
.. autofunction:: torchani.models.ANI2x
.. autofunction:: torchani.models.ANIdr
.. autofunction:: torchani.models.ANImbis

Potentials
==========

.. automodule:: torchani.potentials
.. autoclass:: torchani.potentials.RepulsionXTB
    :members:
.. autoclass:: torchani.potentials.TwoBodyDispersionD3
    :members:
.. autoclass:: torchani.potentials.EnergyAdder
    :members:

Electrostatics
==============

.. automodule:: torchani.electro
.. autoclass:: torchani.electro.DipoleComputer
    :members:
.. autoclass:: torchani.electro.ChargeNormalizer
    :members:
.. autofunction:: torchani.electro.compute_dipole

General Utilities
=================

.. automodule:: torchani.utils
.. autofunction:: torchani.utils.pad_atomic_properties
.. autofunction:: torchani.utils.map_to_central
.. autofunction:: torchani.utils.atomic_numbers_to_masses
.. autoclass:: torchani.utils.AtomicNumbersToMasses
    :members:
.. autoclass:: torchani.utils.ChemicalSymbolsToInts
    :members:
.. autoclass:: torchani.SpeciesConverter
    :members:

Forces, Hessians, Normal Modes
==============================

.. automodule:: torchani.grad
.. autofunction:: torchani.grad.forces
.. autofunction:: torchani.grad.hessians
.. autofunction:: torchani.grad.energies_and_forces
.. autofunction:: torchani.grad.energies_forces_and_hessians
.. autofunction:: torchani.grad.vibrational_analysis

ASE Interface
=============

.. automodule:: torchani.ase
.. autoclass:: torchani.ase.Calculator
    :members:

Units
=====

.. automodule:: torchani.units
.. autofunction:: torchani.units.hartree2ev
.. autofunction:: torchani.units.hartree2kcalpermol
.. autofunction:: torchani.units.hartree2kjoulepermol
.. autofunction:: torchani.units.ev2kcalpermol
.. autofunction:: torchani.units.ev2kjoulepermol
.. autofunction:: torchani.units.mhessian2fconst
.. autofunction:: torchani.units.sqrt_mhessian2invcm
.. autofunction:: torchani.units.sqrt_mhessian2milliev
