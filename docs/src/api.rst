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
====================================

.. autoclass:: torchani.nn.ANIModel
    :members:
.. autoclass:: torchani.nn.Ensemble
    :members:
.. automodule:: torchani.atomics
.. autofunction:: torchani.atomics.like_1x
.. autofunction:: torchani.atomics.like_1ccx
.. autofunction:: torchani.atomics.like_2x
.. autofunction:: torchani.atomics.like_dr

Built-in Models
===============

.. automodule:: torchani.models
.. autoclass:: torchani.models.ANI1x
    :members:
.. autoclass:: torchani.models.ANI1ccx
    :members:
.. autoclass:: torchani.models.ANI2x
    :members:
.. autoclass:: torchani.models.ANIdr
    :members:

Datasets
========

.. automodule:: torchani.datasets
.. autoclass:: torchani.datasets.ANIDataset
    :members:
.. autoclass:: torchani.datasets.ANIBatchedDataset
    :members:
.. autofunction:: torchani.datasets.create_batched_dataset

Dataset Transforms
==================

.. automodule:: torchani.transforms
.. autoclass:: torchani.transforms.Transform
    :members:
.. autoclass:: torchani.transforms.SubtractEnergyAndForce
    :members:
.. autoclass:: torchani.transforms.SubtractRepulsionXTB
    :members:
.. autoclass:: torchani.transforms.SubtractTwoBodyDispersionD3
    :members:
.. autoclass:: torchani.transforms.SubtractSAE
    :members:
.. autoclass:: torchani.transforms.AtomicNumbersToIndices
    :members:
.. autoclass:: torchani.transforms.Compose
    :members:

Potentials
==========

.. automodule:: torchani.potentials
.. autoclass:: torchani.potentials.RepulsionXTB
    :members:
.. autoclass:: torchani.potentials.StandaloneRepulsionXTB
    :members:
.. autoclass:: torchani.potentials.TwoBodyDispersionD3
    :members:
.. autoclass:: torchani.potentials.StandaloneTwoBodyDispersionD3
    :members:
.. autoclass:: torchani.potentials.EnergyAdder
    :members:
.. autoclass:: torchani.potentials.wrapper.PotentialWrapper
    :members:

Creation of 3D Geometries
=========================

.. automodule:: torchani.geometry
.. autofunction:: torchani.geometry.displace
.. autoclass:: torchani.geometry.Displacer
    :members:
.. autofunction:: torchani.geometry.tile_into_tight_cell

Electrostatics
==============

.. automodule:: torchani.electro
.. autoclass:: torchani.electro.DipoleComputer
    :members:
.. autofunction:: torchani.electro.compute_dipole
.. autoclass:: torchani.electro.ChargeNormalizer
    :members:

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

Optimizing Models for Inference
===============================

.. automodule:: torchani.infer
.. autoclass:: torchani.infer.BmmEnsemble
    :members:
.. autoclass:: torchani.infer.BmmAtomicNetwork
    :members:
.. autoclass:: torchani.infer.BmmLinear
    :members:
.. autoclass:: torchani.infer.InferModel
    :members:

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
