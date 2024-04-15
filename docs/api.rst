TorchANI
========

Atomic Environment Vectors
==========================

.. autoclass:: torchani.aev.AEVComputer
    :members:
.. autoclass:: torchani.aev.aev_terms.StandardRadial
    :members:
.. autoclass:: torchani.aev.aev_terms.StandardAngular
    :members:

Cutoff functions
================

.. autoclass:: torchani.cutoffs.Cutoff
    :members:
.. autoclass:: torchani.cutoffs.CutoffSmooth
    :members:
.. autoclass:: torchani.cutoffs.CutoffCosine
    :members:

Neighborlists
=============

.. automodule:: torchani.neighbors
.. autoclass:: torchani.neighbors.FullPairwise
.. autoclass:: torchani.neighbors.CellList

Atomic Networks and containers
==============================

.. autoclass:: torchani.ANIModel
.. autoclass:: torchani.Ensemble
.. automodule:: torchani.atomics

Builtin Models
==============

.. automodule:: torchani.models
.. autoclass:: torchani.models.ANI1x
    :members:
.. autoclass:: torchani.models.ANI1ccx
    :members:
.. autoclass:: torchani.models.ANI2x
    :members:

Datasets
========

.. automodule:: torchani.datasets
.. autoclass:: torchani.datasets.ANIDataset
    :members:
.. autoclass:: torchani.datasets.ANIBatchedDataset
    :members:
.. autofunction:: torchani.datasets.create_batched_dataset

Utilities
=========

.. automodule:: torchani.utils
.. autofunction:: torchani.utils.pad_atomic_properties
.. autofunction:: torchani.utils.present_species
.. autofunction:: torchani.utils.strip_redundant_padding
.. autofunction:: torchani.utils.map_to_central
.. autoclass:: torchani.utils.ChemicalSymbolsToInts
    :members:
.. autofunction:: torchani.utils.hessian
.. autofunction:: torchani.utils.vibrational_analysis
.. autofunction:: torchani.utils.get_atomic_masses
.. autoclass:: torchani.SpeciesConverter
    :members:
.. autoclass:: torchani.EnergyShifter
    :members:

ASE Interface
=============

.. automodule:: torchani.ase
.. autoclass:: torchani.ase.Calculator

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
