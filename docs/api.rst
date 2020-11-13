TorchANI
========

.. autoclass:: torchani.AEVComputer
    :members:
.. autoclass:: torchani.ANIModel
.. autoclass:: torchani.Ensemble
.. autoclass:: torchani.SpeciesConverter
    :members:
.. autoclass:: torchani.EnergyShifter
    :members:
.. autoclass:: torchani.nn.Gaussian


Model Zoo
=========

.. automodule:: torchani.models
.. autoclass:: torchani.models.ANI1x
    :members:
.. autoclass:: torchani.models.ANI1ccx
    :members:
.. autoclass:: torchani.models.ANI2x
    :members:


Datasets
========

.. automodule:: torchani.data



Utilities
=========

.. automodule:: torchani.utils
.. autofunction:: torchani.utils.pad_atomic_properties
.. autofunction:: torchani.utils.present_species
.. autofunction:: torchani.utils.strip_redundant_padding
.. autofunction:: torchani.utils.map2central
.. autoclass:: torchani.utils.ChemicalSymbolsToInts
    :members:
.. autofunction:: torchani.utils.hessian
.. autofunction:: torchani.utils.vibrational_analysis
.. autofunction:: torchani.utils.get_atomic_masses


NeuroChem
=========

.. automodule:: torchani.neurochem
.. autoclass:: torchani.neurochem.Constants
    :members:
.. autofunction:: torchani.neurochem.load_sae
.. autofunction:: torchani.neurochem.load_atomic_network
.. autofunction:: torchani.neurochem.load_model
.. autofunction:: torchani.neurochem.load_model_ensemble
.. autoclass:: torchani.neurochem.Trainer
    :members:
.. automodule:: torchani.neurochem.trainer


ASE Interface
=============

.. automodule:: torchani.ase
.. autoclass:: torchani.ase.Calculator

Units
=====

.. automodule:: torchani.units
.. autofunction:: torchani.units.hartree2ev
.. autofunction:: torchani.units.hartree2kcalmol
.. autofunction:: torchani.units.hartree2kjoulemol
.. autofunction:: torchani.units.ev2kcalmol
.. autofunction:: torchani.units.ev2kjoulemol
.. autofunction:: torchani.units.mhessian2fconst
.. autofunction:: torchani.units.sqrt_mhessian2invcm
.. autofunction:: torchani.units.sqrt_mhessian2milliev
