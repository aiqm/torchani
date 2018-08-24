TorchANI
========

.. automodule:: torchani

.. autoclass:: torchani.AEVComputer
    :members:
.. autoclass:: torchani.ANIModel
.. autoclass:: torchani.Ensemble
.. autoclass:: torchani.EnergyShifter
    :members:


Datasets
========

.. automodule:: torchani.data
.. autoclass:: torchani.data.BatchedANIDataset


Utilities
=========

.. automodule:: torchani.utils
.. autofunction:: torchani.utils.pad_and_batch
.. autofunction:: torchani.utils.present_species
.. autofunction:: torchani.utils.strip_redundant_padding


NeuroChem Importers
===================

.. automodule:: torchani.neurochem
.. autoclass:: torchani.neurochem.Constants
.. autofunction:: torchani.neurochem.load_sae
.. autofunction:: torchani.neurochem.load_model
.. autofunction:: torchani.neurochem.load_model_ensemble
.. autoclass:: torchani.neurochem.Buildins


Ignite Helpers
==============

.. automodule:: torchani.ignite
.. autoclass:: torchani.ignite.Container
.. autoclass:: torchani.ignite.DictLoss
.. autoclass:: torchani.ignite.PerAtomDictLoss
.. autoclass:: torchani.ignite.TransformedLoss
.. autofunction:: torchani.ignite.MSELoss
.. autoclass:: torchani.ignite.DictMetric
.. autofunction:: torchani.ignite.RMSEMetric
