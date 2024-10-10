Datasets
========

.. automodule:: torchani.datasets
.. autoclass:: torchani.datasets.ANIDataset
    :members:
.. autoclass:: torchani.datasets.ANIBatchedDataset
    :members:
.. autoclass:: torchani.datasets.ANIBatchedInMemoryDataset
    :members:
.. autoclass:: torchani.datasets.Batcher
    :members:
.. autofunction:: torchani.datasets.create_batched_dataset
.. autofunction:: torchani.datasets.batch_all_in_ram

Transforms
==========

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

Built-in Datasets
=================

.. automodule:: torchani.datasets.builtin
.. autofunction:: torchani.datasets.builtin.TestData
.. autofunction:: torchani.datasets.builtin.TestDataIons
.. autofunction:: torchani.datasets.builtin.TestDataForcesDipoles
.. autofunction:: torchani.datasets.builtin.IonsVeryHeavy
.. autofunction:: torchani.datasets.builtin.IonsHeavy
.. autofunction:: torchani.datasets.builtin.IonsLight
.. autofunction:: torchani.datasets.builtin.ANI1q
.. autofunction:: torchani.datasets.builtin.ANI2qHeavy
.. autofunction:: torchani.datasets.builtin.ANI1ccx
.. autofunction:: torchani.datasets.builtin.AminoacidDimers
.. autofunction:: torchani.datasets.builtin.ANI1x
.. autofunction:: torchani.datasets.builtin.ANI2x
.. autofunction:: torchani.datasets.builtin.COMP6v1
.. autofunction:: torchani.datasets.builtin.COMP6v2
.. autofunction:: torchani.datasets.builtin.QM9C7O2H10
.. autofunction:: torchani.datasets.builtin.QM9
.. autofunction:: torchani.datasets.builtin.Iso17EquilibriumSet1
.. autofunction:: torchani.datasets.builtin.Iso17EquilibriumSet2
.. autofunction:: torchani.datasets.builtin.Iso17TestSet1
.. autofunction:: torchani.datasets.builtin.Iso17TestSet2
.. autofunction:: torchani.datasets.builtin.Iso17TrainSet1
.. autofunction:: torchani.datasets.builtin.SN2
.. autofunction:: torchani.datasets.builtin.ANICCScan
.. autofunction:: torchani.datasets.builtin.DielsAlder
.. autofunction:: torchani.datasets.builtin.ANI1e
.. autofunction:: torchani.datasets.builtin.SPICEDes370K
.. autofunction:: torchani.datasets.builtin.SPICEDesMonomers
.. autofunction:: torchani.datasets.builtin.SPICEDipeptides
.. autofunction:: torchani.datasets.builtin.SPICEIonPairs
.. autofunction:: torchani.datasets.builtin.SPICEPubChem2xCompatible
.. autofunction:: torchani.datasets.builtin.SPICEPubChem
.. autofunction:: torchani.datasets.builtin.SPICESolvatedAminoacids
.. autofunction:: torchani.datasets.builtin.SolvatedProteinFragments
.. autofunction:: torchani.datasets.builtin.Train3BPAMixedT
.. autofunction:: torchani.datasets.builtin.Train3BPA300K
.. autofunction:: torchani.datasets.builtin.Test3BPA300K
.. autofunction:: torchani.datasets.builtin.Test3BPA600K
.. autofunction:: torchani.datasets.builtin.Test3BPA1200K
.. autofunction:: torchani.datasets.builtin.Test3BPADihedral180
.. autofunction:: torchani.datasets.builtin.Test3BPADihedral150
.. autofunction:: torchani.datasets.builtin.Test3BPADihedral120
.. autofunction:: torchani.datasets.builtin.ANIExCorr
