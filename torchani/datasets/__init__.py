r"""
Functions and classes for creating batched and map-like datasets.
Backends for the on-disk map-like datasets inlcude HDF5, Apache Parquet and Zarr.

Filters and miscellaneous utilities can be found in the ``datasets.utils``
module.

For a tutorial introduction on  the use of this API consult the relevant
examples in the documentation.

If you want to extend the capabilities to support different datasets consult
the datasets backend extension example.

Built-in datasets are downloaded on first instantiation. each built-in dataset
that this module provides access to is calculated with a specific level of
theory (LoT) which is in general specified as a combination of
functional/basis_set or wavefunction_method/basis_set when appropriate.

Some of the provided built-in datasets have been published in ANI papers, and
some are external freely available datasets published elsewhere, that have been
reformatted to conform to TorchANI's API. If you use any of these datasets in
your work please cite the relevate article(s).
"""

from torchani.datasets import utils
from torchani.datasets.utils import datapack

# Some attrs are created programmatically, so a star import is needed
from torchani.datasets.builtin import *  # noqa:F403,F401
from torchani.datasets.builtin import (
    DatasetId,
    LotId,
    datapull,
    datainfo,
    _BUILTIN_DATASETS_SPEC,
)
from torchani.datasets.anidataset import ANIDataset
from torchani.datasets.batching import (
    BatchedDataset,
    ANIBatchedInMemoryDataset,
    ANIBatchedDataset,
    Batcher,
    create_batched_dataset,
    batch_all_in_ram,
)


__all__ = [
    "BatchedDataset",
    "ANIBatchedDataset",
    "ANIBatchedInMemoryDataset",
    "ANIDataset",
    "Batcher",
    "create_batched_dataset",
    "batch_all_in_ram",
    "utils",
    "datapull",
    "datainfo",
    "datapack",
    "DatasetId",
    "LotId",
]

# download_builtin_dataset defined from star import
__all__ += _BUILTIN_DATASETS_SPEC  # noqa:F405
