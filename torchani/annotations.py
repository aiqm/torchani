r"""
Type aliases, common type annotations used throughout the TorchANI codebase
"""
import torch
import typing as tp
from pathlib import Path

import numpy as np
from torch import Tensor
from numpy.typing import NDArray

# Any of these should be interpretable as a 1D index sequence
IdxLike = tp.Union[Tensor, NDArray[np.int_], None, tp.Iterable[int], int]

Conformers = tp.MutableMapping[str, Tensor]
NumberOrStrArray = tp.Union[NDArray[np.int_], NDArray[np.float64], NDArray[np.str_]]
NumpyConformers = tp.MutableMapping[str, NumberOrStrArray]
MixedConformers = tp.MutableMapping[str, tp.Union[Tensor, NumberOrStrArray]]

Device = tp.Union[str, torch.device]

# Mimic typeshed
StrPath = tp.Union[str, Path]

# Datasets
Grouping = tp.Literal["by_num_atoms", "by_formula"]
Backend = tp.Literal["hdf5", "zarr", "pandas", "cudf"]
