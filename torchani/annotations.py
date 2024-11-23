r"""Type aliases; common type annotations used throughout the code"""

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

# These are slightly different from torch annotations, to support None
# Device can be passed to `device=`
# e.g. Tensor.to(device: Device) or tensor(..., device: Device)
Device = tp.Union[torch.device, str, int, None]
# DType can be passed to `dtype=`
# e.g. Tensor.to(dtype: DType) or tensor(..., device: DType)
DType = tp.Optional[torch.dtype]

# Mimic typeshed
StrPath = tp.Union[str, Path]

# Python scalar
PyScalar = tp.Union[bool, int, float, str, None]

# Datasets
Grouping = tp.Literal["by_num_atoms", "by_formula"]
Backend = tp.Literal["hdf5", "zarr", "pandas", "cudf"]

# Ase support
StressKind = tp.Literal["scaling", "fdotr", "numerical"]
