r"""Type aliases"""
import typing as tp
from pathlib import Path

import numpy as np
from torch import Tensor
from numpy.typing import NDArray


# Any of these should be interpretable as a 1D index sequence
IdxLike = tp.Union[Tensor, NDArray[np.int_], None, tp.Iterable[int], int]

Conformers = tp.MutableMapping[str, Tensor]
NumberOrStrArray = tp.Union[NDArray[np.int_], NDArray[np.float_], NDArray[np.str_]]
NumpyConformers = tp.MutableMapping[str, NumberOrStrArray]
MixedConformers = tp.MutableMapping[str, tp.Union[Tensor, NumberOrStrArray]]

# mimic typeshed
StrPath = tp.Union[str, Path]
StrPathODict = tp.Union[tp.OrderedDict[str, str], tp.OrderedDict[str, Path]]
