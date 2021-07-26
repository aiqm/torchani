r"""Mypy type aliases"""
import sys
from typing import Dict, Union, Callable, Iterable
from torch import Tensor
from numpy import ndarray, dtype
from collections import OrderedDict
from os import PathLike

# This is needed for compatibility with python 3.6, where numpy typing doesn't
# work correctly
if sys.version_info[:2] < (3, 7):
    # This doesn't really matter anyways since it is only for mypy
    DTypeLike = dtype
else:
    from numpy import typing as numpy_typing
    DTypeLike = numpy_typing.DTypeLike

Transform = Callable[[Dict[str, Tensor]], Dict[str, Tensor]]

# any of these should be interpretable as a 1D index sequence
IdxLike = Union[Tensor, ndarray, None, Iterable[int], int]

Conformers = Dict[str, Tensor]
NumpyConformers = Dict[str, ndarray]
MixedConformers = Dict[str, Union[Tensor, ndarray]]

# mimic typeshed
StrPath = Union[str, 'PathLike[str]']
StrPathODict = Union['OrderedDict[str, str]', 'OrderedDict[str, PathLike[str]]']
