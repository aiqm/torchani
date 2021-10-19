r"""Mypy type aliases"""
import sys
from typing import Union, Callable, Iterable, MutableMapping, TypeVar
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


_MutMapSubtype = TypeVar('_MutMapSubtype', bound=MutableMapping[str, Tensor])

# Transform = Callable[[MutableMapping[str, Tensor]], MutableMapping[str, Tensor]]
Transform = Callable[[_MutMapSubtype], _MutMapSubtype]

# any of these should be interpretable as a 1D index sequence
IdxLike = Union[Tensor, ndarray, None, Iterable[int], int]

Conformers = MutableMapping[str, Tensor]
NumpyConformers = MutableMapping[str, ndarray]
MixedConformers = MutableMapping[str, Union[Tensor, ndarray]]

# mimic typeshed
StrPath = Union[str, 'PathLike[str]']
StrPathODict = Union['OrderedDict[str, str]', 'OrderedDict[str, PathLike[str]]']
