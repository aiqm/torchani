from abc import ABC, abstractmethod
from typing import ContextManager, Mapping, Set, Tuple, Any
from collections import OrderedDict

import numpy as np

from .._annotations import NumpyConformers, StrPath


class CacheHolder:
    group_sizes: 'OrderedDict[str, int]'
    properties: Set[str]

    def __init__(self) -> None:
        self.group_sizes = OrderedDict()
        self.properties = set()


# ConformerGroupAdaptor and StoreAdaptor are abstract classes from which
# all backends should inherit in order to correctly interact with ANIDataset.
# adding support for a new backend can be done just by coding these two classes and
# adding the support for the backend inside StoreAdaptorFactory
class _ConformerGroupAdaptor(Mapping[str, np.ndarray], ABC):

    def __init__(self, *args, **kwargs) -> None:
        self._dummy_properties = kwargs.pop("dummy_properties", dict())

    def create_numpy_values(self, conformers: NumpyConformers) -> None:
        for p, v in conformers.items():
            self._create_property_with_data(p, v)

    def append_numpy_values(self, conformers: NumpyConformers) -> None:
        for p, v in conformers.items():
            self._append_property_with_data(p, v)

    @property
    @abstractmethod
    def is_resizable(self) -> bool: pass  # noqa E704

    @abstractmethod
    def _append_property_with_data(self, p: str, data: np.ndarray) -> None: pass  # noqa E704

    @abstractmethod
    def _create_property_with_data(self, p: str, data: np.ndarray) -> None: pass  # noqa E704

    @abstractmethod
    def move(self, src: str, dest: str) -> None: pass  # noqa E704

    @abstractmethod
    def __delitem__(self, k: str) -> None: pass  # noqa E704

    def __getitem__(self, p: str) -> np.ndarray:
        try:
            array = self._getitem_impl(p)
        except KeyError:
            # A dummy property is defined with a padding value, a dtype, a shape, and an "is atomic" flag
            # example: dummy_params =
            # {'dtype': np.int64, 'extra_dims': (3,), 'is_atomic': True, 'fill_value': 0.0}
            # this generates a property with shape (C, A, extra_dims), filled with value 0.0
            params = self._dummy_properties[p]
            array = self._make_dummy_property(**params)
        assert isinstance(array, np.ndarray)
        return array

    @abstractmethod
    def _getitem_impl(self, p: str) -> np.ndarray: pass  # noqa E704

    def _make_dummy_property(self, extra_dims: Tuple[int, ...] = tuple(), is_atomic: bool = False, fill_value: float = 0.0, dtype=np.int64):
        try:
            species = self._getitem_impl('species')
        except KeyError:
            species = self._getitem_impl('numbers')
        if species.ndim != 2:
            raise RuntimeError("Attempted to create dummy properties in a legacy dataset, this is not supported!")
        shape: Tuple[int, ...] = (species.shape[0],)
        if is_atomic:
            shape += (species.shape[1],)
        return np.full(shape + extra_dims, fill_value, dtype)


class _StoreAdaptor(ContextManager['_StoreAdaptor'], Mapping[str, '_ConformerGroupAdaptor'], ABC):

    def __init__(self, *args, **kwargs) -> None:
        self._dummy_properties = kwargs.pop("dummy_properties", dict())

    @property
    def dummy_properties(self) -> Mapping[str, Any]:
        return self._dummy_properties.copy()

    @abstractmethod
    def transfer_location_to(self, other_store: '_StoreAdaptor') -> None: pass  # noqa E704

    @abstractmethod
    def validate_location(self) -> None: pass  # noqa E704

    @abstractmethod
    def make_empty(self, grouping: str) -> None: pass  # noqa E704

    @property
    @abstractmethod
    def location(self) -> StrPath: pass  # noqa E704

    @location.setter
    def location(self, value: StrPath) -> None: pass  # noqa E704

    @abstractmethod
    def delete_location(self) -> None: pass  # noqa E704

    @property
    @abstractmethod
    def mode(self) -> str: pass # noqa E704

    @property
    @abstractmethod
    def is_open(self) -> bool: pass # noqa E704

    @abstractmethod
    def close(self) -> '_StoreAdaptor': pass # noqa E704

    @abstractmethod
    def open(self, mode: str = 'r') -> '_StoreAdaptor': pass # noqa E704

    @property
    @abstractmethod
    def grouping(self) -> str: pass # noqa E704

    @property
    @abstractmethod
    def metadata(self) -> Mapping[str, str]:
        pass

    @abstractmethod
    def set_metadata(self, value: Mapping[str, str]) -> None:
        pass

    @abstractmethod
    def __delitem__(self, k: str) -> None: pass # noqa E704

    @abstractmethod
    def create_conformer_group(self, name: str) -> '_ConformerGroupAdaptor': pass # noqa E704

    @abstractmethod
    def update_cache(self, check_properties: bool = False, verbose: bool = True) -> Tuple['OrderedDict[str, int]', Set[str]]:
        pass
