from abc import ABC, abstractmethod
from typing import ContextManager, Mapping, Set, Tuple
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
        pass

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


class _StoreAdaptor(ContextManager['_StoreAdaptor'], Mapping[str, '_ConformerGroupAdaptor'], ABC):

    def __init__(self, *args, **kwargs) -> None:
        pass

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

    @abstractmethod
    def __delitem__(self, k: str) -> None: pass # noqa E704

    @abstractmethod
    def create_conformer_group(self, name: str) -> '_ConformerGroupAdaptor': pass # noqa E704

    @abstractmethod
    def update_cache(self, check_properties: bool = False, verbose: bool = True) -> Tuple['OrderedDict[str, int]', Set[str]]:
        pass
