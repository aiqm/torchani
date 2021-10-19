import shutil
from itertools import chain
from os import fspath
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (ContextManager, MutableMapping, Set, Tuple, Optional,
                    Generic, TypeVar, Iterator, cast, Mapping, Any, Dict)
from collections import OrderedDict

import numpy as np

from .._annotations import NumpyConformers, StrPath


# Keeps track of variables that must be updated each time the datasets get
# modified or the first time they are read from disk
class CacheHolder:
    group_sizes: 'OrderedDict[str, int]'
    properties: Set[str]

    def __init__(self) -> None:
        self.group_sizes = OrderedDict()
        self.properties = set()


class NamedMapping(Mapping):
    name: str


_MutMapSubtype = TypeVar('_MutMapSubtype', bound=MutableMapping[str, np.ndarray])

# _ConformerGroup and _Store are abstract classes from which all backends
# should inherit in order to correctly interact with ANIDataset. Adding
# support for a new backend can be done just by coding these classes and
# adding the support for the backend inside interface.py


# This is kind of like a dict, but with the extra functionality that you can
# directly "append" to it, and rename its keys, it can also create dummy
# properties on the fly.
class _ConformerGroup(MutableMapping[str, np.ndarray], ABC):
    def __init__(self, *args, dummy_properties: Dict[str, Any] = None, **kwargs) -> None:
        self._dummy_properties = dict() if dummy_properties is None else dummy_properties

    def _is_resizable(self) -> bool:
        return True

    def append_conformers(self, conformers: NumpyConformers) -> None:
        if self._is_resizable():
            # We discard all dummy properties that we try to append if they are equal
            # to the ones already present, but if they are not equal we raise an error.
            # The responsibility of materializing the dummy properties before doing the
            # append is the caller's
            for p, v in conformers.items():
                self._append_to_property(p, v)
        else:
            raise ValueError("Can't append conformers, conformer group is not resizable")

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

    # creates a dummy property on the fly
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

    def __len__(self) -> int:
        return self._len_impl() + len(self._dummy_properties)

    def __iter__(self):
        yield from chain(self._iter_impl(), self._dummy_properties.keys())

    @abstractmethod
    def _append_to_property(self, p: str, v: np.ndarray) -> None:
        pass

    @abstractmethod
    def _getitem_impl(self, p: str) -> np.ndarray:
        pass

    @abstractmethod
    def _iter_impl(self, p: str):
        pass

    @abstractmethod
    def _len_impl(self) -> int:
        pass

    @abstractmethod
    def move(self, src_p: str, dest_p: str) -> None:
        pass


class _ConformerWrapper(_ConformerGroup, Generic[_MutMapSubtype]):
    def __init__(self, data: _MutMapSubtype, **kwargs) -> None:
        super().__init__(**kwargs)
        self._data = data

    def __setitem__(self, p: str, v: np.ndarray) -> None:
        self._data[p] = v

    def __delitem__(self, p: str) -> None:
        del self._data[p]

    def _getitem_impl(self, p: str) -> np.ndarray:
        array = self._data[p][()]
        assert isinstance(array, np.ndarray)
        return array

    def _len_impl(self) -> int:
        return len(self._data)

    def _iter_impl(self):
        yield from self._data.keys()

    def move(self, src_p: str, dest_p: str) -> None:
        self._data[dest_p] = self._data.pop(src_p)

    def _append_to_property(self, p: str, v: np.ndarray) -> None:
        self._data[p] = np.append(self._data[p], v, axis=0)


class _LocationManager(ABC):
    @property
    def root(self) -> StrPath:
        pass

    @root.setter
    def root(self, value: StrPath) -> None:
        pass

    @root.deleter
    def root(self) -> None:
        pass

    def transfer_to(self, other_store: '_Store') -> None:
        root = Path(self.root).with_suffix('')
        del self.root
        other_store.location.root = root


# Base location manager for datasets that use either directories or files as
# locations
class _FileOrDirLocation(_LocationManager):
    def __init__(self, root: StrPath, suffix: str = '', kind: str = 'file'):
        if kind not in ['file', 'dir']:
            raise ValueError("Kind must be one of 'file' or 'dir'")
        self._kind = kind
        self._suffix = suffix
        self._root_location: Optional[Path] = None
        self.root = root

    @property
    def root(self) -> StrPath:
        root = self._root_location
        if root is None:
            raise ValueError("Location is invalid")
        return root

    @root.setter
    def root(self, value: StrPath) -> None:
        value = Path(value).resolve()
        if value.suffix == '':
            value = value.with_suffix(self._suffix)
        if value.suffix != self._suffix:
            raise ValueError(f"Incorrect location {value}")
        if self._root_location is not None:
            # pathlib.rename() may fail if src and dst are in different filesystems
            shutil.move(fspath(self._root_location), fspath(value))
        self._root_location = Path(value).resolve()
        self._validate()

    @root.deleter
    def root(self) -> None:
        if self._root_location is not None:
            if self._kind == 'file':
                self._root_location.unlink()
            else:
                shutil.rmtree(self._root_location)
        self._root_location = None

    def _validate(self) -> None:
        root = Path(self.root)
        _kind = self._kind
        if (_kind == 'dir' and not root.is_dir()
           or _kind == 'file' and not root.is_file()):
            raise FileNotFoundError(f"The store in {root} could not be found")


# mypy expects a Protocol here, which specifies that _T must
# support Mapping and ContextManager methods, and also 'close' and 'create_group'
# and have 'mode' and 'attr' attributes
# this is similar to C++20 concepts and it is currently very verbose, so we avoid it
_T = TypeVar('_T', bound=Any)


# A store that wraps another store class (e.g. Zarr, Exedir, HDF5, DataFrame)
# Wrapped store must have a "mode" and "attr" attributes, it may implement close()
# __exit__, __enter__, , __delitem__
class _StoreWrapper(ContextManager['_Store'], MutableMapping[str, '_ConformerGroup'], ABC, Generic[_T]):
    location: Any

    def __init__(self, *args, dummy_properties: Dict[str, Any] = None, **kwargs):
        self._dummy_properties = dict() if dummy_properties is None else dummy_properties
        self._store_obj: Any = None

    @property
    def dummy_properties(self) -> Dict[str, Any]:
        return self._dummy_properties.copy()

    @classmethod
    @abstractmethod
    def make_empty(cls, store_location: StrPath, grouping: str, **kwargs) -> '_Store':
        pass

    @abstractmethod
    def update_cache(self,
                     check_properties: bool = False,
                     verbose: bool = True) -> Tuple['OrderedDict[str, int]', Set[str]]:
        pass

    @property
    def _store(self) -> _T:
        if self._store_obj is None:
            raise RuntimeError("Can't access store")
        return self._store_obj

    @abstractmethod
    def open(self, mode: str = 'r', only_meta: bool = False) -> '_Store':
        pass

    def close(self) -> '_Store':
        try:
            self._store.close()
        except AttributeError:
            pass
        self._store_obj = None
        return self

    @property
    def is_open(self) -> bool:
        try:
            self._store
        except RuntimeError:
            return False
        return True

    @property
    def mode(self) -> str:
        return cast(str, self._store.mode)

    def __enter__(self) -> '_Store':
        try:
            self._store.__enter__()
        except AttributeError:
            pass
        return self

    @property
    def grouping(self) -> str:
        # This detects Roman's formatting style which doesn't have a
        # 'grouping' key but is still grouped by num atoms.
        try:
            self._store.attrs['readme']
            return 'by_num_atoms'
        except (KeyError, OSError):
            pass
        try:
            g = self._store.attrs['grouping']
            return cast(str, g)
        except (KeyError, OSError):
            return 'legacy'

    @property
    def metadata(self) -> Mapping[str, str]:
        try:
            meta = {name: attr for name, attr in self._store.attrs.items() if name not in ['grouping', 'dtypes', 'extra_dims']}
        except Exception:
            meta = dict()
        return meta

    def set_metadata(self, value: Mapping[str, str]) -> None:
        if 'grouping' in value.keys():
            raise ValueError('Grouping is not a valid metadata key')
        for k, v in value.items():
            self._store.attrs[k] = v
        if hasattr(self._store, "_meta_is_dirty"):
            self._store._meta_is_dirty = True

    def __exit__(self, *args) -> None:
        try:
            self._store.__exit__(*args)
        except AttributeError:
            pass
        self._store_obj = None


# alias for convenience
_Store = _StoreWrapper


# A store that wraps another hierarchical store (e.g. Zarr, Exedir, HDF5)
# Wrapped store must implement:
# __exit__, __enter__, create_group, __len__, __iter__ -> Iterator[str], __delitem__
# and have a "mode" and "attr" attributes
class _HierarchicalStoreWrapper(_StoreWrapper[_T]):
    def __init__(self, store_location: StrPath, suffix='', kind='', dummy_properties: Dict[str, Any] = None):
        super().__init__(dummy_properties=dummy_properties)
        self.location = _FileOrDirLocation(store_location, suffix, kind)

    def update_cache(self,
                     check_properties: bool = False,
                     verbose: bool = True) -> Tuple['OrderedDict[str, int]', Set[str]]:
        cache = CacheHolder()
        for k, g in self._store.items():
            self._update_properties_cache(cache, g, check_properties)
            self._update_groups_cache(cache, g)
        if list(cache.group_sizes) != sorted(cache.group_sizes):
            raise RuntimeError("Groups were not iterated upon alphanumerically")
        self._dummy_properties = {k: v for k, v in self._dummy_properties.items() if k not in cache.properties}
        return cache.group_sizes, cache.properties.union(self._dummy_properties)

    def _update_properties_cache(self,
                                 cache: CacheHolder,
                                 conformers: NamedMapping,
                                 check_properties: bool = False) -> None:
        if not cache.properties:
            cache.properties = set(conformers.keys())
        elif check_properties and not set(conformers.keys()) == cache.properties:
            raise RuntimeError(f"Group {conformers.name} has bad keys, "
                               f"found {set(conformers.keys())}, but expected "
                               f"{cache.properties}")

    # updates "group_sizes" which holds the batch dimension (number of
    # molecules) of all groups in the dataset.
    def _update_groups_cache(self, cache: CacheHolder, group: NamedMapping) -> None:
        present_keys = {'coordinates', 'coord', 'energies'}.intersection(set(group.keys()))
        try:
            any_key = next(iter(present_keys))
        except StopIteration:
            raise RuntimeError('To infer conformer size need one of "coordinates", "coord", "energies"')
        cache.group_sizes.update({group.name[1:]: group[any_key].shape[0]})

    def __delitem__(self, k: str) -> None:
        del self._store[k]

    def __setitem__(self, name: str, conformers: '_ConformerGroup') -> None:
        self._store.create_group(name)
        self[name].update(conformers)

    def __len__(self) -> int:
        return len(self._store)

    def __iter__(self) -> Iterator[str]:
        return iter(self._store)
