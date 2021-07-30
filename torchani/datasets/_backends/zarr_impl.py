import shutil
import tempfile
from os import fspath
from pathlib import Path
from typing import ContextManager, Set, Tuple, Optional
from collections import OrderedDict

import numpy as np

from .._annotations import StrPath
from .interface import _StoreAdaptor, _ConformerGroupAdaptor, CacheHolder
from .h5py_impl import _H5StoreAdaptor, _H5ConformerGroupAdaptor


try:
    import zarr  # noqa
    _ZARR_AVAILABLE = True
except ImportError:
    _ZARR_AVAILABLE = False


class _ZarrTemporaryLocation(ContextManager[StrPath]):
    def __init__(self) -> None:
        self._tmp_location = tempfile.TemporaryDirectory(suffix='.zarr')

    def __enter__(self) -> str:
        return self._tmp_location.name

    def __exit__(self, *args) -> None:
        if Path(self._tmp_location.name).exists():  # check necessary for python 3.6
            self._tmp_location.cleanup()


# Backend Specific code starts here
class _ZarrStoreAdaptor(_H5StoreAdaptor):
    def __init__(self, store_location: StrPath):
        self.location = store_location
        self._store_obj = None
        self._mode: Optional[str] = None

    def validate_location(self) -> None:
        if not self._store_location.is_dir():
            raise FileNotFoundError(f"The store in {self._store_location} could not be found")

    def transfer_location_to(self, other_store: '_StoreAdaptor') -> None:
        self.delete_location()
        other_store.location = Path(self.location).with_suffix('')

    @property
    def location(self) -> StrPath:
        return self._store_location

    @location.setter
    def location(self, value: StrPath) -> None:
        value = Path(value).resolve()
        if value.suffix == '':
            value = value.with_suffix('.zarr')
        if value.suffix != '.zarr':
            raise ValueError(f"incorrect location {value}")
        # pathlib.rename() may fail if src and dst are in different mounts
        try:
            shutil.move(fspath(self.location), fspath(value))
        except AttributeError:
            pass
        self._store_location = value

    def delete_location(self) -> None:
        shutil.rmtree(self._store_location.as_posix())

    def make_empty(self, grouping: str) -> None:
        store = zarr.storage.DirectoryStore(self._store_location)
        with zarr.hierarchy.group(store=store, overwrite=True) as g:
            g.attrs['grouping'] = grouping

    def open(self, mode: str = 'r') -> '_StoreAdaptor':
        store = zarr.storage.DirectoryStore(self._store_location)
        self._store_obj = zarr.hierarchy.open_group(store, mode)
        self._mode = mode
        return self

    def close(self) -> '_StoreAdaptor':
        # Zarr Groups actually wrap a store, but DirectoryStore has no "close"
        # method Other stores may have a "close" method though
        try:
            self._store.store.close()
        except AttributeError:
            pass
        self._store_obj = None
        return self

    @property
    def _store(self) -> "zarr.Group":
        if self._store_obj is None:
            raise RuntimeError("Can't access store")
        return self._store_obj

    def update_cache(self,
                     check_properties: bool = False,
                     verbose: bool = True) -> Tuple['OrderedDict[str, int]', Set[str]]:
        cache = CacheHolder()
        for k, g in self._store.items():
            self._update_properties_cache(cache, g, check_properties)
            self._update_groups_cache(cache, g)
        if list(cache.group_sizes) != sorted(cache.group_sizes):
            raise RuntimeError("Groups were not iterated upon alphanumerically")
        return cache.group_sizes, cache.properties

    def _update_properties_cache(self, cache: CacheHolder, conformers: "zarr.Group", check_properties: bool = False) -> None:
        if not cache.properties:
            cache.properties = set(conformers.keys())
        elif check_properties and not set(conformers.keys()) == cache.properties:
            raise RuntimeError(f"Group {conformers.name} has bad keys, "
                               f"found {set(conformers.keys())}, but expected "
                               f"{cache.properties}")

    def _update_groups_cache(self, cache: CacheHolder, group: "zarr.Group") -> None:
        present_keys = {'coordinates', 'coord', 'energies'}.intersection(set(group.keys()))
        try:
            any_key = next(iter(present_keys))
        except StopIteration:
            raise RuntimeError('To infer conformer size need one of "coordinates", "coord", "energies"')
        cache.group_sizes.update({group.name[1:]: group[any_key].shape[0]})

    def _quick_standard_format_check(self) -> bool:
        return True

    @property
    def mode(self) -> str:
        if self._mode is None:
            raise RuntimeError("Can't access a closed store")
        return self._mode

    @property
    def grouping(self) -> str:
        g = self._store.attrs['grouping']
        assert isinstance(g, str)
        return g

    def __getitem__(self, name: str) -> '_ConformerGroupAdaptor':
        return _ZarrConformerGroupAdaptor(self._store[name])


class _ZarrConformerGroupAdaptor(_H5ConformerGroupAdaptor):
    def __init__(self, group_obj: "zarr.Group"):
        self._group_obj = group_obj

    @property
    def is_resizable(self) -> bool:
        return True

    def _append_property_with_data(self, p: str, data: np.ndarray) -> None:
        try:
            self._group_obj[p].append(data, axis=0)
        except TypeError:
            self._group_obj[p].append(data.astype(bytes), axis=0)

    def _create_property_with_data(self, p: str, data: np.ndarray) -> None:
        try:
            self._group_obj.create_dataset(name=p, data=data)
        except TypeError:
            self._group_obj.create_dataset(name=p, data=data.astype(bytes))
