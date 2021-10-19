import tempfile
from pathlib import Path
from typing import ContextManager, Optional, Dict, Any
from collections import OrderedDict  # noqa F401

import numpy as np

from .._annotations import StrPath
from .interface import _Store, _ConformerGroup, _ConformerWrapper, _HierarchicalStoreWrapper

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


class _ZarrStore(_HierarchicalStoreWrapper["zarr.Group"]):
    def __init__(self, store_location: StrPath, dummy_properties: Dict[str, Any]):
        super().__init__(store_location, '.zarr', 'dir', dummy_properties=dummy_properties)
        self._mode: Optional[str] = None

    @classmethod
    def make_empty(cls, store_location: StrPath, grouping: str, **kwargs) -> '_Store':
        store = zarr.storage.DirectoryStore(store_location)
        with zarr.hierarchy.group(store=store, overwrite=True) as g:
            g.attrs['grouping'] = grouping
        return cls(store_location, **kwargs)

    def open(self, mode: str = 'r', only_meta: bool = False) -> '_Store':
        store = zarr.storage.DirectoryStore(self.location.root)
        self._store_obj = zarr.hierarchy.open_group(store, mode)
        setattr(self._store_obj, 'mode', mode)
        return self

    def __getitem__(self, name: str) -> '_ConformerGroup':
        return _ZarrConformerGroup(self._store[name], dummy_properties=self._dummy_properties)


class _ZarrConformerGroup(_ConformerWrapper["zarr.Group"]):
    def __init__(self, data: "zarr.Group", dummy_properties):
        super().__init__(data=data, dummy_properties=dummy_properties)

    def _append_to_property(self, p: str, v: np.ndarray) -> None:
        try:
            self._data[p].append(v, axis=0)
        except TypeError:
            self._data[p].append(v.astype(bytes), axis=0)
