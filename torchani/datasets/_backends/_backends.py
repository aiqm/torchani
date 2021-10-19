from pathlib import Path
from typing import ContextManager, Dict, Any, Type, cast

from .._annotations import StrPath
from .interface import _Store
from .h5py_impl import _H5PY_AVAILABLE, _H5Store, _H5TemporaryLocation
from .zarr_impl import _ZARR_AVAILABLE, _ZarrStore, _ZarrTemporaryLocation
from .pq_impl import _PQ_AVAILABLE, _PqStore, _PqTemporaryLocation

# This should probably be obtained directly from getattr
_BACKEND_AVAILABLE = {'h5py': _H5PY_AVAILABLE, 'zarr': _ZARR_AVAILABLE, 'pq': _PQ_AVAILABLE}
_CONCRETE_STORES = {'h5py': _H5Store, 'zarr': _ZarrStore, 'pq': _PqStore}
_CONCRETE_LOCATIONS = {'h5py': _H5TemporaryLocation, 'zarr': _ZarrTemporaryLocation, 'pq': _PqTemporaryLocation}
_SUFFIXES = {'h5py': '.h5', 'zarr': '.zarr', 'pq': '.pqdir'}


def _infer_backend(store_location: StrPath) -> str:
    suffix = Path(store_location).resolve().suffix
    for k, v in _SUFFIXES.items():
        if suffix == v:
            return k
    raise RuntimeError("Backend could not be infered from store location")


def StoreFactory(store_location: StrPath, backend: str = None, grouping: str = None,
                 create: bool = False, dummy_properties: Dict[str, Any] = None, use_cudf: bool = False) -> '_Store':
    backend = _infer_backend(store_location) if backend is None else backend
    dummy_properties = dict() if dummy_properties is None else dummy_properties

    if not _BACKEND_AVAILABLE.get(backend, False):
        raise ValueError(f'{backend} could not be found, please install it if supported.'
                         f' Supported backends are {set(_BACKEND_AVAILABLE.keys())}')
    cls: Type[_Store] = cast(Type[_Store], _CONCRETE_STORES[backend])
    if create:
        grouping = grouping if grouping is not None else "by_formula"
        store = cls.make_empty(store_location, grouping, dummy_properties=dummy_properties)
    else:
        if grouping is not None:
            raise ValueError("Can't specify a grouping for an already existing dataset")
        kwargs: Dict[str, Any] = {'dummy_properties': dummy_properties}
        if backend == 'pq':
            kwargs.update({'use_cudf': use_cudf})
        store = cls(store_location, **kwargs)
    setattr(store, 'backend', backend)  # Monkey patch
    return store


def TemporaryLocation(backend: str) -> 'ContextManager[StrPath]':
    if not _BACKEND_AVAILABLE.get(backend, False):
        raise ValueError(f'{backend} could not be found, please install it if supported.'
                         f' Supported backends are {set(_BACKEND_AVAILABLE.keys())}')
    return _CONCRETE_LOCATIONS[backend]()
