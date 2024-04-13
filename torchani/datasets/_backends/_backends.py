import typing as tp
from pathlib import Path

from torchani.datasets._annotations import StrPath
from torchani.datasets._backends.interface import _StoreWrapper
from torchani.datasets._backends.h5py_impl import _H5PY_AVAILABLE, _H5Store, _H5TemporaryLocation
from torchani.datasets._backends.zarr_impl import _ZARR_AVAILABLE, _ZarrStore, _ZarrTemporaryLocation
from torchani.datasets._backends.pq_impl import _PQ_AVAILABLE, _PqStore, _PqTemporaryLocation

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


def StoreFactory(store_location: StrPath, backend: tp.Optional[str] = None, grouping: tp.Optional[str] = None,
                 dummy_properties: tp.Optional[tp.Dict[str, tp.Any]] = None, use_cudf: bool = False, _force_overwrite: bool = False) -> '_StoreWrapper':
    backend = _infer_backend(store_location) if backend is None else backend
    dummy_properties = dict() if dummy_properties is None else dummy_properties
    kwargs: tp.Dict[str, tp.Any] = {'dummy_properties': dummy_properties}
    if backend == 'pq':
        kwargs.update({'use_cudf': use_cudf})

    if not _BACKEND_AVAILABLE.get(backend, False):
        raise ValueError(f'{backend} could not be found, please install it if supported.'
                         f' Supported backends are {set(_BACKEND_AVAILABLE.keys())}')
    cls: tp.Type[_StoreWrapper] = tp.cast(tp.Type[_StoreWrapper], _CONCRETE_STORES[backend])
    if not Path(store_location).exists() or _force_overwrite:
        if grouping is not None:
            kwargs.update({"grouping": grouping})
        store = cls.make_empty(store_location, **kwargs)
    else:
        if grouping is not None:
            raise ValueError("Can't specify a grouping for an already existing dataset")
        store = cls(store_location, **kwargs)
    setattr(store, 'backend', backend)  # Monkey patch
    return store


def TemporaryLocation(backend: str) -> tp.ContextManager[StrPath]:
    if not _BACKEND_AVAILABLE.get(backend, False):
        raise ValueError(f'{backend} could not be found, please install it if supported.'
                         f' Supported backends are {set(_BACKEND_AVAILABLE.keys())}')
    return _CONCRETE_LOCATIONS[backend]()  # type: ignore
