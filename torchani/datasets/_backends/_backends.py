from pathlib import Path
from typing import ContextManager

from .._annotations import StrPath
from .interface import _StoreAdaptor
from .h5py_impl import _H5PY_AVAILABLE, _H5StoreAdaptor, _H5TemporaryLocation
from .zarr_impl import _ZARR_AVAILABLE, _ZarrStoreAdaptor, _ZarrTemporaryLocation


def infer_backend(store_location: StrPath) -> str:
    suffix = Path(store_location).resolve().suffix
    if suffix == '.h5':
        return 'h5py'
    elif suffix == '.zarr':
        return 'zarr'
    else:
        raise RuntimeError("Backend could not be infered from store location")


def StoreAdaptorFactory(store_location: StrPath, backend: str) -> '_StoreAdaptor':
    if backend == 'h5py':
        if not _H5PY_AVAILABLE:
            raise ValueError('h5py backend was specified but h5py could not be found, please install h5py')
        return _H5StoreAdaptor(store_location)
    elif backend == 'zarr':
        if not _ZARR_AVAILABLE:
            raise ValueError('zarr backend was specified but zarr could not be found, please install zarr')
        return _ZarrStoreAdaptor(store_location)
    else:
        raise RuntimeError(f"Bad backend {backend}")


def TemporaryLocation(backend: str) -> 'ContextManager[StrPath]':
    if backend == 'h5py':
        return _H5TemporaryLocation()
    elif backend == 'zarr':
        return _ZarrTemporaryLocation()
    else:
        raise ValueError(f"Bad backend {backend}")
