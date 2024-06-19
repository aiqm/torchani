import typing as tp
from pathlib import Path

from torchani.annotations import StrPath, Grouping, Backend
from torchani.datasets.backends.interface import Store
from torchani.datasets.backends.hdf5_impl import _HDF5Store
from torchani.datasets.backends.zarr_impl import _ZarrStore
from torchani.datasets.backends.parquet_impl import _PandasStore, _CudfStore


_STORE_TYPE: tp.Dict[Backend, tp.Type[Store]] = {
    "hdf5": _HDF5Store,
    "zarr": _ZarrStore,
    "pandas": _PandasStore,
    "cudf": _CudfStore,
}

_SUFFIXES: tp.Dict[str, Backend] = {".h5": "hdf5", ".zarr": "zarr", ".pqdir": "pandas"}


def create_store(
    # root can be the string "tmp" to create a temporary store
    root: StrPath,
    backend: tp.Optional[Backend] = None,
    grouping: tp.Optional[Grouping] = None,
    dummy_properties: tp.Optional[tp.Dict[str, tp.Any]] = None,
) -> Store:
    if backend is None:
        try:
            backend = _SUFFIXES[Path(root).resolve().suffix]
        except KeyError:
            raise RuntimeError("Can't infer backend from suffix") from None

    if root == "tmp":
        return _STORE_TYPE[backend].make_tmp(dummy_properties, grouping)
    if not Path(root).exists():
        return _STORE_TYPE[backend].make_new(root, dummy_properties, grouping)
    return _STORE_TYPE[backend](root, dummy_properties, grouping)
