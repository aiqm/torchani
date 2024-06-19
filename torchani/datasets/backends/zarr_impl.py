import typing as tp
from pathlib import Path

from numpy.typing import NDArray

from torchani.annotations import Grouping, Backend
from torchani.datasets.backends.interface import (
    Metadata,
    RootKind,
    _ConformerGroup,
    _ConformerWrapper,
    _HierarchicalStore,
)

try:
    import zarr

    _ZARR_AVAILABLE = True
except ImportError:
    _ZARR_AVAILABLE = False


class _ZarrStore(_HierarchicalStore):
    root_kind: RootKind = "dir"
    suffix: str = ".zarr"
    backend: Backend = "zarr"
    BACKEND_AVAILABLE: bool = _ZARR_AVAILABLE

    @staticmethod
    def init_new(root: Path, grouping: Grouping) -> None:
        zarr_dir_style_data = zarr.storage.DirectoryStore(str(root))
        with zarr.hierarchy.group(store=zarr_dir_style_data, overwrite=True) as g:
            g.attrs["grouping"] = grouping

    def setup(self, root: Path, mode: str) -> None:
        zarr_dir_style_data = zarr.storage.DirectoryStore(root)
        file = zarr.hierarchy.open_group(zarr_dir_style_data, mode)
        meta = Metadata(
            grouping=file.attrs["grouping"],
            dims=dict(),
            dtypes=dict(),
            units=dict(),
        )
        self.set_data(file, mode)
        self.set_meta(meta, mode)

    # Zarr DirectoryStore has no cleanup logic
    def teardown(self) -> None:
        pass

    def __getitem__(self, name: str) -> "_ConformerGroup":
        return _ZarrConformerGroup(
            self.data[name], dummy_properties=self._dummy_properties
        )


class _ZarrConformerGroup(_ConformerWrapper["zarr.Group"]):
    def __init__(self, data: "zarr.Group", dummy_properties):
        super().__init__(data=data, dummy_properties=dummy_properties)

    def _append_to_property(self, p: str, v: NDArray[tp.Any]) -> None:
        try:
            self._data[p].append(v, axis=0)
        except TypeError:
            self._data[p].append(v.astype(bytes), axis=0)
