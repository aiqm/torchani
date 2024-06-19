import typing as tp
import json
from pathlib import Path
from collections import OrderedDict
from dataclasses import asdict

import numpy as np
from numpy.typing import NDArray

from torchani.annotations import StrPath, Grouping, Backend
from torchani.datasets.backends.interface import (
    Store,
    _ConformerGroup,
    Cache,
    RootKind,
    Metadata,
)

try:
    import pandas

    _PANDAS_AVAILABLE = True
    default_engine = pandas
except ImportError:
    _PANDAS_AVAILABLE = False

try:
    import cudf

    _CUDF_AVAILABLE = True
    default_engine = cudf
except ImportError:
    _CUDF_AVAILABLE = False


class _PandasStore(Store):
    root_kind: RootKind = "dir"
    suffix: str = ".pqdir"
    backend: Backend = "pandas"
    BACKEND_AVAILABLE: bool = _PANDAS_AVAILABLE

    def __init__(
        self,
        root: StrPath,
        dummy_properties: tp.Optional[tp.Dict[str, tp.Any]] = None,
        grouping: tp.Optional[Grouping] = None,
    ):
        super().__init__(root, dummy_properties, grouping)
        self._append_ops: tp.List[tp.Union["pandas.DataFrame", "cudf.DataFrame"]] = []
        self._engine = pandas
        self._data_is_dirty = False

    def setup(self, root: Path, mode: str) -> None:
        data = self._engine.read_parquet(self.parquet_path)
        self.set_data(data, mode)

    def setup_meta(self, root: Path, mode: str) -> None:
        with open(self.json_path, mode) as f:
            meta = Metadata(**json.load(f))
        self.set_meta(meta, mode)

    def teardown(self) -> None:
        self.execute_all_queued_append_ops()
        if self._data_is_dirty:
            self.data.to_parquet(self.parquet_path)
        self._data_is_dirty = False

    def teardown_meta(self) -> None:
        with open(self.json_path, "w") as f:
            json.dump(asdict(self.meta), f)

    @property
    def parquet_path(self) -> Path:
        return self.location.root / "data.pq"

    @property
    def json_path(self) -> Path:
        return self.location.root / "meta.json"

    @staticmethod
    def init_new(
        root: Path,
        grouping: Grouping,
    ) -> None:
        if any(root.iterdir()):
            raise RuntimeError("Root for empty parquet store must be empty")

        default_engine.DataFrame().to_parquet(root / "data.pq")
        with open(root / "meta.json", "x") as f:
            meta_kwargs = {
                "info": "",
                "units": dict(),
                "dtypes": dict(),
                "dims": dict(),
                "grouping": grouping,
            }
            json.dump(meta_kwargs, f)

    def update_cache(
        self, check_properties: bool = False, verbose: bool = True
    ) -> tp.Tuple[tp.OrderedDict[str, int], tp.Set[str]]:
        cache = Cache()
        try:
            group_sizes_df = self.data["group"].value_counts().sort_index()
        except KeyError:
            return cache.group_sizes, cache.properties
        if hasattr(group_sizes_df, "to_pandas"):
            group_sizes = group_sizes_df.to_pandas().to_dict()
        else:
            group_sizes = group_sizes_df.to_dict()
        cache.group_sizes = OrderedDict(
            sorted([(k, v) for k, v in group_sizes.items()])
        )
        cache.properties = set(self.data.columns.tolist()).difference({"group"})
        self._dummy_properties = {
            k: v for k, v in self._dummy_properties.items() if k not in cache.properties
        }
        return cache.group_sizes, cache.properties.union(self._dummy_properties)

    # When pickling, store names of modules instead of modules
    def __getstate__(self):
        self.execute_all_queued_append_ops()
        d = self.__dict__.copy()
        d["_engine"] = self._engine.__name__
        return d

    # When unpickling, restore modules from their names
    def __setstate__(self, d: tp.Dict[str, tp.Any]) -> None:
        if d["_engine"] == "pandas":
            import pandas  # noqa

            d["_engine"] == pandas
        elif d["_engine"] == "cudf":
            import cudf  # noqa

            d["_engine"] == cudf
        else:
            raise RuntimeError("Incorrect _engine value")
        self.__dict__ = d

    # Mapping
    def __getitem__(self, name: str) -> "_ConformerGroup":
        self.execute_all_queued_append_ops()
        df_group = self.data[self.data["group"] == name]
        group = _ParquetConformerGroup(df_group, self._dummy_properties, self)
        return group

    def __setitem__(self, name: str, conformers: "_ConformerGroup") -> None:
        # This is asynchronous, it is queued and executed only when the dataset
        # Is closed
        num_conformers = conformers[next(iter(conformers.keys()))].shape[0]
        tmp_df = self._engine.DataFrame()
        tmp_df["group"] = self._engine.Series([name] * num_conformers)
        for k, v in conformers.items():
            # Check dims
            if v.ndim == 1:
                tmp_df[k] = self._engine.Series(v)
            elif v.ndim == 2:
                tmp_df[k] = self._engine.Series(v.tolist())
            else:
                dims = self.meta.dims.get(k, None)
                if dims is not None:
                    dims = tuple(dims)
                    assert v.shape[2:] == dims, "Bad dims in appended property"
                else:
                    self.meta.dims[k] = v.shape[2:]
                tmp_df[k] = self._engine.Series(v.reshape(num_conformers, -1).tolist())

            # Check dtype
            dtype = self.meta.dtypes.get(k, None)
            if dtype is not None:
                assert np.dtype(v.dtype).name == dtype, "Bad dtype in appended property"
            else:
                self.meta.dtypes[k] = np.dtype(v.dtype).name
        self._append_ops.append(tmp_df)

    def execute_all_queued_append_ops(self) -> None:
        if not self._append_ops:
            return
        data, mode = self.get_data()
        data = self._engine.concat([data] + self._append_ops)
        self.set_data(data, mode)
        self._append_ops = []
        self._data_is_dirty = True

    def __delitem__(self, name: str) -> None:
        self.execute_all_queued_append_ops()

        # Reassign data to everything except the requested name, only that part
        # is persisted when closing the store.
        data, mode = self.get_data()
        data = data[data["group"] != name]
        self.set_data(data, mode)
        self._data_is_dirty = True

    def __len__(self) -> int:
        self.execute_all_queued_append_ops()
        return len(self.data["group"].unique())

    def __iter__(self) -> tp.Iterator[str]:
        self.execute_all_queued_append_ops()
        keys = self.data["group"].unique().tolist()
        keys.sort()
        return iter(keys)

    # TODO: Add these to all stores if possible?
    def create_full_direct(
        self, dest_key, is_atomic, extra_dims, fill_value, dtype, num_conformers
    ):
        self.execute_all_queued_append_ops()
        if is_atomic:
            raise ValueError(
                "creation of atomic properties not supported in parquet datasets"
            )
        if extra_dims:
            extra_dims = (np.asarray(extra_dims).prod()[0],)
        new_property = np.full(
            shape=(num_conformers,) + extra_dims, fill_value=fill_value, dtype=dtype
        )
        self.meta.dtypes[dest_key] = np.dtype(dtype).name
        if len(extra_dims) > 1:
            self.data.dims[dest_key] = extra_dims[1:]
        self.data[dest_key] = self._engine.Series(new_property)
        self._data_is_dirty = True

    def rename_direct(self, old_new_dict: tp.Dict[str, str]) -> None:
        self.execute_all_queued_append_ops()
        self.data.rename(columns=old_new_dict, inplace=True)
        self._data_is_dirty = True

    def delete_direct(self, properties: tp.Iterable[str]) -> None:
        self.execute_all_queued_append_ops()
        self.data.drop(labels=list(properties), inplace=True, axis="columns")
        # If the only thing left is "group", delete everything
        if self.data.columns.tolist() == ["group"]:
            self.data.drop(labels=["group"], inplace=True, axis="columns")
        self._data_is_dirty = True


class _CudfStore(_PandasStore):
    backend: Backend = "cudf"
    BACKEND_AVAILABLE: bool = _CUDF_AVAILABLE

    def __init__(
        self,
        root: StrPath,
        dummy_properties: tp.Optional[tp.Dict[str, tp.Any]] = None,
        grouping: tp.Optional[Grouping] = None,
    ):
        super().__init__(root, dummy_properties, grouping)
        self._engine = cudf


class _ParquetConformerGroup(_ConformerGroup):
    def __init__(self, group_obj, dummy_properties, store_ref: Store):
        super().__init__(dummy_properties=dummy_properties)
        self._group_obj = group_obj
        self._store_ref = store_ref

    # "dataframe groups" are not resizable, mutation is done directly on the dataframe
    def _is_resizable(self) -> bool:
        return False

    def _append_to_property(self, p: str, data: NDArray[tp.Any]) -> None:
        raise ValueError("Not implemented for pq groups")

    def move(self, src: str, dest: str) -> None:
        raise ValueError("Not implemented for pq groups")

    def __delitem__(self, k: str) -> None:
        raise ValueError("Not implemented for pq groups")

    def __setitem__(self, p: str, v: NDArray[tp.Any]) -> None:
        raise ValueError("Not implemented for pq groups")

    def _getitem_impl(self, p: str) -> NDArray[tp.Any]:
        # mypy doesn't understand monkey patching
        series = self._group_obj[p]
        _series: "pandas.Series" = (
            series.to_pandas() if hasattr(series, "to_pandas") else series
        )
        # TODO: Currently arrays are saved as 'object' which si probably not
        # efficient, so to_numpy() creates an array of arrays that need to be stacked
        _property = np.stack(_series.to_numpy())
        extra_dims = self._store_ref.meta.dims.get(p, None)
        dtype = self._store_ref.meta.dtypes.get(p, None)
        if extra_dims is not None:
            if _property.ndim == 1:
                _property = _property.reshape(-1, *extra_dims)
            else:
                _property = _property.reshape(_property.shape[0], -1, *extra_dims)
        return _property.astype(dtype)

    def _len_impl(self) -> int:
        return len(self._group_obj.columns) - 1

    def _iter_impl(self):
        for c in self._group_obj.columns:
            if c != "group":
                yield c
