import json
import shutil
from os import fspath
import tempfile
from pathlib import Path
from typing import Iterator, Set, Union, Tuple, Dict, Iterable, Any, Optional, List
from collections import OrderedDict

import numpy as np

from .._annotations import StrPath
from .interface import _Store, _StoreWrapper, _ConformerGroup, CacheHolder, _FileOrDirLocation
from .zarr_impl import _ZarrTemporaryLocation


try:
    import cudf
    _CUDF_AVAILABLE = True
    default_engine = cudf
except ImportError:
    _CUDF_AVAILABLE = False

try:
    import pandas
    _PANDAS_AVAILABLE = True
    if not _CUDF_AVAILABLE:
        default_engine = pandas
except ImportError:
    _PANDAS_AVAILABLE = False

_PQ_AVAILABLE = _PANDAS_AVAILABLE or _CUDF_AVAILABLE


def _to_numpy_pandas(series):
    return series.to_numpy()


def _to_dict_pandas(df, **kwargs):
    return df.to_dict(**kwargs)


def _to_numpy_cudf(series):
    return series.to_pandas().to_numpy()


def _to_dict_cudf(df, **kwargs):
    return df.to_pandas().to_dict(**kwargs)


class _PqLocation(_FileOrDirLocation):
    def __init__(self, root: StrPath):
        self._meta_location: Optional[Path] = None
        self._pq_location: Optional[Path] = None
        super().__init__(root, '.pqdir', 'dir')

    @property
    def meta(self) -> StrPath:
        if self._meta_location is not None:
            return self._meta_location
        else:
            raise RuntimeError("Location is not set")

    @property
    def pq(self) -> StrPath:
        if self._pq_location is not None:
            return self._pq_location
        else:
            raise RuntimeError("Location is not set")

    @property
    def root(self) -> StrPath:
        # mypy can not understand this, we are calling the getter from the superclass
        return super(__class__, __class__).root.fget(self)  # type: ignore

    @root.setter
    def root(self, value: StrPath) -> None:
        value = Path(value).resolve()
        if value.suffix == '':
            value = value.with_suffix(self._suffix)
        if value.suffix != self._suffix:
            raise ValueError(f"Incorrect location {value}")
        meta = self._meta_location
        pq = self._pq_location
        if meta is not None:
            meta.rename(meta.with_name(value.with_suffix('.json').name))
        if pq is not None:
            pq.rename(pq.with_name(value.with_suffix('.pq').name))

        if self._root_location is not None:
            # pathlib.rename() may fail if src and dst are in different filesystems
            shutil.move(fspath(self._root_location), fspath(value))
        self._root_location = Path(value).resolve()
        root = self._root_location
        self._meta_location = root / root.with_suffix('.json').name
        self._pq_location = root / root.with_suffix('.pq').name
        if not (self._pq_location.is_file() or self._meta_location.is_file()):
            raise FileNotFoundError(f"The store in {self._root_location} could not be found or is invalid")
        self._validate

    @root.deleter
    def root(self) -> None:
        # mypy can not understand this, we are calling the deleter from the superclass
        super(__class__, __class__).root.fdel(self)  # type: ignore
        self._meta_location = None
        self._pq_location = None


class DataFrameAdaptor:
    def __init__(self, df=None):
        self._df = df
        self.attrs = dict()
        self.mode: str = None
        self._is_dirty = False
        self._meta_is_dirty = False

    def __getattr__(self, k):
        return getattr(self._df, k)

    def __getitem__(self, k):
        return self._df[k]

    def __setitem__(self, k, v):
        self._df[k] = v


class _PqTemporaryLocation(_ZarrTemporaryLocation):
    def __init__(self) -> None:
        self._tmp_location = tempfile.TemporaryDirectory(suffix='.pqdir')


class _PqStore(_StoreWrapper[Union["pandas.DataFrame", "cudf.DataFrame"]]):
    def __init__(self, store_location: StrPath, use_cudf: bool = False, dummy_properties: Dict[str, Any] = None):
        super().__init__(dummy_properties=dummy_properties)
        self.location = _PqLocation(store_location)
        self._queued_appends: List[Union["pandas.DataFrame", "cudf.DataFrame"]] = []
        if use_cudf:
            self._engine = cudf
            self._to_dict = _to_dict_cudf
            self._to_numpy = _to_numpy_cudf
        else:
            self._engine = pandas
            self._to_dict = _to_dict_pandas
            self._to_numpy = _to_numpy_pandas

    # Avoid pickling modules
    def __getstate__(self):
        d = self.__dict__.copy()
        d['_engine'] = self._engine.__name__
        return d

    # Restore modules from names when unpickling
    def __setstate__(self, d):
        if d['_engine'] == 'pandas':
            import pandas  # noqa
            d['_engine'] == pandas
        elif d['_engine'] == 'cudf':
            import cudf  # noqa
            d['_engine'] == cudf
        else:
            raise RuntimeError("Incorrect _engine value")
        self.__dict__ = d

    def update_cache(self,
                     check_properties: bool = False,
                     verbose: bool = True) -> Tuple['OrderedDict[str, int]', Set[str]]:
        cache = CacheHolder()
        try:
            group_sizes_df = self._store['group'].value_counts().sort_index()
        except KeyError:
            return cache.group_sizes, cache.properties
        cache.group_sizes = OrderedDict(sorted([(k, v) for k, v in self._to_dict(group_sizes_df).items()]))
        cache.properties = set(self._store.columns.tolist()).difference({'group'})
        self._dummy_properties = {k: v for k, v in self._dummy_properties.items() if k not in cache.properties}
        return cache.group_sizes, cache.properties.union(self._dummy_properties)

    @classmethod
    def make_empty(cls, store_location: StrPath, grouping: str, **kwargs) -> '_Store':
        root = Path(store_location).resolve()
        root.mkdir(exist_ok=True)
        assert not list(root.iterdir()), "location is not empty"
        meta_location = root / root.with_suffix('.json').name
        pq_location = root / root.with_suffix('.pq').name
        default_engine.DataFrame().to_parquet(pq_location)
        with open(meta_location, 'x') as f:
            json.dump({'grouping': grouping}, f)
        return cls(store_location, **kwargs)

    # File-like
    def open(self, mode: str = 'r', only_meta: bool = False) -> '_Store':
        if not only_meta:
            self._store_obj = DataFrameAdaptor(self._engine.read_parquet(self.location.pq))
        else:
            self._store_obj = DataFrameAdaptor()
        with open(self.location.meta, mode) as f:
            meta = json.load(f)
        if 'extra_dims' not in meta.keys():
            meta['extra_dims'] = dict()
        if 'dtypes' not in meta.keys():
            meta['dtypes'] = dict()
        self._store_obj.attrs = meta
        # monkey patch
        self._store_obj.mode = mode
        return self

    def close(self) -> '_Store':
        if self._queued_appends:
            self.execute_queued_appends()
        if self._store._is_dirty:
            self._store.to_parquet(self.location.pq)
        if self._store._meta_is_dirty:
            with open(self.location.meta, 'w') as f:
                json.dump(self._store.attrs, f)
        self._store_obj = None
        return self

    # ContextManager
    def __exit__(self, *args, **kwargs) -> None:
        self.close()

    # Mapping
    def __getitem__(self, name: str) -> '_ConformerGroup':
        df_group = self._store[self._store['group'] == name]
        group = _PqConformerGroup(df_group, self._dummy_properties, self._store)
        # mypy does not understand monkey patching
        group._to_numpy = self._to_numpy  # type: ignore
        return group

    def __setitem__(self, name: str, conformers: '_ConformerGroup') -> None:
        num_conformers = conformers[next(iter(conformers.keys()))].shape[0]
        tmp_df = self._engine.DataFrame()
        tmp_df['group'] = self._engine.Series([name] * num_conformers)
        for k, v in conformers.items():
            if v.ndim == 1:
                tmp_df[k] = self._engine.Series(v)
            elif v.ndim == 2:
                tmp_df[k] = self._engine.Series(v.tolist())
            else:
                extra_dims = self._store.attrs['extra_dims'].get(k, None)
                if extra_dims is not None:
                    assert v.shape[2:] == tuple(extra_dims), "Bad dimensions in appended property"
                else:
                    self._store.attrs['extra_dims'][k] = v.shape[2:]
                tmp_df[k] = self._engine.Series(v.reshape(num_conformers, -1).tolist())
            dtype = self._store.attrs['dtypes'].get(k, None)
            if dtype is not None:
                assert np.dtype(v.dtype).name == dtype, "Bad dtype in appended property"
            else:
                self._store.attrs['dtypes'][k] = np.dtype(v.dtype).name
        self._queued_appends.append(tmp_df)

    def execute_queued_appends(self):
        meta = self._store_obj.attrs
        mode = self._store_obj.mode
        self._store_obj = DataFrameAdaptor(self._engine.concat([self._store._df] + self._queued_appends))
        self._store.attrs = meta
        self._store.mode = mode
        self._store._is_dirty = True
        self._store._meta_is_dirty = True
        self._queued_appends = []

    def __delitem__(self, name: str) -> None:
        # Instead of deleting we just reassign the store to everything that is
        # not the requested name here, since this dirties the dataset,
        # only this part will be written to disk on closing
        meta = self._store_obj.attrs
        mode = self._store_obj.mode
        meta_is_dirty = self._store_obj._meta_is_dirty
        self._store_obj = self._store[self._store['group'] != name]
        self._store.attrs = meta
        self._store.mode = mode
        self._store._meta_is_dirty = meta_is_dirty
        self._store._is_dirty = True

    def __len__(self) -> int:
        return len(self._store['group'].unique())

    def __iter__(self) -> Iterator[str]:
        keys = self._store['group'].unique().tolist()
        keys.sort()
        return iter(keys)

    def create_full_direct(self, dest_key, is_atomic, extra_dims, fill_value, dtype, num_conformers):
        if is_atomic:
            raise ValueError("creation of atomic properties not supported in parquet datasets")
        if extra_dims:
            extra_dims = (np.asarray(extra_dims).prod()[0],)
        new_property = np.full(shape=(num_conformers,) + extra_dims, fill_value=fill_value, dtype=dtype)
        self._store.attrs['dtypes'][dest_key] = np.dtype(dtype).name
        if len(extra_dims) > 1:
            self._store.attrs['extra_dims'][dest_key] = extra_dims[1:]
        self._store[dest_key] = self._engine.Series(new_property)
        self._store._meta_is_dirty = True
        self._store._is_dirty = True

    def rename_direct(self, old_new_dict: Dict[str, str]) -> None:
        self._store.rename(columns=old_new_dict, inplace=True)
        self._store._is_dirty = True

    def delete_direct(self, properties: Iterable[str]) -> None:
        self._store.drop(labels=list(properties), inplace=True, axis='columns')
        if self._store.columns.tolist() == ['group']:
            self._store.drop(labels=['group'], inplace=True, axis='columns')
        self._store._is_dirty = True


class _PqConformerGroup(_ConformerGroup):
    def __init__(self, group_obj, dummy_properties, store_pointer):
        super().__init__(dummy_properties=dummy_properties)
        self._group_obj = group_obj
        self._store_pointer = store_pointer

    # parquet groups are immutable, mutable operations are done directly in the
    # store
    def _is_resizable(self) -> bool:
        return False

    def _append_to_property(self, p: str, data: np.ndarray) -> None:
        raise ValueError("Not implemented for pq groups")

    def move(self, src: str, dest: str) -> None:
        raise ValueError("Not implemented for pq groups")

    def __delitem__(self, k: str) -> None:
        raise ValueError("Not implemented for pq groups")

    def __setitem__(self, p: str, v: np.ndarray) -> None:
        raise ValueError("Not implemented for pq groups")

    def _getitem_impl(self, p: str) -> np.ndarray:
        # mypy doesn't understand monkey patching
        property_ = np.stack(self._to_numpy(self._group_obj[p]))  # type: ignore
        extra_dims = self._store_pointer.attrs['extra_dims'].get(p, None)
        dtype = self._store_pointer.attrs['dtypes'].get(p, None)
        if extra_dims is not None:
            if property_.ndim == 1:
                property_ = property_.reshape(-1, *extra_dims)
            else:
                property_ = property_.reshape(property_.shape[0], -1, *extra_dims)
        return property_.astype(dtype)

    def _len_impl(self) -> int:
        return len(self._group_obj.columns) - 1

    def _iter_impl(self):
        for c in self._group_obj.columns:
            if c != 'group':
                yield c
