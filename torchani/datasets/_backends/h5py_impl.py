import warnings
from uuid import uuid4
import tempfile
from pathlib import Path
from functools import partial
from typing import ContextManager, Any, Set, Union, Tuple, Dict
from collections import OrderedDict  # noqa F401

import numpy as np

from .._annotations import StrPath
from ...utils import tqdm
from .interface import _Store, _ConformerGroup, _ConformerWrapper, CacheHolder, _HierarchicalStoreWrapper


try:
    import h5py
    _H5PY_AVAILABLE = True
except ImportError:
    warnings.warn('Currently the only supported backend for ANIDataset is h5py,'
                  ' very limited options are available otherwise. Installing'
                  ' h5py (pip install h5py or conda install h5py) is'
                  ' recommended if you want to use the torchani.datasets'
                  ' module')
    _H5PY_AVAILABLE = False


class _H5TemporaryLocation(ContextManager[StrPath]):
    def __init__(self) -> None:
        self._tmp_location = tempfile.TemporaryDirectory()
        self._tmp_filename = Path(self._tmp_location.name).resolve() / f'{uuid4()}.h5'

    def __enter__(self) -> str:
        return self._tmp_filename.as_posix()

    def __exit__(self, *args) -> None:
        self._tmp_location.cleanup()


class _H5Store(_HierarchicalStoreWrapper["h5py.File"]):
    def __init__(self, store_location: StrPath, dummy_properties: Dict[str, Any]):
        super().__init__(store_location, '.h5', 'file', dummy_properties=dummy_properties)
        self._has_standard_format = True
        self._made_quick_check = False

    @classmethod
    def make_empty(cls, store_location: StrPath, grouping: str, **kwargs) -> '_Store':
        with h5py.File(store_location, 'x') as f:
            f.attrs['grouping'] = grouping
        obj = cls(store_location, **kwargs)
        obj._has_standard_format = True
        return obj

    def open(self, mode: str = 'r', only_meta: bool = False) -> '_Store':
        self._store_obj = h5py.File(self.location.root, mode)
        return self

    def update_cache(self,
                     check_properties: bool = False,
                     verbose: bool = True) -> Tuple['OrderedDict[str, int]', Set[str]]:
        cache = CacheHolder()
        # If the dataset has some semblance of standarization (it is a tree with depth
        # 1, where all groups are directly joined to the root) then it is much faster
        # to traverse the dataset. In any case after the first recursion if this
        # structure is detected the flag is set internally so we never do the recursion
        # again. This speeds up cache updates and lookup x30
        if self.grouping == 'legacy' and not self._made_quick_check:
            self._has_standard_format = self._quick_standard_format_check()
            self._made_quick_check = True

        if self._has_standard_format:
            for k, g in self._store.items():
                if g.name in ['/_created', '/_meta']:
                    continue
                self._update_properties_cache(cache, g, check_properties)
                self._update_groups_cache(cache, g)
        else:
            self._has_standard_format = self._update_cache_nonstandard(cache, check_properties, verbose)
        # By default iteration of HDF5 should be alphanumeric in which case
        # sorting should not be necessary, this internal check ensures the
        # groups were not created with 'track_order=True', and that the visitor
        # function worked properly.
        if list(cache.group_sizes) != sorted(cache.group_sizes):
            raise RuntimeError("Groups were not iterated upon alphanumerically")
        # we get rid of dummy properties if they are already in the dataset
        self._dummy_properties = {k: v for k, v in self._dummy_properties.items() if k not in cache.properties}
        return cache.group_sizes, cache.properties.union(self._dummy_properties)

    def _update_cache_nonstandard(self, cache: CacheHolder, check_properties: bool, verbose: bool) -> bool:
        def visitor_fn(name: str,
                       object_: Union["h5py.Dataset", "h5py.Group"],
                       store: '_H5Store',
                       cache: CacheHolder,
                       check_properties: bool,
                       pbar: Any) -> None:
            pbar.update()
            # We make sure the node is a Dataset, and we avoid Datasets
            # called _meta or _created since if present these store units
            # or other metadata. We also check if we already visited this
            # group via one of its children.
            if not isinstance(object_, h5py.Dataset) or\
                   object_.name in ['/_created', '/_meta'] or\
                   object_.parent.name in cache.group_sizes.keys():
                return
            g = object_.parent
            # Check for format correctness
            for v in g.values():
                if isinstance(v, h5py.Group):
                    raise RuntimeError(f"Invalid dataset format, there shouldn't be "
                                       "Groups inside Groups that have Datasets, "
                                       f"but {g.name}, parent of the dataset "
                                       f"{object_.name}, has group {v.name} as a "
                                       "child")
            store._update_properties_cache(cache, g, check_properties)
            store._update_groups_cache(cache, g)

        with tqdm(desc='Verifying format correctness', disable=not verbose) as pbar:
            self._store.visititems(partial(visitor_fn,
                                               store=self,
                                               cache=cache,
                                               pbar=pbar,
                                               check_properties=check_properties))
        # If the visitor function succeeded and this condition is met the
        # dataset must be in standard format
        has_standard_format = not any('/' in k[1:] for k in cache.group_sizes.keys())
        return has_standard_format

    # Check if the raw hdf5 file is one of a number of known files that can be assumed
    # to have standard format.
    def _quick_standard_format_check(self) -> bool:
        # This check detects the "ani-release" files which have this property
        try:
            key = next(iter(self._store.keys()))
            self._store[key]['hf_dz.energy']
            return True
        except Exception:
            pass
        # This check tests for the '/_created' which is present in "old HTRQ style"
        try:
            self._store['/_created']
            return True
        except KeyError:
            return False

    def __getitem__(self, name: str) -> '_ConformerGroup':
        return _H5ConformerGroup(self._store[name], dummy_properties=self._dummy_properties)


class _H5ConformerGroup(_ConformerWrapper["h5py.Group"]):
    def __init__(self, data: h5py.Group, **kwargs):
        super().__init__(data=data, **kwargs)

    def _is_resizable(self) -> bool:
        return all(ds.maxshape[0] is None for ds in self._data.values())

    def _append_to_property(self, p: str, v: np.ndarray) -> None:
        h5_dataset = self._data[p]
        h5_dataset.resize(h5_dataset.shape[0] + v.shape[0], axis=0)
        try:
            h5_dataset[-v.shape[0]:] = v
        except TypeError:
            h5_dataset[-v.shape[0]:] = v.astype(bytes)

    def __setitem__(self, p: str, data: np.ndarray) -> None:
        # This correctly handles strings and make the first axis resizable
        maxshape = (None,) + data.shape[1:]
        try:
            self._data.create_dataset(name=p, data=data, maxshape=maxshape)
        except TypeError:
            self._data.create_dataset(name=p, data=data.astype(bytes), maxshape=maxshape)
