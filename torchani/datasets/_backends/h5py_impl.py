import warnings
from uuid import uuid4
import shutil
import tempfile
from os import fspath
from pathlib import Path
from functools import partial
from typing import ContextManager, Iterator, Any, Set, Union, Tuple, Mapping
from collections import OrderedDict

import numpy as np

from .._annotations import StrPath
from ...utils import tqdm
from .interface import _StoreAdaptor, _ConformerGroupAdaptor, CacheHolder


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


class _H5StoreAdaptor(_StoreAdaptor):
    def __init__(self, store_location: StrPath):
        self.location = store_location
        self._store_obj = None
        self._has_standard_format = False
        self._made_quick_check = False

    def validate_location(self) -> None:
        if not self._store_location.is_file():
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
            value = value.with_suffix('.h5')
        if value.suffix != '.h5':
            raise ValueError(f"Incorrect location {value}")
        # pathlib.rename() may fail if src and dst are in different mounts
        try:
            shutil.move(fspath(self.location), fspath(value))
        except AttributeError:
            pass
        self._store_location = value

    def delete_location(self) -> None:
        self._store_location.unlink()

    def make_empty(self, grouping: str) -> None:
        self._has_standard_format = True
        with h5py.File(self._store_location, 'x') as f:
            f.attrs['grouping'] = grouping

    def open(self, mode: str = 'r') -> '_StoreAdaptor':
        self._store_obj = h5py.File(self._store_location, mode)
        return self

    def close(self) -> '_StoreAdaptor':
        self._store.close()
        self._store_obj = None
        return self

    @property
    def _store(self) -> h5py.File:
        if self._store_obj is None:
            raise RuntimeError("Can't access store")
        return self._store_obj

    @property
    def is_open(self) -> bool:
        try:
            self._store
        except RuntimeError:
            return False
        return True

    def __enter__(self) -> '_StoreAdaptor':
        self._store.__enter__()
        return self

    def __exit__(self, *args, **kwargs) -> None:
        self._store.__exit__(*args, **kwargs)
        self._store_obj = None

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
        return cache.group_sizes, cache.properties

    def _update_cache_nonstandard(self, cache: CacheHolder, check_properties: bool, verbose: bool) -> bool:
        def visitor_fn(name: str,
                       object_: Union[h5py.Dataset, h5py.Group],
                       store: '_H5StoreAdaptor',
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

    def _update_properties_cache(self, cache: CacheHolder, conformers: h5py.Group, check_properties: bool = False) -> None:
        if not cache.properties:
            cache.properties = set(conformers.keys())
        elif check_properties and not set(conformers.keys()) == cache.properties:
            raise RuntimeError(f"Group {conformers.name} has bad keys, "
                               f"found {set(conformers.keys())}, but expected "
                               f"{cache.properties}")

    # updates "group_sizes" which holds the batch dimension (number of
    # molecules) of all groups in the dataset.
    def _update_groups_cache(self, cache: CacheHolder, group: h5py.Group) -> None:
        present_keys = {'coordinates', 'coord', 'energies'}.intersection(set(group.keys()))
        try:
            any_key = next(iter(present_keys))
        except StopIteration:
            raise RuntimeError('To infer conformer size need one of "coordinates", "coord", "energies"')
        cache.group_sizes.update({group.name[1:]: group[any_key].shape[0]})

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

    @property
    def mode(self) -> str:
        mode = self._store.mode
        assert isinstance(mode, str)
        return mode

    @property
    def metadata(self) -> Mapping[str, str]:
        try:
            meta = {name: attr for name, attr in self._store.attrs.items() if name != 'grouping'}
        except Exception:
            meta = dict()
        return meta

    def set_metadata(self, value: Mapping[str, str]) -> None:
        if 'grouping' in value.keys():
            raise ValueError('Grouping is not a valid metadata key')
        for k, v in value.items():
            self._store.attrs[k] = v

    @property
    def grouping(self) -> str:
        # This detects Roman's formatting style which doesn't have a
        # 'grouping' key but is still grouped by num atoms.
        try:
            self._store.attrs['readme']
            return 'by_num_atoms'
        except (KeyError, OSError):
            pass

        try:
            g = self._store.attrs['grouping']
            assert isinstance(g, str)
            return g
        except (KeyError, OSError):
            return 'legacy'

    def __delitem__(self, k: str) -> None:
        del self._store[k]

    def create_conformer_group(self, name: str) -> '_ConformerGroupAdaptor':
        self._store.create_group(name)
        return self[name]

    def __getitem__(self, name: str) -> '_ConformerGroupAdaptor':
        return _H5ConformerGroupAdaptor(self._store[name])

    def __len__(self) -> int:
        return len(self._store)

    def __iter__(self) -> Iterator[str]:
        return iter(self._store)


class _H5ConformerGroupAdaptor(_ConformerGroupAdaptor):
    def __init__(self, group_obj: h5py.Group):
        self._group_obj = group_obj

    @property
    def is_resizable(self) -> bool:
        return all(ds.maxshape[0] is None for ds in self._group_obj.values())

    def _append_property_with_data(self, p: str, data: np.ndarray) -> None:
        h5_dataset = self._group_obj[p]
        h5_dataset.resize(h5_dataset.shape[0] + data.shape[0], axis=0)
        try:
            h5_dataset[-data.shape[0]:] = data
        except TypeError:
            h5_dataset[-data.shape[0]:] = data.astype(bytes)

    def _create_property_with_data(self, p: str, data: np.ndarray) -> None:
        # This correctly handles strings and make the first axis resizable
        maxshape = (None,) + data.shape[1:]
        try:
            self._group_obj.create_dataset(name=p, data=data, maxshape=maxshape)
        except TypeError:
            self._group_obj.create_dataset(name=p, data=data.astype(bytes), maxshape=maxshape)

    def move(self, src: str, dest: str) -> None:
        self._group_obj.move(src, dest)

    def __delitem__(self, k: str) -> None:
        del self._group_obj[k]

    def __getitem__(self, p: str) -> np.ndarray:
        array = self._group_obj[p][()]
        assert isinstance(array, np.ndarray)
        return array

    def __len__(self) -> int:
        return len(self._group_obj)

    def __iter__(self) -> Iterator[str]:
        yield from self._group_obj.keys()
