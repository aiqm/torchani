import typing as tp
import inspect
import math
import re
from os import fspath
from pathlib import Path
from pprint import pformat
from functools import partial, wraps
from contextlib import ExitStack, contextmanager
from collections import OrderedDict

import torch
from torch import Tensor
import numpy as np
from numpy.typing import DTypeLike, NDArray
from tqdm import tqdm

from torchani.constants import ATOMIC_NUMBER, PERIODIC_TABLE
from torchani.utils import species_to_formula, sort_by_atomic_num, PADDING, ATOMIC_KEYS
from torchani.annotations import (
    Conformers,
    Backend,
    Grouping,
    NumpyConformers,
    MixedConformers,
    StrPath,
    IdxLike,
)
from torchani.datasets.backends import (
    Store,
    create_store,
    _ConformerWrapper,
)

# About _ELEMENT_KEYS:
# Datasets are assumed to have a "numbers" or "species" property, which has
# information about the elements. In the legacy format it may have either
# atomic numbers (1, 6, 8, etc) or strings with the chemical symbols ("H", "C",
# "O", etc), both are allowed for both names. In the new formats they **must be
# integers** If both properties are present one should be deleted to avoid
# redundancies

_ELEMENT_KEYS = {"species", "numbers", "atomic_numbers"}
_LEGACY_NONBATCH_KEYS = {"species", "numbers", "smiles", "atomic_numbers", "lot"}
_ALWAYS_STRING_PATTERNS = {
    r"^_id$",
    r"^smiles_.*",
    r"^inchis_.*",
    r"^lot$",
    r"^smiles$",
}
# These broken keys are in some datasets and are basically impossible to parse
# correctly. If grouping is "legacy" and these are found we give up and ask the
# user to delete them in a warning
_LEGACY_BROKEN_KEYS = {"coordinatesHE", "energiesHE", "smiles"}
# The second dimension of these keys is the number of atoms in the structure


# Helper functions
def _get_any_element_key(properties: tp.Iterable[str]):
    properties = {properties} if isinstance(properties, str) else set(properties)
    if "species" in properties:
        return "species"
    else:
        try:
            return next(iter(_ELEMENT_KEYS & properties))
        except StopIteration:
            raise ValueError(
                "Either species or numbers must be present in conformers"
            ) from None


def _get_formulas(conformers: NumpyConformers) -> tp.List[str]:
    elements = conformers[_get_any_element_key(conformers.keys())]
    if issubclass(elements.dtype.type, np.integer):
        elements = _numbers_to_symbols(elements)
    return species_to_formula(tp.cast(NDArray[np.str_], elements))


def _get_dim_size(
    conformers: NumpyConformers, common_keys: tp.Set[str], dim: int
) -> int:
    # Tries to get dimension size from one of the "common keys" that have the dimension
    present_keys = common_keys.intersection(conformers.keys())
    try:
        any_key = present_keys.pop()
    except KeyError:
        raise KeyError(
            f"Could not get size of dim {dim} in properties"
            f" since {common_keys} are missing from conformers"
        )
    return conformers[any_key].shape[dim]


# calculates number of atoms / conformers in a conformer group
_get_num_atoms = partial(
    _get_dim_size, common_keys={"coordinates", "coord", "forces"}, dim=1
)
_get_num_conformers = partial(
    _get_dim_size, common_keys={"coordinates", "coord", "forces", "energies"}, dim=0
)


def _to_strpath_list(obj: tp.Union[tp.Iterable[StrPath], StrPath]) -> tp.List[StrPath]:
    try:
        # This will raise an exception if obj is tp.Iterable[StrPath]
        list_ = [fspath(obj)]  # type: ignore
    except TypeError:
        list_ = [o for o in obj]  # type: ignore
    return tp.cast(tp.List[StrPath], list_)


# convert to / from symbols and atomic numbers properties
_symbols_to_numbers = np.vectorize(lambda x: ATOMIC_NUMBER[x])
_numbers_to_symbols = np.vectorize(lambda x: PERIODIC_TABLE[x])


# Base class for ANIDataset and _ANISubdataset
class _ANIDatasetBase(tp.Mapping[str, Conformers]):
    def __init__(self, *args, **kwargs) -> None:
        # "properties" is read only, needed for validation of inputs, it may
        # change if a property is renamed or deleted. num_conformers and
        # num_conformer_groups are all calculated on the fly to guarantee
        # synchronization with "group_sizes".
        self._group_sizes: tp.OrderedDict[str, int] = OrderedDict()
        self._properties: tp.Set[str] = set()

    @property
    def group_sizes(self) -> tp.OrderedDict[str, int]:
        return self._group_sizes.copy()

    @property
    def properties(self) -> tp.Set[str]:
        return self._properties

    @property
    def tensor_properties(self) -> tp.Tuple[str, ...]:
        set_ = {
            p
            for p in self._properties
            if not any(re.match(pattern, p) for pattern in _ALWAYS_STRING_PATTERNS)
        }
        return tuple(sorted(set_))

    @property
    def num_conformers(self) -> int:
        return sum(self._group_sizes.values())

    @property
    def num_conformer_groups(self) -> int:
        return len(self._group_sizes.keys())

    @property
    def grouping(self) -> tp.Union[Grouping, tp.Literal["legacy"]]:
        raise NotImplementedError

    def __getitem__(self, key: str) -> Conformers:
        return tp.cast(Conformers, getattr(self, "get_conformers")(key))

    def __len__(self) -> int:
        return self.num_conformer_groups

    def __iter__(self) -> tp.Iterator[str]:
        return iter(self._group_sizes.keys())

    def numpy_items(
        self, limit: float = math.inf, **kwargs
    ) -> tp.Iterator[tp.Tuple[str, NumpyConformers]]:
        count = 0
        for group_name in self.keys():
            count += 1
            yield group_name, getattr(self, "get_numpy_conformers")(
                group_name, **kwargs
            )
            if count >= limit:
                return

    def numpy_values(self, **kwargs) -> tp.Iterator[NumpyConformers]:
        for k, v in self.numpy_items(**kwargs):
            yield v

    def chunked_items(
        self, max_size: int = 2500, limit: float = math.inf, **kwargs
    ) -> tp.Iterator[tp.Tuple[str, int, MixedConformers]]:
        r"""Sequentially iterate over chunked pieces of the dataset with a maximum size

        The iteration is "chunked" into pieces, so instead of yielding groups
        this yields chunks of max size "max_size" which may be useful e.g.
        if groups are too large and they don't fit in GPU memory.

        The minimum size can't be controlled, and chunks may have different
        sizes in general, but they will not exceed max_size. An estimate of
        the number of chunks of the whole dataset is num_conformers //
        max_size.

        "limit" limits the number of output chunks to that number and then
        stops iteration (iteration is still sequential, not random).

        The second element in the yielded tuple is the cumulative conformer count
        previous to the yielded tuple.
        """
        getter = kwargs.pop("getter", "get_conformers")
        chunk_count = 0
        for group_name in self.keys():
            cumulative_conformer_count = 0
            conformers = getattr(self, getter)(group_name, **kwargs)
            any_key = next(iter(conformers.keys()))
            keys_copy = list(conformers.keys())
            splitted_conformers: NumpyConformers = dict()
            for k in keys_copy:
                if getter == "get_conformers":
                    splits = torch.split(conformers.pop(k), max_size)
                else:
                    splits = tuple(
                        conformers[k][j:j + max_size]
                        for j in range(0, len(conformers[k]), max_size)
                    )
                splitted_conformers.update({k: splits})
            num_chunks = len(splitted_conformers[any_key])
            for j in range(num_chunks):
                chunk_count += 1
                chunk_to_yield = {k: v[j] for k, v in splitted_conformers.items()}
                yield group_name, cumulative_conformer_count, chunk_to_yield
                cumulative_conformer_count += len(chunk_to_yield[any_key])
                if chunk_count >= limit:
                    return

    def chunked_numpy_items(
        self, **kwargs
    ) -> tp.Iterator[tp.Tuple[str, int, MixedConformers]]:
        kwargs.update({"getter": "get_numpy_conformers"})
        yield from self.chunked_items(**kwargs)

    def iter_key_idx_conformers(
        self, limit: float = math.inf, **kwargs
    ) -> tp.Iterator[tp.Tuple[str, int, Conformers]]:
        kwargs = kwargs.copy()
        getter = kwargs.pop("getter", "get_conformers")
        count = 0
        for k, size in self._group_sizes.items():
            conformers = getattr(self, getter)(k, **kwargs)
            for idx in range(size):
                count += 1
                single_conformer = {k: conformers[k][idx] for k in conformers.keys()}
                yield k, idx, single_conformer
                if count >= limit:
                    return

    def iter_key_idx_numpy_conformers(
        self, **kwargs
    ) -> tp.Iterator[tp.Tuple[str, int, Conformers]]:
        kwargs.update({"getter": "get_numpy_conformers"})
        yield from self.iter_key_idx_conformers(**kwargs)

    def iter_conformers(self, **kwargs) -> tp.Iterator[Conformers]:
        for _, _, c in self.iter_key_idx_conformers(**kwargs):
            yield c

    def iter_numpy_conformers(self, **kwargs) -> tp.Iterator[Conformers]:
        for _, _, c in self.iter_key_idx_numpy_conformers(**kwargs):
            yield c


# Decorators for ANISubdataset:
# Decorator that wraps functions that modify the dataset in place. Makes
# sure that cache updating happens after dataset modification
def _needs_cache_update(
    method: tp.Callable[..., "_ANISubdataset"]
) -> tp.Callable[..., "_ANISubdataset"]:
    @wraps(method)
    def method_with_cache_update(
        ds: "_ANISubdataset", *args, **kwargs
    ) -> "_ANISubdataset":
        ds = method(ds, *args, **kwargs)
        ds._update_cache()
        return ds

    return method_with_cache_update


# methods marked with this decorator
# should be called on all subdatasets
def _broadcast(method):
    method._mark = "broadcast"
    return method


# methods marked with this decorator
# should be delegated to one subdataset
def _delegate(method):
    method._mark = "delegate"
    return method


# methods marked with this decorator
# should be delegated to one subdataset, and return a value != "self"
def _delegate_with_return(method):
    method._mark = "delegate_with_return"
    return method


# Private wrapper over backing storage, with some modifications it could be
# used for directories with npz files, or other backends. It should never ever
# be used directly by user code.
class _ANISubdataset(_ANIDatasetBase):
    def __init__(
        self,
        store_location: StrPath,
        grouping: tp.Optional[Grouping] = None,
        backend: tp.Optional[Backend] = None,
        verbose: bool = True,
        dummy_properties: tp.Optional[tp.Dict[str, tp.Any]] = None,
    ):
        # dummy_properties must be a dict of the form {'name': {'dtype': dtype,
        # 'is_atomic': is_atomic, 'extra_dims': extra_dims, 'fill_value':
        # fill_value}, ...} with one or more dummy properties. These will be
        # created on the fly only if they are not present in the dataset
        # already.
        super().__init__()
        self._store = create_store(
            store_location,
            backend,
            grouping,
            dummy_properties,
        )
        # StoreFactory monkey patches all stores with "backend" attribute
        self._possible_nonbatch_properties: tp.Set[str]

        if self.grouping not in ["by_formula", "by_num_atoms", "legacy"]:
            raise RuntimeError(f"Read with unsupported grouping {self.grouping}")
        if self.grouping == "legacy":
            self._possible_nonbatch_properties = _LEGACY_NONBATCH_KEYS
        else:
            self._possible_nonbatch_properties = set()
        # In general properties of the dataset should be equal for all
        # groups, this can be an issue for HDF5. We check this in the first
        # call of _update_cache, if it isn't we raise an exception
        self._update_cache(check_properties=True, verbose=verbose)
        if self.grouping == "legacy":
            if self.properties & _LEGACY_BROKEN_KEYS:
                import warnings

                warnings.warn(
                    f"{_LEGACY_BROKEN_KEYS & self.properties}"
                    " found in legacy dataset, this will generate"
                    " unpredictable issues.\n"
                    " Probably .items() and .values() will work but"
                    " not much else. It is highly  recommended that"
                    " you backup these properties (if needed) and"
                    " *delete them* using dataset.delete_properties"
                )

    @contextmanager
    def keep_open(self, mode: str = "r") -> tp.Iterator["_ANISubdataset"]:
        r"""Context manager to keep dataset open while iterating over it

        This speeds up access in the context of many operations in a block,
        Iterating in this context may be much faster than directly iterating
        over conformers

        Usage:
        with ds.keep_open('r') as ro_ds:
            for c in ro_ds.iter_conformers():
                print(c)
                ... etc


        Note: for parquet datasets append operations are queued while the dataset
        is open and are only executed once it is closed, so calling append_conformers
        inside a "keep_open" should be done with care.
        """
        try:
            self._store.open_meta_and_data(mode)
            yield self
        finally:
            self._store.close_meta_and_data()

    # This trick makes methods fetch the open file directly
    # if they are being called from inside a "keep_open" context
    def _get_open_store(
        self, stack: ExitStack, mode: str = "r", only_meta_needed: bool = False
    ) -> Store:
        if mode not in ["r+", "r"]:
            raise ValueError(f"Unsupported mode {mode}")

        try:
            _, store_mode = self._store.get_data()
        except Exception:
            return stack.enter_context(self._store.open(mode, only_meta_needed))

        if mode == "r+" and store_mode == "r":
            raise RuntimeError(
                'Tried to open a store with mode "r+" but'
                ' the store open with mode "r"'
            )
        return self._store

    def _update_cache(
        self, check_properties: bool = False, verbose: bool = True
    ) -> None:
        with ExitStack() as stack:
            store = self._get_open_store(stack, "r")
            self._group_sizes, self._properties = store.update_cache(
                check_properties, verbose
            )

    def __str__(self) -> str:
        str_ = f"ANI {self._store.backend} store:\n"
        d: tp.Dict[str, tp.Any] = {"Conformers": f"{self.num_conformers:,}"}
        d.update({"Conformer groups": self.num_conformer_groups})
        d.update({"Properties": sorted(self.properties)})
        return str_ + pformat(d)

    @property
    def symbols(self) -> tp.Tuple[str, ...]:
        self._check_correct_grouping()
        element_key = _get_any_element_key(self.properties)
        present: tp.Set[str] = set()
        for group_name in self.keys():
            conformers = self.get_numpy_conformers(
                group_name, properties=element_key, chem_symbols=True
            )
            present.update(conformers[element_key].ravel())
        return sort_by_atomic_num(present)

    @property
    def znumbers(self) -> tp.Tuple[int, ...]:
        r"""Get an ordered list with all elements present in the dataset
        list is ordered alphabetically. Function raises ValueError if neither
        'species' or 'numbers' properties are present.
        """
        self._check_correct_grouping()
        element_key = _get_any_element_key(self.properties)
        present: tp.Set[int] = set()
        for group_name in self.keys():
            conformers = self.get_numpy_conformers(
                group_name, properties=element_key, chem_symbols=False
            )
            present.update(conformers[element_key].ravel())
        return tuple(sorted(present))

    def _parse_index(self, idx: IdxLike) -> tp.Optional[NDArray[np.int64]]:
        # internally, idx_ is always a numpy array or None, idx can be a tensor
        # or a list or other iterable, which is must be castable to a numpy int
        # array of ndim 1
        if idx is not None:
            if isinstance(idx, Tensor):
                idx_ = idx.cpu().numpy()
            elif isinstance(idx, int):
                idx_ = np.array(idx)
            else:
                idx_ = np.asarray(idx)
            if idx_.ndim > 1:
                raise ValueError("index must be a 0 or 1 dim tensor")
            return idx_
        return idx

    @_delegate_with_return
    def get_conformers(
        self,
        group_name: str,
        idx: IdxLike = None,
        properties: tp.Optional[tp.Iterable[str]] = None,
    ) -> Conformers:
        r"""Get conformers in a given group in the dataset

        Can obtain conformers with specified indices, and including only
        specified properties. Conformers are dict of the form {property:
        Tensor}, where properties are strings"""
        numpy_conformers = self.get_numpy_conformers(group_name, idx, properties)
        return {
            k: torch.as_tensor(numpy_conformers[k])
            for k in set(numpy_conformers.keys())
            if not any(re.match(pattern, k) for pattern in _ALWAYS_STRING_PATTERNS)
        }

    @_delegate_with_return
    def get_numpy_conformers(
        self,
        group_name: str,
        idx: IdxLike = None,
        properties: tp.Optional[tp.Iterable[str]] = None,
        chem_symbols: bool = False,
        exclude_dummy: bool = False,
    ) -> NumpyConformers:
        r"""Same as get_conformers but conformers are a dict {property: ndarray}"""
        if properties is None:
            properties = self.properties
        needed_properties = (
            {properties} if isinstance(properties, str) else set(properties)
        )
        self._check_properties_are_present(needed_properties)
        nonbatch_properties = needed_properties & self._possible_nonbatch_properties
        batch_properties = needed_properties - self._possible_nonbatch_properties
        with ExitStack() as stack:
            f = self._get_open_store(stack, "r")
            if exclude_dummy:
                needed_properties = needed_properties - set(f._dummy_properties.keys())
            numpy_conformers = {p: f[group_name][p] for p in needed_properties}
        idx_ = self._parse_index(idx)
        if idx_ is not None:
            numpy_conformers.update(
                {k: numpy_conformers[k][idx_] for k in batch_properties}
            )
        # Nonbatch properties, if present, need tiling in the first dim
        if nonbatch_properties:
            tile_shape: tp.Tuple[int, ...]
            if idx_ is None or idx_.ndim == 1:
                tile_shape = (_get_num_conformers(numpy_conformers), 1)
            else:
                tile_shape = (1,)
            numpy_conformers.update(
                {
                    k: np.tile(numpy_conformers[k], tile_shape)
                    for k in nonbatch_properties
                    if numpy_conformers[k].ndim == 1
                }
            )
        # Depending on "chem_symbols", "species" / "numbers" are returned as
        # int64 or as str. In legacy grouping "species" and "numbers" can be
        # str or ints themselves, so we check for that and convert.
        for k in needed_properties & _ELEMENT_KEYS:
            elements = numpy_conformers[k]
            if issubclass(elements.dtype.type, np.integer):
                if chem_symbols:
                    numpy_conformers[k] = _numbers_to_symbols(elements)
            else:
                elements = elements.astype(str)
                if not chem_symbols:
                    numpy_conformers[k] = _symbols_to_numbers(elements)
        for k in needed_properties:
            if any(re.match(pattern, k) for pattern in _ALWAYS_STRING_PATTERNS):
                numpy_conformers[k] = numpy_conformers[k].astype(str)
        return numpy_conformers

    # Convert a dict that maybe has some numpy arrays and / or some torch
    # tensors into a homogeneous dict with all numpy arrays.
    def _to_numpy_conformers(
        self, mixed_conformers: MixedConformers, allow_negative_indices: bool = False
    ) -> NumpyConformers:
        numpy_conformers: NumpyConformers = dict()
        properties = set(mixed_conformers.keys())
        for k in properties:
            # try to convert to numpy, failure means it is already an ndarray
            try:
                _array = mixed_conformers[k].detach().cpu().numpy()  # type: ignore
                numpy_conformers[k] = _array
            except AttributeError:
                numpy_conformers[k] = tp.cast(NDArray[tp.Any], mixed_conformers[k])
        for k in properties & _ELEMENT_KEYS:
            # try to interpret as numeric, failure means we should convert to ints
            try:
                if (mixed_conformers[k] <= 0).any():  # type: ignore
                    if not allow_negative_indices:
                        raise ValueError(f"{k} are atomic numbers, must be positive")
            except TypeError:
                numpy_conformers[k] = _symbols_to_numbers(mixed_conformers[k])
        return tp.cast(NumpyConformers, numpy_conformers)

    @_delegate
    @_needs_cache_update
    def append_conformers(
        self, group_name: str, conformers: MixedConformers
    ) -> "_ANISubdataset":
        r"""Attach a new set of conformers to the dataset.

        Conformers must be a dict {property: Tensor or ndarray}, and they must have the
        same properties that the dataset supports. Appending is only supported
        for grouping 'by_formula' or 'by_num_atoms'
        """
        numpy_conformers = self._to_numpy_conformers(conformers)
        self._check_append_input(group_name, numpy_conformers)
        with ExitStack() as stack:
            f = self._get_open_store(stack, "r+")
            wrapper = _ConformerWrapper(numpy_conformers)
            dummies = f._dummy_properties.copy()
            if dummies:
                # Trying to append to a dataset that has dummy properties
                # triggers the materialization of all dummy properties
                f._dummy_properties = dict()
                self._update_cache(verbose=False)
                for k, v in dummies.items():
                    self.create_full_property(k, **v)
            try:
                f[group_name] = wrapper
            except ValueError:
                f[group_name].append_conformers(wrapper)
        return self

    @_delegate
    @_needs_cache_update
    def delete_conformers(
        self, group_name: str, idx: IdxLike = None
    ) -> "_ANISubdataset":
        r"""Delete a given selected set of conformers"""
        self._check_correct_grouping()
        idx_ = self._parse_index(idx)
        all_conformers = self.get_numpy_conformers(group_name)
        with ExitStack() as stack:
            f = self._get_open_store(stack, "r+")
            del f[group_name]
            # if no index was specified delete everything
            if idx_ is None:
                return self
            good_conformers = {
                k: np.delete(all_conformers[k], obj=idx_, axis=0)
                for k in self.properties
            }
            if all(v.shape[0] == 0 for v in good_conformers.values()):
                # if we deleted everything in the group then just return,
                # otherwise we recreate the group using the good conformers
                return self
            f[group_name] = _ConformerWrapper(good_conformers)
        return self

    @_broadcast
    @_needs_cache_update
    def create_full_property(
        self,
        dest_key: str,
        is_atomic: bool = False,
        extra_dims: tp.Union[int, tp.Tuple[int, ...]] = tuple(),
        fill_value: int = 0,
        dtype: DTypeLike = np.int64,
    ) -> "_ANISubdataset":
        r"""Creates a property for all conformer groups

        Creates a property with a specified shape, dtype and fill value
        for all conformers in the dataset. Example usage:

        # shape (N,)
        ds.create_full_property('new', fill_value=0.0, dtype=np.float64)
        # shape (N, A)
        ds.create_full_property('new', is_atomic=True, fill_value=1, dtype=int)
        # shape (N, A, 3)
        ds.create_full_property('new', extra_dims=3, fill_value=0.0, dtype=np.float32)
        # shape (N, 3, 3)
        ds.create_full_property('new', extra_dims=(3, 3), fill_value=5, dtype=int)
        """
        extra_dims_ = (extra_dims,) if isinstance(extra_dims, int) else extra_dims
        self._check_properties_are_not_present(dest_key)
        with ExitStack() as stack:
            f = self._get_open_store(stack, "r+")
            if hasattr(f, "create_full_direct"):
                # mypy does not understand monkey patching
                f.create_full_direct(
                    dest_key,
                    is_atomic=is_atomic,
                    extra_dims=extra_dims,  # type: ignore
                    fill_value=fill_value,
                    dtype=dtype,
                    num_conformers=self.num_conformers,
                )
            else:
                for group_name in self.keys():
                    shape: tp.Tuple[int, ...] = (_get_num_conformers(f[group_name]),)
                    if is_atomic:
                        shape += (_get_num_atoms(f[group_name]),)
                    f[group_name][dest_key] = np.full(
                        shape + extra_dims_, fill_value, dtype
                    )
        return self

    def _attach_dummy_properties(self, dummy_properties: tp.Dict[str, tp.Any]) -> None:
        with ExitStack() as stack:
            f = self._get_open_store(stack, "r+", only_meta_needed=True)
            f._dummy_properties = dummy_properties

    @property
    def _dummy_properties(self) -> tp.Dict[str, tp.Any]:
        with ExitStack() as stack:
            dummy = self._get_open_store(
                stack, "r+", only_meta_needed=True
            )._dummy_properties
        return dummy

    @_broadcast
    @_needs_cache_update
    def to_backend(
        self,
        backend: tp.Optional[Backend] = None,
        dest_root: tp.Optional[StrPath] = None,
        verbose: bool = True,
        inplace: bool = False,
    ) -> "_ANISubdataset":
        r"""Transforms underlying store into a different format"""
        self._check_correct_grouping()

        if backend is None:
            backend = self._store.backend

        if inplace:
            assert dest_root is None
        elif dest_root is None:
            dest_root = Path(self._store.location.root).parent

        if self._store.backend == backend and backend != "h5py":
            return self

        grouping: Grouping = (
            "by_num_atoms" if self.grouping == "legacy" else self.grouping
        )
        new_ds = _ANISubdataset("tmp", grouping, backend, verbose=False)
        try:
            with new_ds.keep_open("r+") as rwds:
                for group_name, conformers in tqdm(
                    self.numpy_items(exclude_dummy=True),
                    total=self.num_conformer_groups,
                    desc=f"Converting to {backend}",
                    disable=not verbose,
                    leave=False,
                ):
                    # mypy doesn't know that @wrap'ed functions have __wrapped__
                    # attribute, and fixing this is ugly
                    rwds.append_conformers.__wrapped__(  # type: ignore
                        rwds,
                        group_name,
                        conformers,
                    )
            new_ds._attach_dummy_properties(self._dummy_properties)
        except Exception:
            new_ds._store.location.clear()
            raise

        if inplace:
            new_ds._store.overwrite(self._store)
            return new_ds

        new_parent = Path(tp.cast(StrPath, dest_root)).resolve()
        new_ds._store.location.root = (
            new_parent / self._store.location.root.with_suffix("").name
        )
        return self

    @_broadcast
    @_needs_cache_update
    def repack(self, verbose: bool = True) -> "_ANISubdataset":
        r"""Repacks underlying store if it is HDF5

        When a dataset is deleted from an HDF5 file the file size is not
        reduced since unlinked data is still kept in the file. Repacking is
        needed in order to reduce the size of the file. Note that this is only
        useful for the h5py backend, otherwise it is a no-op.
        """
        return self.to_backend.__wrapped__(  # type: ignore
            self,
            verbose=verbose,
            inplace=True,
        )

    @_broadcast
    @_needs_cache_update
    def regroup_by_formula(
        self, repack: bool = True, verbose: bool = True
    ) -> "_ANISubdataset":
        r"""Regroup dataset by formula

        All conformers are extracted and redistributed in groups named
        'C8H5N7', 'C10O3' etc, depending on the formula. Conformers in
        different stores are not mixed. See the 'repack' method for an
        explanation of that argument.
        """
        self._check_unique_element_key()
        new_ds = _ANISubdataset("tmp", "by_formula", self._store.backend, verbose=False)
        try:
            with new_ds.keep_open("r+") as rwds:
                for group_name, conformers in tqdm(
                    self.numpy_items(exclude_dummy=True),
                    total=self.num_conformer_groups,
                    desc="Regrouping by formulas",
                    disable=not verbose,
                    leave=False,
                ):
                    # Get all formulas in the group to discriminate conformers by
                    # formula and then attach conformers with the same formula to the
                    # same groups
                    formulas = np.asarray(_get_formulas(conformers))
                    unique_formulas = np.unique(formulas)
                    formula_idxs = (
                        (formulas == el).nonzero()[0] for el in unique_formulas
                    )

                    for formula, idx in zip(unique_formulas, formula_idxs):
                        selected_conformers = {k: v[idx] for k, v in conformers.items()}
                        rwds.append_conformers.__wrapped__(  # type: ignore
                            rwds,
                            formula,
                            selected_conformers,
                        )
            new_ds._attach_dummy_properties(self._dummy_properties)
        except Exception:
            new_ds._store.location.clear()
            raise

        new_ds._store.overwrite(self._store)
        if repack:
            new_ds._update_cache()
            return new_ds.repack.__wrapped__(new_ds, verbose=verbose)  # type: ignore
        return new_ds

    @_broadcast
    @_needs_cache_update
    def regroup_by_num_atoms(
        self, repack: bool = True, verbose: bool = True
    ) -> "_ANISubdataset":
        r"""Regroup dataset by number of atoms

        All conformers are extracted and redistributed in groups named
        'num_atoms_10', 'num_atoms_8' etc, depending on the number of atoms.
        Conformers in different stores are not mixed. See the 'repack' method
        for an explanation of that argument.
        """
        self._check_unique_element_key()
        new_ds = _ANISubdataset(
            "tmp",
            "by_num_atoms",
            self._store.backend,
            verbose=False
        )
        try:
            with new_ds.keep_open("r+") as rwds:
                for group_name, conformers in tqdm(
                    self.numpy_items(exclude_dummy=True),
                    total=self.num_conformer_groups,
                    desc="Regrouping by number of atoms",
                    disable=not verbose,
                    leave=False,
                ):
                    # This is done to accomodate the current group convention
                    new_name = str(_get_num_atoms(conformers)).zfill(3)
                    rwds.append_conformers.__wrapped__(  # type: ignore
                        rwds,
                        new_name,
                        conformers,
                    )
            new_ds._attach_dummy_properties(self._dummy_properties)
        except Exception:
            new_ds._store.location.clear()
            raise

        new_ds._store.overwrite(self._store)
        if repack:
            new_ds._update_cache()
            return new_ds.repack.__wrapped__(new_ds, verbose=verbose)  # type: ignore
        return new_ds

    @_broadcast
    @_needs_cache_update
    def delete_properties(
        self, properties: tp.Iterable[str], verbose: bool = True
    ) -> "_ANISubdataset":
        r"""Delete some properties from the dataset"""
        properties = {properties} if isinstance(properties, str) else set(properties)
        self._check_properties_are_present(properties)
        with ExitStack() as stack:
            f = self._get_open_store(stack, "r+")

            for property_ in properties.copy():
                if property_ in f._dummy_properties.keys():
                    f._dummy_properties.pop(property_)
                    properties.remove(property_)
            if hasattr(f, "delete_direct"):
                # mypy does not understand monkey patching
                f.delete_direct(properties)  # type: ignore
            else:
                for group_key in tqdm(
                    self.keys(),
                    total=self.num_conformer_groups,
                    desc="Deleting properties",
                    disable=not verbose,
                    leave=False,
                ):
                    for property_ in properties:
                        del f[group_key][property_]
                    if not f[group_key].keys():
                        del f[group_key]
        return self

    @_broadcast
    @_needs_cache_update
    def rename_properties(self, old_new_dict: tp.Dict[str, str]) -> "_ANISubdataset":
        r"""Rename some properties from the dataset

        Expects a dictionary of the form: {old_name: new_name}
        """
        self._check_properties_are_present(old_new_dict.keys())
        self._check_properties_are_not_present(old_new_dict.values())
        with ExitStack() as stack:
            f = self._get_open_store(stack, "r+")

            for old_name, new_name in old_new_dict.copy().items():
                if old_name in f._dummy_properties.keys():
                    f._dummy_properties[new_name] = f._dummy_properties.pop(old_name)
                    old_new_dict.pop(old_name)

            if hasattr(f, "rename_direct"):
                # mypy does not understand monkey patching
                f.rename_direct(old_new_dict)  # type: ignore
            else:
                for k in self.keys():
                    for old_name, new_name in old_new_dict.items():
                        f[k].move(old_name, new_name)
        return self

    @property
    def grouping(self) -> tp.Union[Grouping, tp.Literal["legacy"]]:
        r"""Get the dataset grouping

        Grouping is a string that describes how conformers are grouped in
        hierarchical datasets. Can be one of 'by_formula', 'by_num_atoms', 'legacy'.
        """
        with ExitStack() as stack:
            grouping = self._get_open_store(stack, "r", only_meta_needed=True).grouping
        return grouping

    def _check_unique_element_key(
        self, properties: tp.Optional[tp.Iterable[str]] = None
    ) -> None:
        if properties is None:
            properties = self.properties
        else:
            properties = (
                {properties} if isinstance(properties, str) else set(properties)
            )
        if len(properties.intersection(_ELEMENT_KEYS)) > 1:
            raise ValueError(
                f"There can be at most one of {_ELEMENT_KEYS}"
                f" present, but found {set(properties)}"
            )

    def _check_correct_grouping(self) -> None:
        if self.grouping not in ["by_formula", "by_num_atoms"]:
            calling_fn_name = inspect.stack()[1][3]
            raise ValueError(
                f"Can't use the function {calling_fn_name}"
                " if the grouping is not by_formula or"
                f" by_num_atoms. Grouping is {self.grouping}."
                " Please regroup your dataset"
            )

    def _check_append_input(self, group_name: str, conformers: NumpyConformers) -> None:
        self._check_correct_grouping()
        conformers_properties = set(conformers.keys())
        self._check_unique_element_key(conformers_properties)
        # check that all formulas are the same
        if self.grouping == "by_formula":
            if len(set(_get_formulas(conformers))) > 1:
                raise ValueError("All appended conformers must have the same formula")
        # If this is the first conformer added update the dataset to support
        # these properties, otherwise check that all properties are present
        if not self.properties:
            self._properties = conformers_properties
        elif not self.properties == conformers_properties:
            raise ValueError(
                f"Expected {self.properties} but got {conformers_properties}"
            )
        if "/" in group_name:
            raise ValueError('Character "/" not supported in group_name')
        # All properties must have the same batch dimension
        size = conformers[next(iter(conformers_properties))].shape[0]
        if not all(conformers[k].shape[0] == size for k in self.properties):
            raise ValueError(
                f"All batch keys {self.properties} must have the same batch dimension"
            )

    def _check_properties_are_present(self, properties: tp.Iterable[str]) -> None:
        properties = {properties} if isinstance(properties, str) else set(properties)
        if not properties <= self.properties:
            raise ValueError(
                f"Some of the properties requested {properties} are not"
                f" in the dataset, which has properties {self.properties}"
            )

    def _check_properties_are_not_present(self, properties: tp.Iterable[str]) -> None:
        properties = {properties} if isinstance(properties, str) else set(properties)
        if properties <= self.properties:
            raise ValueError(
                f"Requested properties: {properties} should not be in the dataset."
                f" which has properties: {self.properties}."
            )


# ANIDataset implementation details:
#
# ANIDataset is a mapping, The mapping has keys "group_names" and,
# values "conformers" or "conformer_group". Each group of conformers is also a
# mapping, where keys are "properties" and values are numpy arrays / torch
# tensors (they are just referred to as "values" or "data").
#
# In the current HDF5 datasets the group names are formulas (in some
# CCCCHHH.... etc, in others C2H4, etc) groups could also be smiles or number
# of atoms. Since HDF5 is hierarchical this grouping is essentially hardcoded
# into the dataset format.
#
# To parse all current HDF5 dataset types it is necessary to first determine
# where all the conformer groups are. HDF5 has directory structure, and in
# principle they could be arbitrarily located. One would think that there is
# some sort of standarization between the datasets, but unfortunately there is
# none (!!), and the legacy reader, anidataloader, just scans all the groups
# recursively...
#
# Cache update part 1:
# --------------------
# Since scanning recursively is super slow we just do this once and cache the
# location of all the groups, and the sizes of all the groups inside
# "groups_sizes". After this, it is not necessary to do the recursion again
# unless some modification to the dataset happens, in which case we need a
# cache update, to get "group_sizes" and "properties" again. Methods that
# modify the dataset are decorated so that the internal cache is updated.
#
# Cache update part 2:
# --------------------
# There is in principle no guarantee that all conformer groups have the same
# properties. Due to this we have to first traverse the dataset and check that
# this is the case. We do this only once and then we store the properties the
# dataset supports inside an internal variable _properties (e.g. it may happen
# that one molecule has forces but not coordinates, if this happens then
# ANIDataset raises an error), which gets updated if a property changes.
#
# Multiple files:
# ---------------
# Current datasets need ANIDataset to be able to manage multiple files, this is
# achieved by delegating execution of the methods to one of the _ANISubdataset
# instances that ANIDataset
# contains. Basically any method is either:
# 1 - delegated to a subdataset: If you ask for the conformer group "ANI1x/CH4"
#     in the "full ANI2x" dataset (ANI1x + ANI2x_FSCl + dimers),
#     then this will be delegated to the "ANI1x" subdataset. Methods that take
#     a group_name parameter are delegated to subdatasets.
# 2 - broadcasted to all subdatasets: e.g. If you want to rename a property or
#     delete a property it will be deleted / renamed in all subdatasets.
# The mechanism for delegation involves overriding __getattr__.
#
# ContextManager usage:
# ----------------
# You can turn the dataset into a context manager that keeps all stores
# open simultaneously by using 'with ds.keep_open('r') as ro_ds:', for example.
# It seems that HDF5 is quite slow when opening files, it has to aqcuire locks,
# and do other things, so this speeds up iteration by 12 - 13 % usually. Since
# many files may need to be opened at the same time then ExitStack is needed to
# properly clean up everything. Each time a method needs to open a file it first
# checks if it is already open (i.e. we are inside a 'keep_open' context) in that
# case it just fetches the already opened file.
class ANIDataset(_ANIDatasetBase):
    r"""Dataset that supports multiple stores and manages them as one single entity.

    Datasets have a "grouping" for the different conformers, which can be
    "by_formula", "by_num_atoms", "legacy".
    Regrouping to one of the standard groupings can be done using
    'regroup_by_formula' or 'regroup_by_num_atoms'.

    Conformers can be extracted as {property: Tensor} or {property: ndarray}
    dicts, and can also be appended or deleted from the backing stores.

    All conformers in a datasets must have the same properties and the first
    dimension in all Tensors/arrays is the same for all conformer groups (it is
    the batch dimension). Property manipulation (renaming, deleting, adding)
    is also supported.
    """

    def __init__(
        self,
        locations: tp.Union[tp.Iterable[StrPath], StrPath],
        names: tp.Optional[tp.Union[tp.Iterable[str], str]] = None,
        **kwargs,
    ):
        super().__init__()
        # _datasets is an OrderedDict {name: _ANISubdataset}.
        # "locations" and "names" are collections used to build it
        # First we convert locations / names into lists of strpath / str
        # if no names are provided they are just '0', '1', '2', etc.
        locations = _to_strpath_list(locations)
        if names is None:
            names = (str(j) for j in range(len(locations)))
        names = [names] if isinstance(names, str) else [n for n in names]
        if not len(names) == len(locations):
            raise ValueError("Length of locations and names must be equal")
        self._datasets = OrderedDict(
            (n, _ANISubdataset(loc, **kwargs)) for n, loc in zip(names, locations)
        )
        self._update_cache()

    @classmethod
    def from_dir(cls, dir_: StrPath, **kwargs):
        r"""
        Initializes datasets from all files in a given directory

        File backends are inferred from the suffixes. All files must have
        different names.
        """
        dir_ = Path(dir_).resolve()
        if not dir_.is_dir():
            raise ValueError("Input should be a directory")
        locations = sorted([p for p in dir_.iterdir() if p.suffix != ".tar.gz"])
        names = [p.stem for p in locations]
        return cls(locations=locations, names=names, **kwargs)

    @property
    def grouping(self) -> tp.Union[Grouping, tp.Literal["legacy"]]:
        return self._first_subds.grouping

    @contextmanager
    def keep_open(self, mode: str = "r") -> tp.Iterator["ANIDataset"]:
        with ExitStack() as stack:
            for k in self._datasets.keys():
                self._datasets[k] = stack.enter_context(
                    self._datasets[k].keep_open(mode)
                )
            yield self

    @property
    def symbols(self) -> tp.Tuple[str, ...]:
        return sort_by_atomic_num(
            {s for ds in self._datasets.values() for s in ds.symbols}
        )

    @property
    def znumbers(self) -> tp.Tuple[int, ...]:
        return tuple(sorted({s for ds in self._datasets.values() for s in ds.znumbers}))

    @property
    def store_locations(self) -> tp.List[str]:
        return [fspath(ds._store.location.root) for ds in self._datasets.values()]

    @property
    def num_stores(self) -> int:
        return len(self._datasets)

    def auto_append_conformers(
        self,
        conformers: MixedConformers,
        atomic_properties: tp.Iterable[str] = ATOMIC_KEYS,
        padding: int = int(PADDING["numbers"]),
    ) -> "ANIDataset":
        assert (
            self.num_stores == 1
        ), "Can currently only perform auto-appending for single store datasets"
        atomic_properties = set(atomic_properties)
        numpy_conformers = self._first_subds._to_numpy_conformers(
            conformers, allow_negative_indices=True
        )
        element_key = _get_any_element_key(numpy_conformers.keys())
        elements = numpy_conformers[element_key]
        groups: tp.Dict[str, tp.Any] = {}
        for j, znumbers in enumerate(elements):
            mask = znumbers != padding
            if self.grouping == "by_formula":
                group_key = species_to_formula(_numbers_to_symbols(znumbers[mask]))[0]
            elif self.grouping == "by_num_atoms":
                group_key = str(znumbers[mask].shape[0]).zfill(3)
            else:
                raise ValueError("Incorrect grouping")
            if group_key not in groups.keys():
                groups[group_key] = {k: [] for k in numpy_conformers}
            for k in numpy_conformers:
                if k in atomic_properties:
                    groups[group_key][k].append(numpy_conformers[k][j, mask])
                else:
                    groups[group_key][k].append(numpy_conformers[k][j])
        for j, (name, group) in enumerate(groups.items()):
            self.append_conformers(name, {k: np.asarray(v) for k, v in group.items()})
        return self

    # Mechanism for delegating calls to the correct _ANISubdatasets:
    # Functions with a "group_name" argument are delegated to one specific
    # subdataset, other ones are performed in all subdatasets on a loop
    # (broadcasted)
    def __getattr__(self, method: str) -> tp.Callable:
        unbound_method = getattr(_ANISubdataset, method)
        mark = unbound_method._mark
        if mark == "delegate":

            @wraps(unbound_method)
            def delegated_call(group_name: str, *args, **kwargs) -> "ANIDataset":
                name, k = self._parse_key(group_name)
                self._datasets[name] = getattr(self._datasets[name], method)(
                    k, *args, **kwargs
                )
                return self._update_cache()

        elif mark == "delegate_with_return":

            @wraps(unbound_method)
            def delegated_call(group_name: str, *args, **kwargs) -> tp.Any:
                name, k = self._parse_key(group_name)
                return getattr(self._datasets[name], method)(k, *args, **kwargs)

        elif mark == "broadcast":

            @wraps(unbound_method)
            def delegated_call(*args, **kwargs) -> "ANIDataset":
                for name in self._datasets.keys():
                    self._datasets[name] = getattr(self._datasets[name], method)(
                        *args, **kwargs
                    )
                return self._update_cache()

        else:
            raise AttributeError(
                "Attribute {unbound_method.__name__} can't be accessed by ANIDataset"
            )
        return delegated_call

    def __str__(self) -> str:
        return "\n".join(
            f"Name: {name}" + "\n" + str(ds) for name, ds in self._datasets.items()
        )

    @property
    def _first_name(self) -> str:
        return next(iter(self._datasets.keys()))

    @property
    def _first_subds(self) -> "_ANISubdataset":
        return next(iter(self._datasets.values()))

    def _update_cache(self) -> "ANIDataset":
        self._group_sizes = OrderedDict(
            (k if self.num_stores == 1 else f"{name}/{k}", v)
            for name, ds in self._datasets.items()
            for k, v in ds._group_sizes.items()
        )
        for name, ds in self._datasets.items():
            if not ds.grouping == self._first_subds.grouping:
                raise RuntimeError(
                    "Datasets have incompatible groupings,"
                    f" got {self._first_subds.grouping} for"
                    f" {self._first_name}"
                    f" and {ds.grouping} for {name}"
                )

            if not ds.properties == self._first_subds.properties:
                raise RuntimeError(
                    "Datasets have incompatible properties"
                    f" got {self._first_subds.properties} for"
                    f" {self._first_name}"
                    f" and {ds.properties} for {name}"
                )
        self._properties = self._first_subds.properties
        return self

    def _parse_key(self, key: str) -> tp.Tuple[str, str]:
        tokens = key.split("/")
        if self.num_stores == 1:
            return self._first_name, "/".join(tokens)
        return tokens[0], "/".join(tokens[1:])


def concatenate(
    source: ANIDataset,
    dest_location: StrPath,
    verbose: bool = True,
    backend: Backend = "hdf5",
    delete_originals: bool = False,
) -> ANIDataset:
    r"""Combine all the backing stores in a given ANIDataset into one"""
    dest_location = Path(dest_location).resolve()
    if source.grouping not in ["by_formula", "by_num_atoms"]:
        raise ValueError("Please regroup your dataset before concatenating")

    dest = ANIDataset("tmp", backend=backend, grouping=source.grouping, verbose=False)
    try:
        for k, v in tqdm(
            source.numpy_items(),
            desc="Concatenating datasets",
            total=source.num_conformer_groups,
            disable=not verbose,
        ):
            dest.append_conformers(k.split("/")[-1], v)
    except Exception:
        dest._first_subds._store.location.clear()

    dest._first_subds._store.location.root = dest_location
    if delete_originals:
        for subds in tqdm(
            source._datasets.values(),
            desc="Deleting original stores",
            total=source.num_stores,
            disable=not verbose,
        ):
            subds._store.location.clear()
    return dest
